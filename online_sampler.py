"""
online_sampler.py — streaming wrapper around the PlaiV1 sampling primitives.
 
It runs two cooperating pieces:
  1. INGEST: push_dataframe() is called ~every 200 ms with ONE encoded dataframe
     (video latents / key_press latents / normalized mouse — the same encoded
     format your .hdf5 + .db hold). These land in a pending buffer.
  2. GENERATE: a loop wakes every CADENCE_SECONDS, drains the pending buffer,
     and rolls the model forward GEN_SECONDS into the future, reusing the exact
     MemoryContext + generate_chunk used by offline sampling.
 
You supply: a live encoder that turns raw gameplay into encoded dataframes and
calls push_dataframe(). That encoder is NOT in this file.
 
Run from the repo root so `import`s resolve, with the same env vars + venv you
use for eval.
"""
import os, sys, time, threading
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
from pathlib import Path
 
# Make sure THIS repo's `src/` is importable, then import data_classes the SAME
# way the model + live_agent do — NON-prefixed (`data.*`, not `src.data.*`).
# The src/ layout means `src.data.data_classes` and `data.data_classes` load as
# two different modules with two different FullData classes; mixing them breaks
# cat_time's isinstance check. Everyone must use the same one.
_REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_REPO = Path(os.environ.get("PLAICRAFT_MODEL_REPO")).expanduser()
sys.path.append(MODEL_REPO)
 
from src.data.data_classes import FullData
from src.data.datamodule import DataModule
from src.inference.memory import MemoryContext
from src.inference.denoising import generate_chunk
 
 
# 1 dataframe = 200 ms  ->  5 dataframes / second
DATAFRAME_MS     = 200     # 1 unit = 200 ms = 2 video latents (10 fps)
GEN_SECONDS      = float(os.environ.get("PLAICRAFT_GEN_SECONDS", "2.0"))      # predict per tick
CADENCE_SECONDS  = float(os.environ.get("PLAICRAFT_CADENCE_SECONDS", "0.4"))  # how often to generate
CONTEXT_SECONDS  = 5.0     # warmup: how much history is required before the first generation
 
GEN_FRAMES     = round(GEN_SECONDS    * 1000 / DATAFRAME_MS)   # = 10
CONTEXT_FRAMES = round(CONTEXT_SECONDS * 1000 / DATAFRAME_MS)  # = 25
 
# PLAICRAFT_PROFILE=1 prints a per-chunk breakdown of where the time goes:
# memory/context-embedder (update_and_get_memory) vs decoder (generate_chunk).
_PROFILE = os.environ.get("PLAICRAFT_PROFILE", "0").lower() in ("1", "true", "yes")
# PLAICRAFT_TORCH_PROFILE=1 runs ONE generate_chunk under torch.profiler (after a
# couple of warmup gens) and prints the top CUDA ops, then never again. This names
# the exact kernel(s) eating the time instead of us guessing from the architecture.
_TORCH_PROFILE = os.environ.get("PLAICRAFT_TORCH_PROFILE", "0").lower() in ("1", "true", "yes")
 
 
load_dotenv()
 
 
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
 
 
def _gpu_util_sampler(stop_evt, out_list, interval=0.04):
    """Poll GPU utilization (%) via NVML while the decoder runs, in a background
    thread, so we can tell launch/sync-bound (low util) from compute-bound (high
    util) without a second terminal. Silently no-ops if pynvml is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop_evt.is_set():
            out_list.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            time.sleep(interval)
        pynvml.nvmlShutdown()
    except Exception:
        pass
 
 
def _cast_fd_float(fd, dtype):
    """Cast only the FLOAT modalities of a FullData to `dtype`, leaving
    dataframe_indices (long) and metadata untouched. Used to feed a bf16 model
    without corrupting the integer index tensor."""
    if dtype is None:
        return fd
    d = fd.to_dict()
    for k, v in list(d.items()):
        if torch.is_tensor(v) and v.is_floating_point():
            d[k] = v.to(dtype)
    return FullData(batch=d)
 
 
def _clone_state(obj):
    """Deep-clone tensors inside nested dict/list/tuple state (for snapshots)."""
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_state(v) for v in obj)
    return obj
 
 
_NOISE_PATCHED = False
 
 
def _patch_noise_dtype(dtype):
    """Make generate_chunk's noise target come out in `dtype` (e.g. bf16) without
    editing the model repo. generate_chunk calls the module-global init_noise_target,
    so wrapping that attribute on its own module is enough; the wrapper casts the
    float modalities (leaving dataframe_indices long)."""
    global _NOISE_PATCHED
    if _NOISE_PATCHED:
        return
    dn = sys.modules[generate_chunk.__module__]      # whichever module it came from
    orig = dn.init_noise_target
 
    def _init_noise_target(*args, **kwargs):
        return _cast_fd_float(orig(*args, **kwargs), dtype)
 
    dn.init_noise_target = _init_noise_target
    _NOISE_PATCHED = True
 
 
class OnlineSampler:
    # Plausible attribute names for the context embedder's streaming cache.
    # MemoryContext.reset() calls enable_streaming_cache(reset_cache=True), which
    # means the embedder holds state OUTSIDE MemoryContext. If the rollout's STM
    # passes (which see fed-back predictions) mutate that cache, it must be part
    # of the snapshot/restore or predictions leak into the next real tick.
    _EMB_CACHE_ATTRS = ("streaming_cache", "_streaming_cache", "stm_cache", "_cache")
 
    def __init__(self, model, inference_cfg, target_modalities, device="cuda"):
        self.model  = model
        self.infcfg    = OmegaConf.create({"inference": inference_cfg})
        self.device = device
        self.target_modalities = list(target_modalities)
 
        # Model compute dtype (bf16 if live_agent cast it). The noise target and the
        # context we feed must match this or the first Linear hits a dtype mismatch.
        self.model_dtype = next(self.model.parameters()).dtype
        print(f"[sampler] model dtype = {self.model_dtype}")
        if self.model_dtype != torch.float32:
            print(f"[sampler] patching noise target to {self.model_dtype}")
            _patch_noise_dtype(self.model_dtype)
 
        self.k        = int(self.model.context_embedder.ltm_downsample_chunk_length)
        self.stm_len  = int(self.model.context_embedder.stm_context_length)
        self.warmup_ltm_chunk_length = int(
            OmegaConf.select(self.infcfg, "inference.warmup_ltm_chunk_length") or self.k * 8
        )
        self.chunk_len = int(OmegaConf.select(self.infcfg, "inference.chunk_length") or 2)
 
        # ONE persistent memory, warmed up once then streamed incrementally
        # (NOT reset per tick). This is what keeps each step ~0.5 s instead of
        # reprocessing the whole context every time.
        self.memory = MemoryContext(self.model, self.stm_len, self.k,
                                    self.warmup_ltm_chunk_length)
        self._mem_ready = False
 
        # The real observed stream. Frames are normalized (dtype + device) ONCE,
        # at drain time, so nothing downstream re-casts the whole window per tick.
        # After each generation `real` is trimmed to the STM window; the trimmed
        # frames live on only inside memory.cached_ltm. Bounded as a safety net:
        # if generation stalls and real grows past _max_frames, keep the last
        # _keep_frames and re-warm memory from scratch (rare).
        self.real = None
        self._pending = []          # new frames since last generation (cheap append)
        self._max_frames  = CONTEXT_FRAMES * 20
        self._keep_frames = CONTEXT_FRAMES * 10
        self._lock  = threading.Lock()   # guards self._pending ONLY; self.real is
                                         # touched exclusively by the generate thread
        self._stop  = threading.Event()
        # Raised while a rollout is generating, so the ingest thread can skip
        # GPU encodes during that window instead of timeslicing the denoiser.
        self.busy   = threading.Event()
 
        self._warned_cache = False
        self._last_mem_ms = 0.0
 
    # ---------- INGEST (call from your encoder thread, ~every 200 ms) ----------
    def push_dataframe(self, frame: FullData):
        """`frame`: FullData with time-length 1, encoded modalities, on any device.
        O(1) append only — normalization and the memory fold happen later, in
        _generate_once, so ingestion never competes with generation."""
        with self._lock:
            self._pending.append(frame)
 
    # ---------- snapshot/restore the streaming memory ----------
    # A rollout folds its own fed-back predictions into memory so later chunks
    # can see earlier ones; snapshot/restore discards those folds afterwards so
    # the persistent memory stays on the real timeline.
    def _snapshot(self):
        ltm = self.memory.cached_ltm
        emb_cache = None
        for name in self._EMB_CACHE_ATTRS:
            if hasattr(self.model.context_embedder, name):
                emb_cache = (name, _clone_state(getattr(self.model.context_embedder, name)))
                break
        if emb_cache is None and not self._warned_cache:
            # reset() clears a streaming cache we can't find to snapshot. If the
            # embedder mutates it during STM passes, rollout predictions will
            # leak into the next tick's real context. Verify against the
            # embedder implementation and add the attribute name above.
            print("[sampler][warn] context_embedder streaming cache not found; "
                  "snapshot covers cached_ltm + watermark only.", flush=True)
            self._warned_cache = True
        return (ltm.clone() if ltm is not None else None,
                self.memory.processed_ltm_frames,
                emb_cache)
 
    def _restore(self, snap):
        ltm, watermark, emb_cache = snap
        self.memory.cached_ltm = ltm
        self.memory.processed_ltm_frames = watermark
        if emb_cache is not None:
            name, state = emb_cache
            setattr(self.model.context_embedder, name, state)
 
    @staticmethod
    def _time_length(fd):
        return int(fd.dataframe_indices.shape[1])
 
    # ---------- memory + decoder wrappers (profiling lives here) ----------
    def _mem_update(self, fd_idx):
        if _PROFILE:
            _sync(); t0 = time.time()
        mem = self.memory.update_and_get_memory(fd_idx)
        if _PROFILE:
            _sync(); self._last_mem_ms = (time.time() - t0) * 1000
        return mem
 
    def _run_generate(self, mem, cur_idx):
        """Single generate_chunk call. ONE code path shared by the normal case,
        the one-shot torch profiler, and the wall-clock profiler, so they can't
        drift apart."""
        def _call():
            return generate_chunk(
                model=self.model,
                config=self.infcfg.inference,
                memory=mem,
                batch_size=FullData.infer_batch_size(cur_idx),  # 1: single player
                target_pred_len=self.chunk_len,
                target_modalities=self.target_modalities,
                metadata=cur_idx.metadata,
            )
 
        # One-shot torch profiler: name the kernels eating the time.
        if _TORCH_PROFILE and not getattr(self, "_torch_prof_done", False):
            self._torch_prof_n = getattr(self, "_torch_prof_n", 0) + 1
            if self._torch_prof_n >= 3:                  # skip warmup gens
                from torch.profiler import profile, ProfilerActivity
                _sync()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as _prof:
                    pred = _call()
                _sync()
                self._torch_prof_done = True
                print("\n========== TORCH PROFILER (one generate_chunk) ==========", flush=True)
                print(_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20), flush=True)
                print("========================================================\n", flush=True)
                return pred
 
        if not _PROFILE:
            return _call()
 
        stop = threading.Event(); samples = []
        smp = threading.Thread(target=_gpu_util_sampler, args=(stop, samples), daemon=True)
        smp.start()
        _sync(); t0 = time.time()
        pred = _call()
        _sync(); t_gen = (time.time() - t0) * 1000
        stop.set(); smp.join(timeout=0.3)
        util = (f"gpu_util peak={max(samples)}% mean={sum(samples)//len(samples)}% (n={len(samples)})"
                if samples else "gpu_util n/a (pip install pynvml)")
        print(f"[profile] mem(context-embedder)={self._last_mem_ms:.0f}ms  "
              f"gen(decoder)={t_gen:.0f}ms  {util}", flush=True)
        return pred
 
    # ---------- GENERATE one rollout (incremental streaming memory) ----------
    @torch.no_grad()
    def _generate_once(self):
        # Drain newly-ingested frames and extend the real sequence by ONLY those
        # (O(new), not O(total)). Normalize dtype + device HERE, once per frame
        # batch, so `self.real` is always model-ready and never re-cast wholesale.
        with self._lock:
            pending, self._pending = self._pending, []
        if pending:
            add = FullData.cat_time(pending)
            add = _cast_fd_float(add, self.model_dtype).to(self.device)
            # CRITICAL: the training dataloader z-scores EVERY modality via
            # DataModule.normalize_full_data before the model sees it (see
            # datamodule.py: train/val/test loaders all wrap normalize_fn).
            # The live encoders produce RAW-space tensors (raw AE key latents,
            # raw pixel-delta mouse bins, raw scaled SDXL video latents, raw
            # Encodec audio), so they must be normalized HERE or the model is
            # conditioned off-distribution in every modality at once.
            add = DataModule.normalize_full_data(add)
            # normalize_full_data constructs its result with the DATAMODULE's
            # FullData class (data.data_classes), which is a different class
            # object from this module's import (plaicraft_model.src.data...)
            # under the src-layout — cat_time's isinstance check rejects it.
            # Rebuild with the local class.
            add = FullData(batch=add.to_dict())
            self.real = add if self.real is None else FullData.cat_time([self.real, add])
            n = self._time_length(self.real)
            if n > self._max_frames:                       # bound RAM; re-warm once
                self.real = FullData.slice_time(self.real, n - self._keep_frames, n)
                self._mem_ready = False                    # reset() below refolds all kept frames
 
        real = self.real
        if real is None:
            return None
        # First generation only: wait for a full warmup window, then prime the
        # streaming memory once. After warmup `real` is a bounded recent window
        # (smaller than CONTEXT_FRAMES), so we must NOT re-gate on its length.
        if not self._mem_ready:
            if self._time_length(real) < CONTEXT_FRAMES:
                return None
            self.memory.reset()
            self._mem_ready = True
 
        self.busy.set()
        if (os.environ.get("PLAICRAFT_DEBUG_KP", "0") == "1"
                and not getattr(self, "_dbg_kp_printed", False)):
            kp = real.key_press                      # (1, U, 10, 16), z-scored model space
            torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)
            unit = kp[0, -1].float().cpu()           # last (most recent) 200ms unit: (10, 16)
            print("[debug] context key_press shape:", tuple(kp.shape))
            print("[debug] last context unit (2 windows x (5,16)):\n", unit)
            print("[debug] per-channel mean:\n", unit.mean(dim=0))
            print("[debug] per-channel std :\n", unit.std(dim=0))
            self._dbg_kp_printed = True

        try:
            n = self._time_length(real)

            # Training was bf16-MIXED: fp32 weights/buffers + bf16 autocast
            # compute. Reproduce exactly that when weights are fp32 on CUDA;
            # if the model was cast to native bf16 (or runs on CPU), autocast
            # adds nothing, so it self-disables.
            import contextlib
            use_ac = (self.model_dtype == torch.float32
                      and str(self.device).startswith("cuda"))
            ac = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                  if use_ac else contextlib.nullcontext())
            with ac:
                # Fold any new real frames into the persistent LTM, then snapshot so
                # the rollout's fed-back predictions can be discarded afterwards.
                # NOTE the order: fold real -> snapshot -> rollout -> restore. Frames
                # arriving DURING the rollout sit safely in _pending until next tick.
                real_idx = DataModule.assign_indices(real, -n)
                mem = self._mem_update(real_idx)
                snap = self._snapshot()
    
                produced, cur, cur_idx = [], real, real_idx    # `cur` stays this-module's class
                for step in range(0, GEN_FRAMES, self.chunk_len):
                    if step > 0:
                        clen = self._time_length(cur)
                        # assign_indices builds the model's own FullData class; use it
                        # only to feed the model -- do NOT store it back into `cur`.
                        cur_idx = DataModule.assign_indices(cur, -clen)
                        # Fold the previously fed-back prediction chunk (transient;
                        # undone by _restore). step==0 reuses the pre-loop mem: cur
                        # is identical to real there, so calling again would just
                        # burn a full STM embedder pass for the same result.
                        mem = self._mem_update(cur_idx)
    
                    pred = self._run_generate(mem, cur_idx)
    
                    # Rebuild the predicted chunk as a LOCAL-class FullData (mirrors
                    # the offline sampler's _merge_generated): a slice of the local
                    # context is the template; target modalities overwritten.
                    clen = self._time_length(cur)
                    template = FullData.slice_time(cur, clen - self.chunk_len, clen)
                    merged = template.to_dict()
                    for m in self.target_modalities:
                        merged[m] = pred.get_modality(m)
                    pred_fd = FullData(batch=merged)
    
                    produced.append(pred_fd)
                    cur = FullData.cat_time([cur, pred_fd])
 
            # Discard the rollout's prediction-folding; keep only the real-timeline LTM.
            self._restore(snap)
 
            # Trim `real` down to the STM window: older frames are already folded
            # into cached_ltm, so keeping them only makes cat_time/assign_indices
            # O(history). CRITICAL: shift the fold watermark by the trim instead
            # of zeroing it. valid_ltm_len can extend PAST n - stm_len (the STM
            # tail is typically both folded into LTM and re-embedded as STM), so
            # "kept frames were never folded" is false — zeroing the watermark
            # re-folds the kept prefix next tick and duplicates content in the
            # LTM on every trim cycle. The trim is rounded down to a multiple of
            # k so LTM downsample-group boundaries stay aligned to the new start.
            keep = max(self.stm_len, self.chunk_len)
            n_real = self._time_length(self.real)
            trim = n_real - keep
            if trim > 0:
                trim = (trim // self.k) * self.k
            if trim > 0:
                self.real = FullData.slice_time(self.real, trim, n_real)
                self.memory.processed_ltm_frames = max(
                    0, self.memory.processed_ltm_frames - trim
                )
            # Everything above (context, memory, fed-back predictions) lives in
            # the model's z-scored space. Consumers (keypress AE decoder, mouse
            # pixel deltas, video VAE) expect RAW space — mirror decode.py and
            # invert the normalization on the way out.
            out = FullData.cat_time(produced)
            out = DataModule.denormalize_full_data(out)
            return FullData(batch=out.to_dict())    # rebuild as local class (see drain)
        finally:
            self.busy.clear()
 
    # ---------- LOOP: GEN_SECONDS every CADENCE_SECONDS ----------
    def run(self, on_output):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                out = self._generate_once()
                if out is not None:
                    on_output(out)        # consumer: decode key/mouse, log, act, ...
            except Exception:
                # One bad tick shouldn't kill the sampler thread; log and retry.
                import traceback
                traceback.print_exc()
            dt = time.time() - t0
            if dt > CADENCE_SECONDS:
                print(f"[warn] generation took {dt:.2f}s > {CADENCE_SECONDS}s budget "
                      f"-- reduce inference.num_denoising_steps or chunk_length, "
                      f"or you need a faster GPU.")
            time.sleep(max(0.0, CADENCE_SECONDS - dt))
 
    def stop(self):
        self._stop.set()
 
 
# --------------------------------------------------------------------------
# Example wiring (model loading mirrors eval.py: model=plai_v1_trained + ckpt)
# --------------------------------------------------------------------------
def load_model(ckpt_path, device="cuda"):
    import hydra
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        cfg = compose(config_name="eval", overrides=[
            "model=plai_v1_trained",
            "model.context_modalities=[video,audio_hear,audio_speak,key_press,mouse_movement]",
            "model.target_modalities=[video,audio_hear,audio_speak,key_press,mouse_movement]",
        ])
    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    # strip common training prefixes, then load non-strict (audio experts drop)
    clean = {}
    for kk, vv in sd.items():
        for p in ("module.model.", "model.", "module."):
            if kk.startswith(p):
                kk = kk[len(p):]; break
        clean[kk] = vv
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"loaded; {len(missing)} missing / {len(unexpected)} unexpected keys")

    return model, cfg.inference
 
 
if __name__ == "__main__":
    model, inf_cfg = load_model(os.path.expanduser(
        os.environ["PROJECT_ROOT"] + "/last_fp32/pytorch_model.bin"))
    # speed knobs for the real-time budget:
    inf_cfg = OmegaConf.merge(inf_cfg, {"num_denoising_steps": 1, "chunk_length": 1})
 
    sampler = OnlineSampler(model, inf_cfg,
                            target_modalities=["video", "key_press", "mouse_movement"])
 
    # def encoder_loop():
    #     while True:
    #         frame = encode_live_gameplay()   # -> FullData, time-length 1, encoded
    #         sampler.push_dataframe(frame)
    #         time.sleep(DATAFRAME_MS / 1000)
    # threading.Thread(target=encoder_loop, daemon=True).start()
 
    def on_output(fd):
        # decode/consume the 2 s rollout (e.g. keypress AE -> key states, mouse denorm)
        print("generated", OnlineSampler._time_length(fd), "dataframes")
 
    sampler.run(on_output)