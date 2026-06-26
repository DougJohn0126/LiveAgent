"""
online_sampler.py — streaming wrapper around the PlaiV1 sampling primitives.
 
It runs two cooperating pieces:
  1. INGEST: push_dataframe() is called ~every 200 ms with ONE encoded dataframe
     (video latents / key_press latents / normalized mouse — the same encoded
     format your .hdf5 + .db hold). These land in a ring buffer.
  2. GENERATE: a loop wakes every CADENCE_SECONDS, takes the recent context out
     of the ring buffer, and rolls the model forward GEN_SECONDS into the future,
     reusing the exact MemoryContext + generate_chunk used by offline sampling.
 
You supply: a live encoder that turns raw gameplay into encoded dataframes and
calls push_dataframe(). That encoder is NOT in this file (see notes in chat).
 
Run from the repo root so `import`s resolve, with the same env vars + venv you
use for eval.
"""
import os, sys, time, threading, collections
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
 
# Make sure THIS repo's `src/` is importable, then import data_classes the SAME
# way the model + live_agent do — NON-prefixed (`data.*`, not `src.data.*`).
# The src/ layout means `src.data.data_classes` and `data.data_classes` load as
# two different modules with two different FullData classes; mixing them breaks
# cat_time's isinstance check. Everyone must use the same one.
_REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
 

from plaicraft_model.src.data.data_classes  import FullData
from plaicraft_model.src.data.datamodule import DataModule
from plaicraft_model.src.inference.memory import MemoryContext
from plaicraft_model.src.inference.denoising import generate_chunk

 
# 1 dataframe = 200 ms  ->  5 dataframes / second
DATAFRAME_MS     = 200     # 1 unit = 200 ms = 2 video latents (10 fps)
GEN_SECONDS      = float(os.environ.get("PLAICRAFT_GEN_SECONDS", "2.0"))      # predict per tick
CADENCE_SECONDS  = float(os.environ.get("PLAICRAFT_CADENCE_SECONDS", "0.4"))  # how often to generate
CONTEXT_SECONDS  = 5.0     # how much recent history to condition on each tick
 
# Optional autocast to use the 3060's tensor cores (fp32 -> bf16/fp16). bf16 is
# safest. Set PLAICRAFT_AUTOCAST=bf16 to enable. ~2x faster; small risk of
# degraded/NaN output with an fp32-trained model, so it's opt-in.
_AC = os.environ.get("PLAICRAFT_DTYPE", "bf16").lower()
_AUTOCAST_DTYPE = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                   "fp16": torch.float16, "float16": torch.float16}.get(_AC)
 
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
 
 
_NOISE_PATCHED = False
 
 
def _patch_noise_dtype(dtype):
    """Make generate_chunk's noise target come out in `dtype` (e.g. bf16) without
    editing the model repo. generate_chunk calls the module-global init_noise_target,
    so wrapping that attribute on its own module is enough; the wrapper casts the
    float modalities (leaving dataframe_indices long)."""
    global _NOISE_PATCHED
    if _NOISE_PATCHED:
        return
    import sys
    dn = sys.modules[generate_chunk.__module__]      # whichever module it came from
    orig = dn.init_noise_target
 
    def _init_noise_target(*args, **kwargs):
        return _cast_fd_float(orig(*args, **kwargs), dtype)
 
    dn.init_noise_target = _init_noise_target
    _NOISE_PATCHED = True
 
 
class OnlineSampler:
    def __init__(self, model, inference_cfg, target_modalities, device="cuda"):
        self.model  = model
        self.cfg    = OmegaConf.create({"inference": inference_cfg})
        self.device = device
        self.target_modalities = list(target_modalities)
 
        # Model compute dtype (bf16 if live_agent cast it). The noise target and the
        # context we feed must match this or the first Linear hits a dtype mismatch.
        self.model_dtype = next(self.model.parameters()).dtype
        print (f"[sampler] {self.model_dtype}")
        if self.model_dtype != torch.float32:
            print ("[sampler] patching model dtype to fp32")
            _patch_noise_dtype(self.model_dtype)
 
        self.k        = int(self.model.context_embedder.ltm_downsample_chunk_length)
        self.stm_len  = int(self.model.context_embedder.stm_context_length)
        self.warmup_ltm_chunk_length = int(
            OmegaConf.select(self.cfg, "inference.warmup_ltm_chunk_length") or self.k * 8
        )
        self.chunk_len = int(OmegaConf.select(self.cfg, "inference.chunk_length") or 2)
 
        # ONE persistent memory, warmed up once then streamed incrementally
        # (NOT reset per tick). This is what keeps each step ~0.5 s instead of
        # reprocessing the whole context every time.
        self.memory = MemoryContext(self.model, self.stm_len, self.k,
                                    self.warmup_ltm_chunk_length)
        self._mem_ready = False
 
        # The real observed stream on an absolute timeline (so memory's frame
        # indexing stays valid). Bounded: when it grows past _max_frames we keep
        # the last _keep_frames and re-warm once (rare).
        self.real = None
        self._pending = []          # new frames since last generation (cheap append)
        self._max_frames  = CONTEXT_FRAMES * 20
        self._keep_frames = CONTEXT_FRAMES * 10
        self._lock  = threading.Lock() # ensure only one thread can access self._pending at a time
        self._stop  = threading.Event()
        # Raised while a rollout is generating, so the ingest thread can skip
        # GPU encodes during that window instead of timeslicing the denoiser.
        self.busy   = threading.Event()
 
    # ---------- INGEST (call from your encoder thread, ~every 200 ms) ----------
    def push_dataframe(self, frame: FullData):
        """`frame`: FullData with time-length 1, encoded modalities, on any device.
        O(1) append only — the growing sequence is extended later, in _generate_once,
        so ingestion never competes with generation by rebuilding a big GPU tensor."""
        with self._lock:
            self._pending.append(frame)
 
    # snapshot/restore the streaming memory so a rollout's fed-back predictions
    # don't pollute the real-timeline LTM cache.
    def _snapshot(self):
        ltm = self.memory.cached_ltm
        return (ltm.clone() if ltm is not None else None,
                self.memory.processed_ltm_frames)
 
    def _restore(self, snap):
        self.memory.cached_ltm, self.memory.processed_ltm_frames = snap
 
    @staticmethod
    def _time_length(fd):
        return int(fd.dataframe_indices.shape[1])
 
    # ---------- GENERATE one rollout (incremental streaming memory) ----------
    @torch.no_grad()
    def _generate_once(self):
        
        # Drain newly-ingested frames and extend the real sequence by ONLY those
        # (O(new), not O(total)). Normalizing to the local class happens here, once.
        with self._lock:
            pending, self._pending = self._pending, []
        if pending:
            add = FullData.cat_time([p for p in pending])
            self.real = add if self.real is None else FullData.cat_time([self.real, add])
            n = self._time_length(self.real)
            if n > self._max_frames:                       # bound RAM; re-warm once
                self.real = FullData.slice_time(self.real, n - self._keep_frames, n)
                self._mem_ready = False
 
        real = self.real
        if real is None:
            return None
        # First generation only: wait for a full warmup window, then prime the
        # streaming memory once. After warmup `real` is a bounded recent window
        # (smaller than CONTEXT_FRAMES), so we must NOT re-gate on its length.
        if not self._mem_ready:
            if self._time_length(real) < CONTEXT_FRAMES:
                return None
            else: 
                self.memory.reset()
                self._mem_ready = True
        
        self.busy.set()
        try:
            # Feed the model in its own dtype (bf16 if enabled); indices stay long.
            real = _cast_fd_float(real, self.model_dtype).to(self.device)
            n = self._time_length(real)
 
            # Fold any new real frames into the persistent LTM, then snapshot so the
            # rollout's fed-back predictions can be discarded afterwards.
            real_idx = DataModule.assign_indices(real, -n)
            self.memory.update_and_get_memory(real_idx)
            snap = self._snapshot()
 
            # Optionally wrap generation in autocast. This is effectively a no-op passthrough since
            # we switched to using native bf16 weights, not autocast — the autocast path forced slow attention earlier), 
            import contextlib
            ac = (torch.autocast(device_type="cuda", dtype=_AUTOCAST_DTYPE)
                  if (_AUTOCAST_DTYPE is not None and str(self.device).startswith("cuda"))
                  else contextlib.nullcontext())
 
            produced, cur = [], real                 # `cur` stays this-module's class
            #with ac:
            for _ in range(0, GEN_FRAMES, self.chunk_len):
                    clen = self._time_length(cur)
                    # assign_indices builds the model's own FullData class; use it only
                    # to feed the model -- do NOT store it back into `cur`.
                    cur_idx = DataModule.assign_indices(cur, -clen)
                    if _PROFILE: _sync(); _t0 = time.time()
                    mem = self.memory.update_and_get_memory(cur_idx)   # only new frames
                    if _PROFILE:
                        _sync(); _t_mem = time.time() - _t0
                        _stop = threading.Event(); _samples = []
                        _smp = threading.Thread(target=_gpu_util_sampler,
                                                args=(_stop, _samples), daemon=True)
                        _smp.start(); _t0 = time.time()
 
                    # One-shot torch profiler: name the kernels eating the time.
                    _do_trace = False
                    if _TORCH_PROFILE and not getattr(self, "_torch_prof_done", False):
                        self._torch_prof_n = getattr(self, "_torch_prof_n", 0) + 1
                        _do_trace = self._torch_prof_n >= 3      # skip warmup gens
                    if _do_trace:
                        from torch.profiler import profile, ProfilerActivity
                        _sync()
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as _prof:
                            pred = generate_chunk(
                                model=self.model, config=self.cfg.inference, memory=mem,
                                batch_size=FullData.infer_batch_size(cur_idx), # this is just 1 since we are only generating for one player 
                                target_pred_len=self.chunk_len,
                                target_modalities=self.target_modalities,
                                metadata=cur_idx.metadata,
                            )
                        _sync()
                        self._torch_prof_done = True
                        print("\n========== TORCH PROFILER (one generate_chunk) ==========", flush=True)
                        print(_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20), flush=True)
                        print("========================================================\n", flush=True)
                    else:
                        pred = generate_chunk(
                            model=self.model,
                            config=self.cfg.inference,
                            memory=mem,
                            batch_size=FullData.infer_batch_size(cur_idx), # this is just 1 since we are only generating for one player 
                            target_pred_len=self.chunk_len,
                            target_modalities=self.target_modalities,
                            metadata=cur_idx.metadata,
                        )
                    if _PROFILE:
                        _sync(); _t_gen = time.time() - _t0
                        _stop.set(); _smp.join(timeout=0.3)
                        if _samples:
                            _util = f"gpu_util peak={max(_samples)}% mean={sum(_samples)//len(_samples)}% (n={len(_samples)})"
                        else:
                            _util = "gpu_util n/a (pip install pynvml)"
                        print(f"[profile] mem(context-embedder)={_t_mem*1000:.0f}ms  "
                              f"gen(decoder)={_t_gen*1000:.0f}ms  {_util}", flush=True)
                    # Rebuild the predicted chunk as a LOCAL-class FullData (mirrors the
                    # offline sampler's _merge_generated): a slice of the local context
                    # is the template; modalities overwritten with the prediction.
                    template = FullData.slice_time(cur, clen - self.chunk_len, clen)
                    merged = template.to_dict()
                    for m in self.target_modalities:
                        merged[m] = pred.get_modality(m)
                    pred_fd = FullData(batch=merged)
 
                    produced.append(pred_fd)
                    cur = FullData.cat_time([cur, pred_fd])
            # Discard the rollout's prediction-folding; keep only the real-timeline LTM.
            self._restore(snap)
 
            # Drop everything older than the STM window: those frames are already folded
            # into the LTM cache, so keeping them only makes the next cat_time/assign_indices
            # O(history). Trimming to stm_len keeps every future generation constant-time.
            keep = max(self.stm_len, self.chunk_len)
            n_real = self._time_length(self.real)
            if n_real > keep:
                with self._lock:
                    self.real = FullData.slice_time(self.real, n_real - keep, n_real)
                self.memory.processed_ltm_frames = 0   # kept frames are STM, none folded
            return FullData.cat_time(produced)
        finally:
            self.busy.clear()
 
    # ---------- LOOP: GEN_SECONDS every CADENCE_SECONDS ----------
    def run(self, on_output):
        while not self._stop.is_set():
            t0 = time.time()
            print("generating")
            try:
                #out = None
                out = self._generate_once()
                if out is not None:
                    on_output(out)        # consumer: decode key/mouse, log, act, ...
            except Exception:
                # One bad tick shouldn't kill the sampler thread; log and retry.
                import traceback
                traceback.print_exc()
                out = None
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
            "model.context_modalities=[video,key_press,mouse_movement]",
            "model.target_modalities=[video,key_press,mouse_movement]",
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
    # speed knobs for the 5 s budget:
    inf_cfg = OmegaConf.merge(inf_cfg, {"num_denoising_steps": 1, "chunk_length": 1})
 
    sampler = OnlineSampler(model, inf_cfg,
                            target_modalities=["video", "key_press", "mouse_movement"])
 
    # ---- YOUR ENCODER THREAD pushes encoded dataframes here ----
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