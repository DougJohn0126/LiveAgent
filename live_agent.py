"""
 
live_agent.py  —  bridge between AgentDeployer (capture + encode + inject) and
the PLAICraft online sampler (online_sampler.OnlineSampler).
 
It turns AgentDeployer's "record 5 s -> replay your own recording" loop into:
 
      capture window ──▶ ENCODE to dataframes ──▶ push to sampler ring buffer
                                                          │
                                       (sampler generates 2 s every 5 s)
                                                          ▼
            game  ◀── INJECT predicted 2 s ◀── decode key/mouse ◀── on_output(fd)
 
Two halves, both reusing code that already exists in this repo:
 
  ENCODE half  (ingest_window):
    * keys + mouse: reuse data_preprocessor.preprocess_data + encode_key_press
      (they already emit the model's exact (16,5) keypress latents into a DB and
      bin mouse the way the model's dataset expects)
    * video: a PRELOADED SDXL VAE + the repo's own encode_latents()
    * assemble each 200 ms unit into a FullData dataframe with the EXACT shapes
      the model's collate produces, then push to sampler.push_dataframe()
 
  INJECT half  (_on_sampler_output):
    * reuse decode.decode_keypress_latents (AE decoder -> 79x10 -> threshold ->
      key/mouse-button events) and decode_mouse_movement (dx/dy)
    * drive the game via inputInjector.make_backend() + replay_keys / replay_mouse
 
------------------------------------------------------------------------------
RUNNING (single process, in the model's venv, e.g. ~/plaicraft-env):
 
  export PLAICRAFT_MODEL_REPO=/mnt/c/Users/DougJohn/Documents/GitHub/plaicraft-model-pi0
  export PLAICRAFT_CKPT="$PLAICRAFT_MODEL_REPO/last_fp32/pytorch_model.bin"
  # online_sampler.py must live in $PLAICRAFT_MODEL_REPO (repo root)
  cd /path/to/AgentDeployer
  python minecraft_input_recorder.py     # now drives live_agent (see the patch)
 
Verify the encoder matches the offline path BEFORE going live:
  python -c "import live_agent; live_agent.self_check()"
------------------------------------------------------------------------------
"""
from __future__ import annotations
 
import os
import sys
import json
import time
import pickle
import threading
import traceback
import os
import torch
from pathlib import Path
from dotenv import load_dotenv
import scripts_replay.inputInjector
 
import numpy as np

load_dotenv()
 
# --------------------------------------------------------------------------- #
# Path wiring: this repo + the model repo (src-layout) + online_sampler.py
# --------------------------------------------------------------------------- #
AGENT_DIR = Path(__file__).resolve().parent
MODEL_REPO = Path(os.environ.get("PLAICRAFT_MODEL_REPO")).expanduser()
CKPT = os.environ.get("PLAICRAFT_CKPT")
 
for p in (str(AGENT_DIR), str(MODEL_REPO), str(MODEL_REPO / "src")):
    if p and p not in sys.path:
        sys.path.insert(0, p)
 
# Encoding cadence / model layout constants (verified against the model repo).
DATAFRAME_MS = 200          # 1 unit = 200 ms = 2 video latents (10 fps)
LATENT_FPS = 10             # video latents per second
VIDEO_FRAMES_PER_UNIT = 2
MOUSE_TOKENS_PER_UNIT = 20  # 20 bins of 10 ms per 200 ms
KEYBOARD_TOKENS_PER_UNIT = 10
KEY_LATENT_DIM = 16
BINS_PER_FRAME = 10         # mouse 10 ms bins per 100 ms video frame
CLIP_MOUSE_DX = (-150.0, 150.0)
CLIP_MOUSE_DY = (-100.0, 100.0)
VIDEO_LATENT_SHAPE = (4, 96, 160)   # SDXL VAE of a 1280x768 (padded) frame
ENCODE_FRAME_WH = (1280, 720)       # what every captured frame is resized to
PAD_FRAME_WH = (1280, 768)          # then letterbox-padded to (matches training)
 
OUTPUT_DIR = "processed_data"
 
# --- audio (context modality): Encodec 24kHz continuous embeddings ---------- #
AUDIO_SR               = 24000
AUDIO_TOKENS_PER_UNIT  = 15        # 75 Hz * 0.2 s
AUDIO_FEATURE_DIM      = 128
AUDIO_SAMPLES_PER_UNIT = AUDIO_SR // 5   # 4800 samples per 200 ms unit
 
# Filled in by init().
_S: dict = {}
 
 
# --------------------------------------------------------------------------- #
# INIT — load everything once, start the sampler thread
# --------------------------------------------------------------------------- #
def init(device: str = "cuda", num_denoising_steps: int = 50, make_backend: bool = True):
    """Preload model + VAE + keypress AE; start the sampler loop.
 
    on_output(fd): called with each 2 s rollout. Defaults to local injection.
                   The IPC server passes a callback that sends predictions to the
                   Windows client instead.
    make_backend:  build a local input backend (pydirectinput/uinput). Set False
                   on a headless server that only decodes + forwards.
    """
    import torch
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL
 
    if not MODEL_REPO or not (MODEL_REPO / "configs").is_dir():
        raise RuntimeError(
            "Set PLAICRAFT_MODEL_REPO to the plaicraft-model-pi0 repo root "
            "(the folder containing configs/ and online_sampler.py)."
        )
    if not CKPT or not Path(CKPT).is_file():
        raise RuntimeError("Set PLAICRAFT_CKPT to your pytorch_model.bin checkpoint.")
 
    from online_sampler import OnlineSampler  # lives in the model repo root
 
    dev = device if torch.cuda.is_available() else "cpu"
    print(f"[live_agent] device = {dev}")
    
    # 4 performance levers: TF32 (faster fp32 without the bf16-autocast fallback), 
    # cuDNN benchmark (autotuned convs for the fixed-shape VAE), the precision hint (same TF32 intent via the modern API), 
    # and the dynamo cache bump (stop flex_attention from recompiling every tick). 
    if str(dev).startswith("cuda"):
        # TF32: use the Ampere tensor cores for fp32 matmuls without changing dtypes.
        # Unlike bf16 autocast, this keeps attention on its fast SDPA kernel (bf16
        # forced the slow math fallback here), so it speeds up fp32 with no fallback.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # The VAE encode always sees the same input shape, so let cuDNN autotune
        # the fastest convolution algorithm for it (one-time warmup, then faster
        # every encode).
        torch.backends.cudnn.benchmark = True
        # This is the newer, higher-level knob that overlaps with the TF32 switches — "
        # high" tells torch it may use TF32-style reduced precision for fp32 matmuls. 
        # Setting both this and allow_tf32 is belt-and-suspenders; they're expressing the same intent through old and new APIs. 
        # It's wrapped in try/except because it doesn't exist on older torch versions, and a failure here shouldn't stop startup.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # flex_attention is torch.compile'd; the default recompile cache holds only
        # 8 shapes and then evicts+recompiles. Our varied sequence lengths blow past
        # that, so every tick recompiles (2 s/step instead of ~0.5 s). Raise the
        # limits so all the shapes stay cached and steps run at the GPU's real speed after a one-time warmup.
        try:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 256
            torch._dynamo.config.accumulated_cache_size_limit = 512
        except Exception:
            pass
        print("[live_agent] TF32 enabled; dynamo cache limit raised")
 
    # --- the world model + streaming sampler -------------------------------- #
    model, inf_cfg = _load_model(CKPT, dev)
    steps = int(os.environ.get("PLAICRAFT_DENOISE_STEPS", num_denoising_steps))
    chunk_len = int(os.environ.get("PLAICRAFT_CHUNK_LEN", ))
    OmegaConf.set_struct(inf_cfg, False)        # allow adding/overriding keys
    inf_cfg.num_denoising_steps = steps         # generate_chunk reads this exact key
    inf_cfg.chunk_length = chunk_len
    print(f"[live_agent] num_denoising_steps = {steps}")
    print(f"[live_agent] num_chunk_len = {inf_cfg.chunk_length}")
 
    # Optional native bf16 inference (the model was TRAINED in bf16 under DeepSpeed;
    # min_gru/LayerNorm/MPConv all upcast their fragile math to fp32 internally, so
    # bf16 weights are the native regime, ~2x the fp32 decoder). Opt-in via env.
    _dtype = os.environ.get("PLAICRAFT_DTYPE").lower()
    if _dtype in ("bf16", "bfloat16") and str(dev).startswith("cuda"):
        model = model.to(torch.bfloat16)
        print("[live_agent] world model cast to bf16")
 
    # The output path (_decode_prediction) only consumes key_press + mouse_movement
    # and never decodes predicted video, so by default we DON'T denoise video --
    # that's ~94% of the target tokens generated and thrown away. STM still carries
    # the full real video history, so the model is still conditioned on what it sees;
    # it just predicts actions instead of pixels. Set PLAICRAFT_TARGET_MODALITIES=
    # "video,key_press,mouse_movement" to restore joint video generation.
    _tm_env = os.environ.get("PLAICRAFT_TARGET_MODALITIES")
    target_modalities = [m.strip() for m in _tm_env.split(",") if m.strip()]
    print(f"[live_agent] target_modalities = {target_modalities}")
    sampler = OnlineSampler(
        model, inf_cfg,
        target_modalities=target_modalities,
        device=dev,
    )

    # --- audio context encoder (Encodec 24kHz). Matches the corpus builder
    #     main_continuous_hdf5.py exactly (no set_target_bandwidth); the
    #     encode -> quantizer.decode round-trip lives in _encode_audio_inmemory.
    from encodec import EncodecModel as _EncodecPkg
    _S["encodec"] = _EncodecPkg.encodec_model_24khz().to(dev).eval()
    print("[live_agent] Encodec audio context encoder loaded")
 
    # --- SDXL VAE for live video encode (loaded ONCE; the repo's encoder
    #     reloads it every call, which is far too slow for a live loop) ------- #
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    ).to(dev).eval()
    try:
        vae = vae.to(memory_format=torch.channels_last)   # faster conv kernels
    except Exception:
        print ("[live_agent] faster conversion kernels failed")
        pass
    
    # --- optional: torch.compile the SDXL encoder (~1.3-1.8x). Compile the
    #     ENCODER submodule, not the AutoencoderKL: vae.forward is the
    #     encode->decode roundtrip we never call, and vae.encode() returns a
    #     DiagonalGaussianDistribution that graph-breaks. The conv stack is ~all
    #     the compute. Warm it up here so the first live frame doesn't stall. ---
    if str(dev).startswith("cuda") and os.environ.get("PLAICRAFT_VAE_COMPILE", "0").lower() in ("1", "true", "yes"):
        try:
            vae.encoder = torch.compile(
                vae.encoder,
                mode="default",             # best kernels; cudagraphs are
                                                    # fragile with fresh-allocated
                                                    # inputs each call
                fullgraph=True,                     # drop this if it raises on a
                                                    # graph break
            )
            # Warm up EVERY batch size the chunk loop can hand it, so none
            # recompiles mid-stream. Your live path encodes 2 frames/unit; the
            # clip path chunks by 8 with a ragged tail. Warm 8, 2, 1 (cheap vs.
            # one mid-loop recompile, and the dynamo cache holds 256).
            with torch.inference_mode():
                dummy = torch.zeros(
                    8, 3, PAD_FRAME_WH[1], PAD_FRAME_WH[0],
                    device=dev, dtype=torch.float16,
                    ).to(memory_format=torch.channels_last)
                vae.encode(2 * dummy - 1).latent_dist.sample()
            torch.cuda.synchronize()
            print("[live_agent] SDXL VAE encoder compiled + warmed up")
        except Exception as e:
            print(f"[live_agent] VAE compile failed ({e}); falling back to eager")
    
    # --- optional FAST VAE (TAESDXL): a tiny distilled autoencoder that produces
    #     SDXL-compatible latents ~100x faster than the full SDXL VAE. The full VAE
    #     encode (~490 ms/frame on a 3060) is the live loop's true bottleneck; this
    #     replaces it. We keep the SDXL VAE around for a one-time per-channel
    #     calibration (see _encode_frames) so the tiny VAE's latents are mapped onto
    #     the exact distribution the world model trained on, then free it. ------- #
    vae_mode = os.environ.get("PLAICRAFT_VAE", "sdxl").lower()
    vae_fast = None
    if vae_mode in ("fast", "taesd", "taesdxl", "tiny"):
        from diffusers import AutoencoderTiny
        vae_fast = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=torch.float16
        ).to(dev).eval()
        try:
            vae_fast = vae_fast.to(memory_format=torch.channels_last)
        except Exception:
            pass
        print("[live_agent] fast VAE (TAESDXL) loaded; will calibrate on first frame")
 
    # --- keypress AE for BOTH directions (encode + decode); loaded once ----- #
    from plaicraft_model.decode import decode_keypress_latents as dkl
    ae = dkl.build_autoencoder(dkl.CHECKPOINT_DIR, device="cpu")
    index_to_name, mouse_indices = dkl.load_name_maps()
    # forward map (key_id/button -> channel index) for in-memory encoding
    from plaicraft_model.encode_key_press.scripts.constants import id_to_index, id_to_name
 
    # --- input backend (only when injecting locally; skip on a server) ------- #
    backend = None
    inputInjector = None
    if make_backend:
        import scripts_replay.inputInjector as inputInjector
        backend = inputInjector.make_backend()
 
    _S.update(
        torch=torch, sampler=sampler, vae=vae, ae=ae, dkl=dkl,
        vae_fast=vae_fast, vae_mode=("fast" if vae_fast is not None else "sdxl"),
        vae_calib=None,   # (scale[4,1,1], bias[4,1,1]) fitted on first encode
        index_to_name=index_to_name, mouse_indices=mouse_indices,
        id_to_index=id_to_index, id_to_name=id_to_name,
        held_keys={}, held_clicks={},   # carried across windows
        backend=backend, inputInjector=inputInjector, device=dev,
    )
 
    # Sampler runs in its own thread; every tick it emits a 2 s rollout, handed to
    # on_output (local injection by default; the IPC server forwards it instead).
    t = threading.Thread(target=sampler.run, args=(_on_sampler_output,), daemon=True)
    t.start()
    _S["sampler_thread"] = t
    print("[live_agent] ready — sampler loop started.")
    return _S
 
 
def _encode_audio_inmemory(pcm_hear, pcm_speak):
    """Two mono 24kHz float32 clips (~4800 samples) -> two (15,128) arrays,
    matching main_continuous_hdf5.py: encode -> quantizer.decode -> (T,128)."""
    import torch
    enc = _S["encodec"]
    def _emb(pcm):
        x = torch.as_tensor(pcm, dtype=torch.float32, device=_S["device"]).flatten()
        n = AUDIO_SAMPLES_PER_UNIT
        x = torch.nn.functional.pad(x, (0, n - x.numel())) if x.numel() < n else x[:n]
        frames = enc.encode(x.view(1, 1, n))            # [(codes, scale)], codes:(1,K,T)
        zs = []
        for code, _scale in frames:
            z = enc.quantizer.decode(code.transpose(0, 1))   # (1,128,T)  <-- the key step
            zs.append(z)
        z = torch.cat(zs, dim=-1).squeeze(0).transpose(0, 1).contiguous()  # (T,128)
        if z.shape[0] < AUDIO_TOKENS_PER_UNIT:
            z = torch.nn.functional.pad(z, (0, 0, 0, AUDIO_TOKENS_PER_UNIT - z.shape[0]))
        return z[:AUDIO_TOKENS_PER_UNIT].float().cpu().numpy()   # (15,128)
    return _emb(pcm_hear), _emb(pcm_speak)
 
 
def _torch_load_low_mem(ckpt_path):
    """Load a big checkpoint without copying the whole thing into RAM.
    mmap=True keeps weights file-backed (low CPU peak); fall back gracefully."""
    import torch
    attempts = (
        dict(map_location="cpu", mmap=True, weights_only=True),
        dict(map_location="cpu", mmap=True, weights_only=False),
        dict(map_location="cpu", weights_only=True),
        dict(map_location="cpu", weights_only=False),
    )
    last = None
    for kw in attempts:
        try:
            return torch.load(ckpt_path, **kw)
        except (TypeError, RuntimeError, OSError, ValueError) as e:
            last = e
    raise last
 
 
def _load_model(ckpt_path, device):
    """Mirror online_sampler.load_model but with an explicit config dir, and a
    memory-frugal load (the fp32 1.7B checkpoint is ~7 GB; we avoid holding two
    copies in RAM at once)."""
    import gc
    import torch, hydra
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(MODEL_REPO / "configs"), version_base=None):
        cfg = compose(config_name="eval", overrides=[
            "model=plai_v1_trained",
            "model.context_modalities=[video,audio_hear,audio_speak,key_press,mouse_movement]",
            "model.target_modalities=[video,audio_hear,audio_speak,key_press,mouse_movement]",
        ])
    # Instantiate, then move to the GPU FIRST so the CPU copy is freed before we
    # read the checkpoint — otherwise model (CPU) + checkpoint (CPU) = 2x ~7 GB.
    model = hydra.utils.instantiate(cfg.model).to(device).eval()
    gc.collect()
    sd = _torch_load_low_mem(ckpt_path)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    clean = {}
    for k, v in sd.items():
        for pre in ("module.model.", "model.", "module."):
            if k.startswith(pre):
                k = k[len(pre):]; break
        clean[k] = v
    miss, unexp = model.load_state_dict(clean, strict=False)   # mmap source -> GPU
    del sd, clean
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"[live_agent] model loaded ({len(miss)} missing / {len(unexp)} unexpected keys)")
    return model, cfg.inference
 
# --------------------------------------------------------------------------- #
# ENCODE HALF — capture window -> dataframes -> ring buffer
# --------------------------------------------------------------------------- #
def ingest_window(mouse_events, click_events, key_events, clip_path):
    """Push one captured window to the model's ring buffer.
 
    mouse_events / click_events / key_events are the RAW lists of dicts the
    recorder already holds (JSON strings are still
    accepted for backward compatibility.)
    """
    def _as_list(x):
        return json.loads(x) if isinstance(x, str) else (x or [])
    try:
        _ingest_window(_as_list(mouse_events), _as_list(click_events),
                       _as_list(key_events), Path(clip_path))
    except Exception:
        print("[live_agent] ingest_window failed:")
        traceback.print_exc()
 
 
def _ingest_window(mouse_events, click_events, key_events, clip_path: Path):
    # 1) Video first (its frame count defines the window's coverage). Preloaded VAE.
    video = _encode_video(clip_path)                 # (F,4,96,160) np.float32
    n_video = video.shape[0]
    if n_video < VIDEO_FRAMES_PER_UNIT:
        print("[live_agent] window too short to form a 200ms unit; skipped.")
        return
 
    # 2) Shared t0 = earliest mouse timestamp (matches the offline trim_start).
    if mouse_events:
        t0 = min(int(e["timestamp"]) for e in mouse_events)
    elif key_events:
        t0 = min(int(e["timestamp"]) for e in key_events)
    else:
        print("[live_agent] no input events in window; skipped.")
        return
 
    n_frames = n_video                               # one 100 ms frame per latent
    # 3) keys+clicks -> (16,5) per frame, straight from raw events (no JSON/DB).
    key_by_frame = _encode_keys_inmemory(key_events, click_events, t0, n_frames)
    # 4) mouse -> (2,10) per frame, binned + clipped exactly like the dataset.
    mouse_by_frame = _mouse_bins_inmemory(mouse_events, t0, n_frames)
 
    n_units = n_frames // VIDEO_FRAMES_PER_UNIT
    pushed = 0
    for u in range(n_units):
        fd = _assemble_dataframe(u, video, key_by_frame, mouse_by_frame)
        _S["sampler"].push_dataframe(fd)
        pushed += 1
    print(f"[live_agent] pushed {pushed} dataframes "
          f"({pushed*DATAFRAME_MS/1000:.1f}s) to the ring buffer.")
 
 
def _encode_frames(frames_bgr) -> np.ndarray:
    """Encode a list of BGR frames (any size) with the preloaded VAE.
    Resizes to the training resolution, letterbox-pads 720->768, returns
    (N,4,96,160) float32. Shared by the clip path and the live stream path."""
    import cv2
    import torch
    from torchvision.transforms import functional as TF
    from PIL import Image
 
    if not frames_bgr:
        return np.zeros((0, *VIDEO_LATENT_SHAPE), dtype=np.float32)
 
    pad_top = (PAD_FRAME_WH[1] - ENCODE_FRAME_WH[1]) // 2
    batch = []
    for f in frames_bgr:
        if (f.shape[1], f.shape[0]) != ENCODE_FRAME_WH:
            f = cv2.resize(f, ENCODE_FRAME_WH)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t = TF.to_tensor(TF.pad(Image.fromarray(rgb), (0, pad_top), fill=0))
        batch.append(t.half())
    imgs = torch.stack(batch).to(_S["device"]).to(memory_format=torch.channels_last)
 
    lat = []
    import time as _time
    _vae_prof = os.environ.get("PLAICRAFT_PROFILE", "0").lower() in ("1", "true", "yes")
    if _vae_prof and torch.cuda.is_available():
        torch.cuda.synchronize(); _vt0 = _time.time()
    if _S.get("vae_mode") == "fast" and _S.get("vae_fast") is not None:
        out = _encode_frames_fast(imgs)
    else:
        bs = 8
        with torch.inference_mode():
            for i in range(0, imgs.shape[0], bs):
                chunk = imgs[i:i + bs]
                real = chunk.shape[0]
                if real < bs:
                    # Pad the ragged tail up to the fixed batch so the compiled
                    # encoder sees ONE shape (no recompiles). Pad rows are dropped
                    # below. .contiguous(channels_last) keeps the memory format
                    # identical to what compile traced during warmup.
                    pad = imgs.new_zeros((bs - real, *chunk.shape[1:]))
                    chunk = torch.cat([chunk, pad], 0) \
                                 .contiguous(memory_format=torch.channels_last)
                z = _S["vae"].encode(2 * chunk - 1).latent_dist.sample() * 0.13025
                lat.append(z[:real].float().cpu())   # slice off the pad rows
        out = torch.cat(lat, 0).numpy()
    if _vae_prof and torch.cuda.is_available():
        torch.cuda.synchronize()
        _dt = (_time.time() - _vt0) * 1000
        print(f"[profile] VAE encode {imgs.shape[0]} frame(s) = {_dt:.0f}ms "
              f"({_dt / max(imgs.shape[0], 1):.0f}ms/frame)", flush=True)
    return out
 
 
def _encode_frames_fast(imgs) -> np.ndarray:
    """Encode with TAESDXL (tiny VAE). On the first call, fit a per-channel
    scale+bias mapping TAESDXL latents -> the SDXL-VAE latent distribution the
    world model trained on (z = vae.encode(2x-1).sample()*0.13025), using the SDXL
    VAE that's still loaded; then free the SDXL VAE. After that it's pure TAESDXL +
    an elementwise affine -- single-digit ms/frame."""
    import torch
    taesd = _S["vae_fast"]
 
    def _tae_latents(x01):
        zs = []
        with torch.inference_mode():
            for i in range(0, x01.shape[0], 8):
                z = taesd.encode(2 * x01[i:i + 8] - 1).latents   # diffusers remaps internally
                zs.append(z)
        return torch.cat(zs, 0)
 
    if _S.get("vae_calib") is None and _S.get("vae") is not None:
        with torch.inference_mode():
            z_sdxl = []
            for i in range(0, imgs.shape[0], 8):
                z = _S["vae"].encode(2 * imgs[i:i + 8] - 1).latent_dist.sample() * 0.13025
                z_sdxl.append(z.float())
            z_sdxl = torch.cat(z_sdxl, 0)                          # (N,4,96,160)
        z_tae = _tae_latents(imgs).float()
        dims = (0, 2, 3)                                           # per-channel moment match
        mu_s = z_sdxl.mean(dims, keepdim=True); sd_s = z_sdxl.std(dims, keepdim=True)
        mu_t = z_tae.mean(dims, keepdim=True);  sd_t = z_tae.std(dims, keepdim=True)
        scale = sd_s / (sd_t + 1e-6)
        bias = mu_s - scale * mu_t
        _S["vae_calib"] = (scale, bias)
        print(f"[live_agent] TAESDXL calibrated to SDXL latents "
              f"(per-channel scale {[round(v,3) for v in scale.flatten().tolist()]})", flush=True)
        try:
            _S["vae"] = None
            torch.cuda.empty_cache()
        except Exception:
            pass
 
    scale, bias = _S["vae_calib"]
    z = _tae_latents(imgs).float()
    z = z * scale.to(z.device) + bias.to(z.device)
    return z.cpu().numpy()
 
 
def _encode_video(clip_path: Path) -> np.ndarray:
    """Read a window clip, take every Nth frame for 10 fps, encode (batch path)."""
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    src_fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) and (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round((src_fps or 30.0) / LATENT_FPS)))   # 30fps -> every 3rd
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return _encode_frames(frames)
 
 
def _intervals_from_events(events, held, t_end):
    """events: list of (id_str, 'PRESS'|'RELEASE', ts_ms). Pair into
    (id, start_ms, end_ms) intervals. `held` (mutated) carries ids still down
    ACROSS flushes, so a key held over a 5 s boundary isn't lost the way the
    offline PRESS/RELEASE-in-one-batch pairing loses it. Anything still held at
    t_end gets an open interval to t_end and stays held for the next window."""
    intervals = []
    for ids, act, ts in sorted(events, key=lambda x: x[2]):
        if act == "PRESS":
            held.setdefault(ids, int(ts))
        elif act == "RELEASE":
            if ids in held:
                intervals.append((ids, held.pop(ids), int(ts)))
    for ids, st in held.items():
        intervals.append((ids, st, t_end))
    return intervals
 
 
def _keys_to_multihot(key_events, click_events, t0, n_frames):
    """Raw key/click events -> (n_frames, 79, 10) float32 multi-hot.
 
    Byte-for-byte port of the offline KeyPressDataset featurization (same 100 ms
    windows, same `start<bin_end and end>bin_start` overlap test, same
    id_to_index / scroll handling). Split out from the AE encode so it can be
    unit-tested against the offline path with no torch dependency.
    """
    id_to_index = _S["id_to_index"]
    id_to_name = _S["id_to_name"]
    frame_ms = 1000.0 / LATENT_FPS                 # 100 ms
    seq = 10
    t_end = int(t0 + n_frames * frame_ms)
 
    kev = [(str(e["key"]), e["action"], e["timestamp"]) for e in key_events]
    cev = []
    for e in click_events:
        a = e["action"]
        if a.endswith("_PRESS"):
            act = "PRESS"
        elif a.endswith("_RELEASE"):
            act = "RELEASE"
        else:
            continue                                # scroll is instantaneous; skip
        cev.append((a.rsplit("_", 1)[0].lower(), act, e["timestamp"]))
 
    intervals = (_intervals_from_events(kev, _S["held_keys"], t_end)
                 + _intervals_from_events(cev, _S["held_clicks"], t_end))
 
    frames = np.zeros((n_frames, 79, seq), dtype=np.float32)
    for ids, istart, iend in intervals:
        idx = id_to_index.get(ids)
        if idx is None or iend <= istart:
            continue
        scroll = id_to_name.get(ids, "").startswith("scroll_")
        first = max(0, int((istart - t0) // frame_ms))
        last = min(n_frames - 1, int((iend - t0) // frame_ms))
        for i in range(first, last + 1):
            w0 = t0 + i * frame_ms
            rs = max(0.0, (istart - w0) / frame_ms)
            re = min(1.0, (iend - w0) / frame_ms)
            if scroll:
                re = min(re + 1.0 / seq, 1.0)
            for t in range(seq):
                if rs < (t + 1) / seq and re > t / seq:
                    frames[i, idx, t] = 1.0
    return frames
 
 
def _encode_keys_inmemory(key_events, click_events, t0, n_frames):
    """Raw key/click events -> {frame_idx: (16,5)} via the preloaded AE
    (no JSON, no SQLite, no per-call AE reload)."""
  
    frames = _keys_to_multihot(key_events, click_events, t0, n_frames)
    with torch.no_grad():
        z = _S["ae"].encoder(torch.from_numpy(frames)).cpu().numpy()   # (n_frames,16,5)
    return {i: z[i] for i in range(n_frames)}
 
 
def _mouse_bins_inmemory(mouse_events, t0, n_frames):
    """Raw mouse-movement events -> {frame_idx: (2,10)}, binned + outlier-clipped
    exactly like the model dataset's _load_mouse_movement (no DB)."""
    out = {}
    frame_len = 1000.0 / LATENT_FPS                # 100 ms
    bin_w = frame_len / BINS_PER_FRAME             # 10 ms
    horizon = n_frames * frame_len
    for e in mouse_events:
        dt = int(e["timestamp"]) - t0
        if dt < 0 or dt >= horizon:
            continue
        fi = int(dt // frame_len)
        b = min(max(int((dt - fi * frame_len) // bin_w), 0), BINS_PER_FRAME - 1)
        f = out.setdefault(fi, np.zeros((2, BINS_PER_FRAME), dtype=np.float32))
        f[0, b] += float(e.get("mouseDX", 0.0))
        f[1, b] += float(e.get("mouseDY", 0.0))
    for f in out.values():
        f[0][(f[0] < CLIP_MOUSE_DX[0]) | (f[0] > CLIP_MOUSE_DX[1])] = 0.0
        f[1][(f[1] < CLIP_MOUSE_DY[0]) | (f[1] > CLIP_MOUSE_DY[1])] = 0.0
    return out
 
 
def _assemble_dataframe(u, video, key_by_frame, mouse_by_frame,  audio_hear=None, audio_speak=None):
    """Build ONE FullData unit with the exact post-collate shapes:
         video [1,1,2,4,96,160]  mouse [1,1,20,2]  key [1,1,10,16]."""
    import torch
    from plaicraft_model.src.data.data_classes import FullData
 
    f0 = u * VIDEO_FRAMES_PER_UNIT
    # video: 2 consecutive 100ms latents
    v = torch.from_numpy(video[f0:f0 + VIDEO_FRAMES_PER_UNIT]).float()         # (2,4,96,160)
    v = v.reshape(1, 1, VIDEO_FRAMES_PER_UNIT, *VIDEO_LATENT_SHAPE)            # [1,1,2,4,96,160]
 
    # key: 2 frames x (16,5) -> concat on time -> (16,10) -> (10,16) -> [1,1,10,16]
    kf = [key_by_frame.get(f0 + j, np.zeros((KEY_LATENT_DIM, 5), np.float32))
          for j in range(VIDEO_FRAMES_PER_UNIT)]
    k = torch.from_numpy(np.concatenate(kf, axis=1)).float()                  # (16,10)
    k = k.permute(1, 0).reshape(1, 1, KEYBOARD_TOKENS_PER_UNIT, KEY_LATENT_DIM)
 
    # mouse: 2 frames x (2,10) -> concat -> (2,20) -> (20,2) -> [1,1,20,2]
    mf = [mouse_by_frame.get(f0 + j, np.zeros((2, BINS_PER_FRAME), np.float32))
          for j in range(VIDEO_FRAMES_PER_UNIT)]
    m = torch.from_numpy(np.concatenate(mf, axis=1)).float()                  # (2,20)
    m = m.permute(1, 0).reshape(1, 1, MOUSE_TOKENS_PER_UNIT, 2)
    
    def _au(a):
        if a is None:
            a = np.zeros((AUDIO_TOKENS_PER_UNIT, AUDIO_FEATURE_DIM), np.float32)
        return torch.from_numpy(a).float().reshape(1, 1, AUDIO_TOKENS_PER_UNIT, AUDIO_FEATURE_DIM)
    
    batch = {
        "video": v,
        "audio_hear":  _au(audio_hear),
        "audio_speak": _au(audio_speak),
        "key_press": k,
        "mouse_movement": m,
        "dataframe_indices": torch.zeros((1, 1), dtype=torch.long),
        "metadata": {"session_id": "live", "start_time": 0},
    }
    return FullData(batch=batch).to(_S["device"])
 
 
# --------------------------------------------------------------------------- #
# INJECT HALF — sampler output -> decode -> drive the game
# --------------------------------------------------------------------------- #
def _decode_prediction(fd):
    """fd (~2 s rollout) -> (key_events, mouse_rows). No backend needed, so this
    runs on the headless WSL2 server too. key_events: [{key_name,start_ms,end_ms}],
    mouse_rows: [(time_ms, dx, dy)]."""
    dkl = _S["dkl"]
 
    # keys: [1,U,10,16] -> per-100ms (16,5) latents -> press intervals -> events
    kp = fd.key_press
    if kp is not None:
        U = kp.shape[1]
        win = kp[0].reshape(U * VIDEO_FRAMES_PER_UNIT, 5, KEY_LATENT_DIM)
        # The model stores each window as (latent_seq_len=5, latent_dim=16).
        # decode_latents_to_activations applies its own .T to get the decoder's
        # required (16,5), so pass the native (5,16) here (do NOT pre-transpose).
        latents = win.float().contiguous().cpu().numpy()              # (Nwin,5,16)
        acts = dkl.decode_latents_to_activations(_S["ae"], latents, device="cpu")
        bins = dkl.binarize(acts, thresh=dkl.KEY_ON_THRESH)            # (Nwin,79,10)
        parsed = dkl.parse_events(bins, _S["index_to_name"], _S["mouse_indices"], 0)
        key_events = parsed["keyboard_events"] + parsed["mouse_button_events"]
    else:
        key_events = []
 
    # mouse: [1,U,20,2] -> per-100ms (2,10) -> (time_ms, dx, dy) rows
    mm = fd.mouse_movement
    mouse_rows = []
    if mm is not None:
        U = mm.shape[1]
        frames = mm[0].reshape(U * VIDEO_FRAMES_PER_UNIT, BINS_PER_FRAME, 2)
        per_window = [f.float().permute(1, 0).cpu().numpy() for f in frames]   # (2,10)
        series = dkl.decode_mouse_movement(per_window, mm_stats=None)["series"]
        mouse_rows = [(s["time_ms"], s["dx"], s["dy"]) for s in series]
 
    return key_events, mouse_rows
 
 
def _on_sampler_output(fd):
    """Default (local) callback for OnlineSampler.run(): decode + inject here."""
    try:
        _inject(fd)
    except Exception:
        print("[live_agent] inject failed:")
        traceback.print_exc()
 
 
def _inject(fd):
    key_events, mouse_rows = _decode_prediction(fd)
    drive_backend(key_events, mouse_rows)
 
 
def drive_backend(key_events, mouse_rows):
    """Inject a decoded 2 s action plan with the local backend (Windows/uinput)."""
    inj = _S["inputInjector"]
    backend = _S["backend"]
    if inj is None or backend is None:
        raise RuntimeError("No local input backend (init was called with make_backend=False).")
    inj._timer_begin()
    start_perf = time.perf_counter()
    tk = threading.Thread(target=inj.replay_keys,
                          args=(key_events, start_perf, backend), daemon=True)
    tm = threading.Thread(target=inj.replay_mouse,
                          args=(mouse_rows, 10, start_perf, backend), daemon=True)
    tk.start(); tm.start()
    #tk.join(); tm.join()
    inj._timer_end()
 
 
# --------------------------------------------------------------------------- #
# SELF CHECK — assemble one dataframe and assert the shapes match the model
# --------------------------------------------------------------------------- #
def self_check():
    """Build a dummy dataframe and assert it matches the model's expected
    FullData layout. Run this before going live."""
    import torch
    from plaicraft_model.src.data.data_classes import FullData
    F = 4
    video = np.zeros((F, *VIDEO_LATENT_SHAPE), dtype=np.float32)
    keys = {i: np.zeros((KEY_LATENT_DIM, 5), np.float32) for i in range(F)}
    mouse = {i: np.zeros((2, BINS_PER_FRAME), np.float32) for i in range(F)}
    # init() not required for the shape check; assemble directly without .to(device)
    global _S
    saved = _S.get("device")
    _S["device"] = "cpu"
    fd = _assemble_dataframe(0, video, keys, mouse)
    _S["device"] = saved
    assert tuple(fd.video.shape) == (1, 1, 2, 4, 96, 160), fd.video.shape
    assert tuple(fd.key_press.shape) == (1, 1, 10, 16), fd.key_press.shape
    assert tuple(fd.mouse_movement.shape) == (1, 1, 20, 2), fd.mouse_movement.shape
    assert FullData.infer_time_length(fd) == 1
    assert FullData.infer_batch_size(fd) == 1
    print("[live_agent] self_check OK — dataframe layout matches the model.")
 
 
if __name__ == "__main__":
    self_check()