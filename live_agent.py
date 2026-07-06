import os
import sys
import importlib
import time
from pathlib import Path
import threading
import time
import traceback
import gc
import hydra
from hydra import compose, initialize_config_dir
import torch
from diffusers import AutoencoderTiny
from encodec import EncodecModel as _EncodecPkg
import numpy as np

import cv2
from PIL import Image
from torchvision.transforms import functional as F

import scripts_replay.inputInjector as inputInjector
import decode.decode_keypress_latents as dkl


# --------------------------------------------------------------------------- #
# Path wiring: this repo + the model repo (src-layout) + online_sampler.py
# --------------------------------------------------------------------------- #
AGENT_DIR = Path(__file__).resolve().parent
MODEL_REPO = Path(os.environ.get("PLAICRAFT_MODEL_REPO")).expanduser()
sys.path.append(str(MODEL_REPO))
from src.data.data_classes import FullData

CKPT = os.environ.get("PLAICRAFT_CKPT")
STEPS = int(os.environ.get("PLAICRAFT_DENOISE_STEPS", 8))
CHUNK_LEN = int(os.environ.get("PLAICRAFT_CHUNK_LEN", 5))
DTYPE = os.environ.get("PLAICRAFT_DTYPE").lower()
TARGET_MODALITIES = os.environ.get("PLAICRAFT_TARGET_MODALITIES")
VAE_MODE = os.environ.get("PLAICRAFT_VAE", "sdxl").lower()
PROFILE_VAE = os.environ.get("PLAICRAFT_PROFILE", "0").lower() in ("1", "true", "yes")

VAE_BATCH_SIZE = os.environ.get("PLAICRAFT_VAE_BATCH_SIZE", 2)
 
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
_S: dict = dict.fromkeys(['torch', 'sampler', 'vae', 'ae', 'dkl', 'vae_fast', 'vae_mode', 'vae_calib', 'id_to_index', 
                          'id_to_name', 'held_keys', 'held_clicks', 'backend', 'inputInjector', 'device', 'sampler_thread'], 'Unknown')


# --------------------------------------------------------------------------- #
# INIT — load everything once, start the sampler thread
# --------------------------------------------------------------------------- #
def init(device: str = "cuda", make_backend: bool = True):
    """Preload model + VAE + keypress AE; start the sampler loop.
 
    on_output(fd): called with each 2 s rollout. Defaults to local injection.
                   The IPC server passes a callback that sends predictions to the
                   Windows client instead.
    make_backend:  build a local input backend (pydirectinput/uinput). Set False
                   on a headless server that only decodes + forwards.
    """
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL
 
    if not MODEL_REPO or not (MODEL_REPO / "configs").is_dir():
        raise RuntimeError(
            "Set PLAICRAFT_MODEL_REPO to the plaicraft-model-pi0 repo root "
            "(the folder containing configs/ and online_sampler.py)."
        )
    if not CKPT or not Path(CKPT).is_file():
        raise RuntimeError("Set PLAICRAFT_CKPT to your pytorch_model.bin checkpoint.")
 
 
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
            import torch._dynamo as _dynamo
            _dynamo.config.cache_size_limit = 256
            _dynamo.config.accumulated_cache_size_limit = 512
        except Exception:
            pass
        print("[live_agent] TF32 enabled; dynamo cache limit raised")
 
    # --- the world model + streaming sampler -------------------------------- #
    model, inf_cfg = _load_model(CKPT, dev)
    OmegaConf.set_struct(inf_cfg, False)        # allow adding/overriding keys
    inf_cfg.num_denoising_steps = STEPS         # generate_chunk reads this exact key
    inf_cfg.chunk_length = CHUNK_LEN
    print(f"[live_agent] num_denoising_steps = {STEPS }")
    print(f"[live_agent] num_chunk_len = {CHUNK_LEN}")
 
    # Optional native bf16 inference (the model was TRAINED in bf16 under DeepSpeed;
    # min_gru/LayerNorm/MPConv all upcast their fragile math to fp32 internally, so
    # bf16 weights are the native regime, ~2x the fp32 decoder). Opt-in via env.
    #if DTYPE in ("bf16", "bfloat16") and str(dev).startswith("cuda"):
     #   model = model.to(torch.bfloat16)
     #   print("[live_agent] world model cast to bf16")
 
    # Setup the online sampler
    target_modalities = [m.strip() for m in TARGET_MODALITIES.split(",") if m.strip()]
    print(f"[live_agent] target_modalities = {target_modalities}")
    from online_sampler import OnlineSampler  # lives in the model repo root
    sampler = OnlineSampler(
        model, inf_cfg,
        target_modalities=target_modalities,
        device=dev,
    )

    # --- load the audio context encoder to the gpu(Encodec 24kHz). Matches the corpus builder
    #     main_continuous_hdf5.py exactly (no set_target_bandwidth); the
    #     encode -> quantizer.decode round-trip lives in _encode_audio_inmemory. 
    encodec = _EncodecPkg.encodec_model_24khz().to(dev).eval()
    print("[live_agent] Encodec audio context encoder loaded")
 
    # --- load the SDXL VAE for live video encode 
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    ).to(dev).eval()
    try:
        #vae = vae.to(memory_format=torch.channels_last)   # faster conv kernels
        vae = vae.half()
    except Exception:
        print ("[live_agent] faster conversion kernels failed")
    
    # --- optional FAST VAE (TAESDXL): a tiny distilled autoencoder that produces
    #     SDXL-compatible latents ~100x faster than the full SDXL VAE. The full VAE
    #     encode (~390 ms/frame on a 3060) is the live loop's true bottleneck.
    #     We keep the SDXL VAE around for a one-time per-channel
    #     calibration (see _encode_frames) so the tiny VAE's latents are mapped onto
    #     the exact distribution the world model trained on, then free it. ------- #
    vae_fast = None
    if VAE_MODE in ("fast", "taesd", "taesdxl", "tiny"):
        vae_fast = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=torch.float16
        ).to(dev).eval()
        try:
            #vae = vae.to(memory_format=torch.channels_last)   # faster conv kernels
            pass
        except Exception:
            pass
        print("[live_agent] fast VAE (TAESDXL) loaded; will calibrate on first frame")
 
    # --- keypress AE for BOTH directions (encode + decode); loaded once -----  #
    ae = dkl._build_autoencoder(dkl.CHECKPOINT_DIR, device=dev)
    # --- forward map (key_id/button -> channel index) for in-memory encoding-- #
    from encode_key_press.scripts.constants import id_to_index, id_to_name
 
    # --- input backend (only when injecting locally; skip on a server) ------- #
    backend = None
    if make_backend:
        backend = inputInjector.init_injector()
 
    _S.update(
        torch=torch, sampler=sampler, vae=vae, encodec=encodec, ae=ae, dkl=dkl,
        vae_fast=vae_fast, vae_mode=("fast" if vae_fast is not None else "sdxl"),
        vae_calib=None,   # (scale[4,1,1], bias[4,1,1]) fitted on first encode
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
 
def _load_model(ckpt_path, device):
    """Mirror online_sampler.load_model but with an explicit config dir, and a
    memory-frugal load (the fp32 1.7B checkpoint is ~7 GB; we avoid holding two
    copies in RAM at once)."""
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
    # Load the checkpoint
    sd = _torch_load_low_mem(ckpt_path)
    # The for loop is filtering out unwanted string prefixes from the 
    # weight names (keys) so they perfectly match the model definition
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    clean = {}
    for k, v in sd.items():
        # pre looks for three common wrappers
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

def _torch_load_low_mem(ckpt_path):
    """Load a big checkpoint without copying the whole thing into RAM.
    mmap=True keeps weights file-backed (low CPU peak); fall back gracefully."""
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
 

def encode_keys_inmemory(key_events, click_events, t0, n_frames):
    """Raw key/click events -> (n_frames,79,10) -> {frame_idx: (16,5)} float32 CUDA tensors
    via the preloaded AE."""
    frames = _keys_to_multihot(key_events, click_events, t0, n_frames)
    #  disable gradient calculation, which saves memory and speeds up computations
    with torch.no_grad():
        z = _S["ae"].encoder(torch.from_numpy(frames).to(_S["device"]))   # (n_frames,16,5)
    z = z.float()
    return {i: z[i] for i in range(n_frames)}
 
 
def _keys_to_multihot(key_events, click_events, t0, n_frames):
    """Turns discrete input events into a dense (n_frames, 79, 10) array:."""
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


def mouse_bins_inmemory(mouse_events, t0, n_frames) -> dict[int, np.ndarray]:
    """Raw mouse-movement events -> {frame_idx: (2,10)}, binned + outlier-clipped."""
    out = {}
    frame_len = 1000.0 / LATENT_FPS                # 100 ms
    bin_w = frame_len / BINS_PER_FRAME             # 10 ms
    horizon = n_frames * frame_len
    for e in mouse_events:
        dt = int(e["timestamp"]) - t0
        if dt < 0 or dt >= horizon:
            continue
        # which frame the event falls in 
        fi = int(dt // frame_len)
        # which sub-bin  within the frame it falls in
        b = min(max(int((dt - fi * frame_len) // bin_w), 0), BINS_PER_FRAME - 1)
        # Fetch (or create) the (2, 10) array for this frame, then accumulate the event's deltas: mouseDX into row 0 at sub-bin b, mouseDY into row 1.
        f = out.setdefault(fi, np.zeros((2, BINS_PER_FRAME), dtype=np.float32))
        f[0, b] += float(e.get("mouseDX", 0.0))
        f[1, b] += float(e.get("mouseDY", 0.0))
    for f in out.values():
        f[0][(f[0] < CLIP_MOUSE_DX[0]) | (f[0] > CLIP_MOUSE_DX[1])] = 0.0
        f[1][(f[1] < CLIP_MOUSE_DY[0]) | (f[1] > CLIP_MOUSE_DY[1])] = 0.0
    return out


def encode_frames(frames_bgr) -> torch.Tensor:
    """Encode a list of BGR frames (any size) with the preloaded VAE.
    Resizes to the training resolution, letterbox-pads 720->768, returns
    (N,4,96,160) float32"""
    if not frames_bgr:
        return torch.zeros((0, *VIDEO_LATENT_SHAPE), dtype=torch.float32, device="cuda")

    # Preprocessing
    imgs = pad_and_transform_frames(frames_bgr, True, ENCODE_FRAME_WH, PAD_FRAME_WH).to("cuda")
 
    # Sets up an empty list to collect latent chunks. 
    lat = []
    if PROFILE_VAE  and torch.cuda.is_available():
        torch.cuda.synchronize(); _vt0 = time.time()
    if _S.get("vae_mode") == "fast" and _S.get("vae_fast") is not None:
        out = _encode_frames_fast(imgs)
    else:
        with torch.inference_mode():         # inference_mode() is like no_grad() but stricter/faster — no autograd tracking at all, no version counters.
            # Processes frames in batches of VAE_BATCH_SIZE to avoid running out of GPU memory when many frames come in at once.
            #       2 * chunk - 1 — rescales pixel values from [0, 1] to [-1, 1], the range the VAE was trained on.
            #       .latent_dist.sample() — the VAE encoder outputs a distribution (mean + variance per latent), and this draws a sample from it rather than taking the mean.
            #       * 0.13025 — the VAE scaling factor, which normalizes latents to roughly unit variance. This is the SDXL VAE's constant, so this is almost certainly the SDXL autoencoder (SD 1.5's is 0.18215).
            for i in range(0, imgs.shape[0], VAE_BATCH_SIZE ):
                chunk = imgs[i:i + VAE_BATCH_SIZE ]
                z = _S["vae"].encode(2 * chunk - 1).latent_dist.sample() * 0.13025
                lat.append(z.float())   # Casts to float32 (the VAE may run in fp16) and copies the latents from GPU to CPU, appending each chunk. 
        out = torch.cat(lat, 0)         # Concatenates all chunks along the batch dimension and converts to a numpy array.
    if PROFILE_VAE and torch.cuda.is_available():
        torch.cuda.synchronize()
        _dt = (time.time() - _vt0) * 1000
        print(f"[profile] VAE encode {imgs.shape[0]} frame(s) = {_dt:.0f}ms "
              f"({_dt / max(imgs.shape[0], 1):.0f}ms/frame)", flush=True)
    return out
 
def _encode_frames_fast(imgs) -> torch.Tensor:
    """Encode with TAESDXL (tiny VAE). On the first call, fit a per-channel
    scale+bias mapping TAESDXL latents -> the SDXL-VAE latent distribution the
    world model trained on (z = vae.encode(2x-1).sample()*0.13025), using the SDXL
    VAE that's still loaded; then free the SDXL VAE. After that it's pure TAESDXL +
    an elementwise affine -- single-digit ms/frame. Returns float32 CUDA tensor."""
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
        _S["vae_calib"] = (scale.to(imgs.device), bias.to(imgs.device))   # keep calib on GPU
        print(f"[live_agent] TAESDXL calibrated to SDXL latents "
              f"(per-channel scale {[round(v,3) for v in scale.flatten().tolist()]})", flush=True)
        try:
            _S["vae"] = None
            torch.cuda.empty_cache()
        except Exception:
            pass

    scale, bias = _S["vae_calib"]
    z = _tae_latents(imgs).float()
    return z * scale + bias                                        # stays on GPU
 
def pad_and_transform_frames(frames, use_fp16, original_frame_size, padded_frame_size):
    """Preprocessing: resizes frames to the training resolution, 
    letterbox-pads the height from 720 to 768 (VAEs typically need dimensions divisible by 8 or 64, and 720 isn't divisible by 64 — 768 is), c
    onverts to a torch tensor, and moves it onto the GPU."""
    padding = (0, (padded_frame_size[1] - original_frame_size[1]) // 2) 
    return torch.stack([
        F.to_tensor(
            F.pad(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), padding, fill=0)
        ).half() if use_fp16 else F.to_tensor(
            F.pad(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), padding, fill=0)
        )
        for frame in frames
    ])

 
def encode_audio_inmemory(pcm_hear, pcm_speak) -> torch.Tensor:
    """Two mono 24kHz float32 clips (~4800 samples) -> two (15,128) float32 CUDA
    tensors, matching main_continuous_hdf5.py: encode -> quantizer.decode -> (T,128)."""
    enc = _S["encodec"]
    
    def _emb(pcm):
        x = torch.as_tensor(pcm, dtype=torch.float32, device=_S["device"]).flatten() # converts the input to a float32 tensor on the GPU and flattens it to 1-D. as_tensor avoids a copy if pcm is already a suitable tensor.
        n = AUDIO_SAMPLES_PER_UNIT # The exact number of samples one unit must contain — ~4800 samples = 200 ms at 24 kHz, matching the 2×100 ms video frames per unit.
        x = torch.nn.functional.pad(x, (0, n - x.numel())) if x.numel() < n else x[:n] # Forces the clip to exactly n samples: too short → zero-pad on the right (the (0, n - x.numel()) means "0 zeros on the left, the shortfall on the right"); too long → truncate.
        with torch.inference_mode():
            # Reshapes to (batch=1, channels=1, samples=n) — the 3-D layout EnCodec expects — and encodes. 
            # EnCodec internally chunks audio into frames and returns a list of (codes, scale) pairs, where codes has shape (1, K, T): 
            # K codebooks (residual quantizer levels) × T time steps of discrete token IDs. For a 200 ms clip there's typically just one frame in the list.
            frames = enc.encode(x.view(1, 1, n))            # [(codes, scale)], codes:(1,K,T)
            zs = []
            for code, _scale in frames:
                z = enc.quantizer.decode(code.transpose(0, 1))   # (1,128,T)  <-- the key step
                zs.append(z)
            z = torch.cat(zs, dim=-1).squeeze(0).transpose(0, 1).contiguous()  # (T,128)
        if z.shape[0] < AUDIO_TOKENS_PER_UNIT:
            z = torch.nn.functional.pad(z, (0, 0, 0, AUDIO_TOKENS_PER_UNIT - z.shape[0]))
        return z[:AUDIO_TOKENS_PER_UNIT].float()             # (15,128) stays on GPU
    
    return _emb(pcm_hear), _emb(pcm_speak)

 
def _assemble_dataframe(u, video: torch.Tensor, key_by_frame: torch.Tensor, mouse_by_frame: dict[int, np.ndarray],  audio_hear :torch.Tensor =None, audio_speak : torch.Tensor=None):
    """Build ONE FullData unit with the exact post-collate shapes:
         video [1,1,2,4,96,160]  mouse [1,1,20,2]  key [1,1,10,16]."""
 
    f0 = u * VIDEO_FRAMES_PER_UNIT
    # video: 2 consecutive 100ms latents, (already a float32 CUDA tensor on the GPU)
    v = video[f0:f0 + VIDEO_FRAMES_PER_UNIT]                                   # (2,4,96,160)
    v = v.reshape(1, 1, VIDEO_FRAMES_PER_UNIT, *VIDEO_LATENT_SHAPE)            # [1,1,2,4,96,160]
 
    # key: 2 frames x (16,5) -> concat on time -> (16,10) -> (10,16) -> [1,1,10,16]
    _kzero = torch.zeros((KEY_LATENT_DIM, 5), dtype=torch.float32, device=_S["device"])
    kf = [key_by_frame.get(f0 + j, _kzero) for j in range(VIDEO_FRAMES_PER_UNIT)]
    k = torch.cat(kf, dim=1).float()                                          # (16,10)
    k = k.permute(1, 0).reshape(1, 1, KEYBOARD_TOKENS_PER_UNIT, KEY_LATENT_DIM)
 
    # mouse: 2 frames x (2,10) -> concat -> (2,20) -> (20,2) -> [1,1,20,2]
    mf = [mouse_by_frame.get(f0 + j, np.zeros((2, BINS_PER_FRAME), np.float32))
          for j in range(VIDEO_FRAMES_PER_UNIT)]
    m = torch.from_numpy(np.concatenate(mf, axis=1)).float()                  # (2,20)
    m = m.permute(1, 0).reshape(1, 1, MOUSE_TOKENS_PER_UNIT, 2)
    
    def _au(a):
        if a is None:
            a = torch.zeros((AUDIO_TOKENS_PER_UNIT, AUDIO_FEATURE_DIM),
                            dtype=torch.float32, device=_S["device"])
        return a.reshape(1, 1, AUDIO_TOKENS_PER_UNIT, AUDIO_FEATURE_DIM)
    
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
 
    dec = dkl.FullDataDecoder(device="cpu")

    return dec.decode(fd)
 
 
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