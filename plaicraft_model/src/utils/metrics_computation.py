#!/usr/bin/env python3
"""Metrics computation utilities.

This module contains the core functions for computing video, audio,
keypress and mouse movement metrics and the top-level `validate()` entrypoint.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.io
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from pytorch_fid import fid_score as fid
from pytorch_fid.inception import InceptionV3
import soundfile as sf
import torchaudio
from scipy import linalg
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Constants (kept as in original validator)
BIN_MS = 10
EXPECTED_KEYS_PER_FRAME = 79
EXPECTED_KEY_BIN_LEN = 10
EXPECTED_MM_SHAPE = (2, 10)

PSNR_SSIM_RANGE = 1.0
LPIPS_NORMALIZE = True

# Metric caps / clamps (JSON-safe, deterministic)
PSNR_CAP_DB = 120.0
SSIM_MIN, SSIM_MAX = 0.0, 1.0
LPIPS_MIN = 0.0
FID_MIN = 0.0
FAD_MIN = 0.0

# Audio embedding backend
AUDIO_BACKEND = "wav2vec2-base"
AUDIO_SR = 16000

# FID settings
FID_DIMS = 2048
FID_BATCH = 64
FID_EPS_START = 1e-6
FID_EPS_MAX = 1e-1
FID_IMAG_TOL = 1e-6

@dataclass(frozen=True)
class ClipPaths:
    base: Path
    gen_dir: Path
    gt_dir: Path
    gen_video: Path
    gt_video: Path
    gen_ai: Path
    gt_ai: Path
    gen_ao: Path
    gt_ao: Path
    gen_keys: Path
    gt_keys: Path
    gen_mouse: Path
    gt_mouse: Path


def _require(p: Path, desc: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {desc}: {p}")


def find_clip_paths(sample_root: Path) -> List[ClipPaths]:
    sample_root = Path(sample_root)
    if not sample_root.is_dir():
        raise NotADirectoryError(f"sample_root not a directory: {sample_root}")

    clips: List[ClipPaths] = []
    for dirpath, dirnames, _ in os.walk(sample_root):
        p = Path(dirpath)
        if "generated" in dirnames and "gt" in dirnames:
            gen_dir = p / "generated"
            gt_dir = p / "gt"

            gen_video = gen_dir / "video.mp4"
            gt_video = gt_dir / "video.mp4"
            gen_ai = gen_dir / "audio_speak.wav"
            gt_ai = gt_dir / "audio_speak.wav"
            gen_ao = gen_dir / "audio_hear.wav"
            gt_ao = gt_dir / "audio_hear.wav"
            gen_keys = gen_dir / "key_press.json"
            gt_keys = gt_dir / "key_press.json"
            gen_mouse = gen_dir / "mouse_movement.json"
            gt_mouse = gt_dir / "mouse_movement.json"

            _require(gen_video, "generated video")
            _require(gt_video, "gt video")
            _require(gen_ai, "generated audio_speak.wav")
            _require(gt_ai, "gt audio_speak.wav")
            _require(gen_ao, "generated audio_hear.wav")
            _require(gt_ao, "gt audio_hear.wav")
            _require(gen_keys, "generated key_press.json")
            _require(gt_keys, "gt key_press.json")
            _require(gen_mouse, "generated mouse_movement.json")
            _require(gt_mouse, "gt mouse_movement.json")

            clips.append(ClipPaths(
                base=p,
                gen_dir=gen_dir,
                gt_dir=gt_dir,
                gen_video=gen_video, gt_video=gt_video,
                gen_ai=gen_ai, gt_ai=gt_ai,
                gen_ao=gen_ao, gt_ao=gt_ao,
                gen_keys=gen_keys, gt_keys=gt_keys,
                gen_mouse=gen_mouse, gt_mouse=gt_mouse
            ))

    if not clips:
        raise RuntimeError(f"No clip folders found under {sample_root} with required generated/ and gt/ subfolders")
    return sorted(clips, key=lambda c: str(c.base))


def make_video_metrics(device: torch.device):
    psnr = PeakSignalNoiseRatio(data_range=PSNR_SSIM_RANGE).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=PSNR_SSIM_RANGE).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=LPIPS_NORMALIZE).to(device)
    return psnr, ssim, lpips


def _read_video_tensor(path: Path) -> torch.Tensor:
    frames, _, _ = torchvision.io.read_video(str(path), pts_unit="sec", output_format="TCHW")
    if frames.numel() == 0:
        raise RuntimeError(f"Empty video: {path}")
    frames = (frames.float() / 255.0).clamp(0.0, 1.0)
    return frames


def compute_video_clip_metrics(gen_video: Path, gt_video: Path, device: torch.device) -> Dict[str, float]:
    psnr_m, ssim_m, lpips_m = make_video_metrics(device)

    gen = _read_video_tensor(gen_video).to(device)
    gt  = _read_video_tensor(gt_video).to(device)

    if gen.shape[0] != gt.shape[0]:
        raise RuntimeError(f"Frame-count mismatch for {gen_video.name} vs {gt_video.name}: {gen.shape[0]} != {gt.shape[0]}")
    if gen.shape[1:] != gt.shape[1:]:
        raise RuntimeError(f"Frame-shape mismatch: {tuple(gen.shape[1:])} != {tuple(gt.shape[1:])}")

    T = gen.shape[0]
    psnrs: List[float] = []
    ssims: List[float] = []
    lpips_vals: List[float] = []

    for i in range(T):
        g = gen[i].unsqueeze(0)
        t = gt[i].unsqueeze(0)
        psnrs.append(float(psnr_m(g, t).item()))
        ssims.append(float(ssim_m(g, t).item()))
        lpips_vals.append(float(lpips_m(g, t).item()))

    psnr_m.reset(); ssim_m.reset(); lpips_m.reset()

    psnr_val = min(float(np.mean(psnrs)), PSNR_CAP_DB)
    ssim_val = float(np.mean(ssims))
    if ssim_val < SSIM_MIN:
        ssim_val = SSIM_MIN
    elif ssim_val > SSIM_MAX:
        ssim_val = SSIM_MAX
    lpips_val = float(np.mean(lpips_vals))
    if lpips_val < LPIPS_MIN:
        lpips_val = LPIPS_MIN

    return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}


def _save_all_frames_to_dirs(clips: List[ClipPaths]) -> Tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="fid_frames_"))
    real_dir = tmp / "real"
    fake_dir = tmp / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for cp in clips:
        for vpath, outdir in [(cp.gt_video, real_dir), (cp.gen_video, fake_dir)]:
            frames = _read_video_tensor(vpath)
            frames_u8 = (frames.cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
            for t, fr in enumerate(frames_u8):
                fp = outdir / f"{idx:08d}_{t:08d}.png"
                torchvision.io.write_png(torch.from_numpy(fr).permute(2, 0, 1), str(fp))
            idx += 1
    return real_dir, fake_dir


def _frechet_distance_stable(mu1: np.ndarray, sigma1: np.ndarray,
                             mu2: np.ndarray, sigma2: np.ndarray) -> float:
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)

    diff = mu1 - mu2
    eps = FID_EPS_START
    eye = np.eye(sigma1.shape[0], dtype=np.float64)

    while True:
        covmean, info = linalg.sqrtm((sigma1 + eps * eye) @ (sigma2 + eps * eye), disp=False)
        if np.isfinite(covmean).all() and not np.iscomplexobj(covmean):
            break
        eps *= 10.0
        if eps > FID_EPS_MAX:
            raise RuntimeError(f"FID sqrtm failed to converge (eps>{FID_EPS_MAX}). Provide more/less-degenerate frames.")

    if np.iscomplexobj(covmean):
        imag = np.max(np.abs(np.imag(covmean)))
        if imag > FID_IMAG_TOL:
            raise RuntimeError(f"FID sqrtm produced significant imaginary component ({imag}). Increase frame count/diversity.")
        covmean = np.real(covmean)

    tr_covmean = np.trace(covmean)
    fid_value = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return fid_value


def compute_fid_over_all_videos(clips: List[ClipPaths], device: torch.device) -> float:
    real_dir, fake_dir = _save_all_frames_to_dirs(clips)

    real_imgs = list(real_dir.glob("*.png"))
    fake_imgs = list(fake_dir.glob("*.png"))
    if len(real_imgs) < 2 or len(fake_imgs) < 2:
        shutil.rmtree(real_dir.parent, ignore_errors=True)
        raise RuntimeError(f"Need at least 2 frames in each set for FID, got real={len(real_imgs)}, fake={len(fake_imgs)}")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
    model = InceptionV3([block_idx]).to(device)
    mu1, sigma1 = fid.compute_statistics_of_path(str(real_dir), model, FID_BATCH, FID_DIMS, device, num_workers=0)
    mu2, sigma2 = fid.compute_statistics_of_path(str(fake_dir), model, FID_BATCH, FID_DIMS, device, num_workers=0)

    fid_value = _frechet_distance_stable(mu1, sigma1, mu2, sigma2)
    shutil.rmtree(real_dir.parent, ignore_errors=True)
    if fid_value < FID_MIN:
        fid_value = FID_MIN
    return fid_value


_W2V2_MODEL: Wav2Vec2Model | None = None
_W2V2_PROC: Wav2Vec2Processor | None = None


def _get_wav2vec2(device: torch.device) -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    global _W2V2_MODEL, _W2V2_PROC
    if _W2V2_MODEL is None or _W2V2_PROC is None:
        _W2V2_PROC = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        _W2V2_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    return _W2V2_PROC, _W2V2_MODEL


def _load_audio_mono_16k(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.size == 0:
        raise RuntimeError(f"Audio is empty: {path}")
    y = y.mean(axis=1)
    if sr != AUDIO_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_SR)
        with torch.no_grad():
            y = resampler(torch.from_numpy(y).unsqueeze(0)).squeeze(0).numpy()
    return y


def _embed_wav2vec2(y: np.ndarray, device: torch.device) -> np.ndarray:
    proc, model = _get_wav2vec2(device)
    inputs = proc(y, sampling_rate=AUDIO_SR, return_tensors="pt", padding=False)
    with torch.no_grad():
        out = model(inputs.input_values.to(device))
    hs = out.last_hidden_state
    if hs.shape[1] == 0:
        raise RuntimeError("Wav2Vec2 produced zero frames; input too short or invalid.")
    emb = hs.mean(dim=1).squeeze(0).detach().cpu().numpy()
    return emb


def _stats_from_embeddings(embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if embs.ndim != 2 or embs.shape[0] < 2:
        raise RuntimeError(f"Need at least 2 audio embeddings to compute covariance, got shape {embs.shape}")
    mu = embs.mean(axis=0)
    sigma = np.cov(embs, rowvar=False)
    return mu, sigma


def compute_fad_wav2vec2(gen_files: List[Path], gt_files: List[Path], device: torch.device) -> float:
    if len(gen_files) == 0 or len(gt_files) == 0:
        raise RuntimeError(f"Wav2Vec2-FAD: missing files. gen={len(gen_files)} gt={len(gt_files)}")

    gen_embs: List[np.ndarray] = []
    gt_embs: List[np.ndarray] = []

    for p in gt_files:
        y = _load_audio_mono_16k(p)
        gt_embs.append(_embed_wav2vec2(y, device))
    for p in gen_files:
        y = _load_audio_mono_16k(p)
        gen_embs.append(_embed_wav2vec2(y, device))

    gt_arr = np.stack(gt_embs, axis=0)
    gen_arr = np.stack(gen_embs, axis=0)

    mu1, sigma1 = _stats_from_embeddings(gt_arr)
    mu2, sigma2 = _stats_from_embeddings(gen_arr)
    val = _frechet_distance_stable(mu1, sigma1, mu2, sigma2)
    if val < FAD_MIN:
        val = FAD_MIN
    return val


def compute_w2v2_pair_distance(gen_file: Path, gt_file: Path, device: torch.device) -> float:
    gt_audio = _load_audio_mono_16k(gt_file)
    gen_audio = _load_audio_mono_16k(gen_file)
    gt_emb = _embed_wav2vec2(gt_audio, device)
    gen_emb = _embed_wav2vec2(gen_audio, device)
    dist = float(np.sum((gen_emb - gt_emb) ** 2))
    if dist < FAD_MIN:
        dist = FAD_MIN
    return dist


def _load_keypress_frames(path: Path) -> Tuple[List[Dict[str, List[int]]], int, Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "raw_decoded" not in obj or "frames" not in obj["raw_decoded"]:
        raise ValueError(f"Invalid key_press.json schema: {path}")
    frames = obj["raw_decoded"]["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in key_press.json: {path}")

    shape = tuple(obj["raw_decoded"].get("shape_per_frame", []))
    if shape and shape != (EXPECTED_KEYS_PER_FRAME, EXPECTED_KEY_BIN_LEN):
        raise ValueError(f"Unexpected key_press frame shape {shape}, expected {(EXPECTED_KEYS_PER_FRAME, EXPECTED_KEY_BIN_LEN)} at {path}")

    bin_ms = int(obj.get("bin_ms", BIN_MS))
    if bin_ms != BIN_MS:
        raise ValueError(f"BIN_MS mismatch in {path}: {bin_ms} != {BIN_MS}")

    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            raise ValueError(f"Frame {i} not a dict in {path}")
        if len(fr) != EXPECTED_KEYS_PER_FRAME:
            raise ValueError(f"Frame {i} got {len(fr)} keys, expected {EXPECTED_KEYS_PER_FRAME} in {path}")
        for k, arr in fr.items():
            if not isinstance(arr, list) or len(arr) != EXPECTED_KEY_BIN_LEN:
                raise ValueError(f"Frame {i} key '{k}' has length {len(arr)}, expected {EXPECTED_KEY_BIN_LEN} in {path}")
            for v in arr:
                if v not in (0, 1):
                    raise ValueError(f"Frame {i} key '{k}' has non-binary value {v} in {path}")

    return frames, bin_ms, (EXPECTED_KEYS_PER_FRAME, EXPECTED_KEY_BIN_LEN)


def _frames_to_binary_matrix(frames: List[Dict[str, List[int]]], key_order: List[str]) -> np.ndarray:
    T = len(frames) * EXPECTED_KEY_BIN_LEN
    K = len(key_order)
    out = np.zeros((K, T), dtype=np.int8)
    for i, fr in enumerate(frames):
        start = i * EXPECTED_KEY_BIN_LEN
        for ki, kname in enumerate(key_order):
            arr = fr[kname]
            out[ki, start:start + EXPECTED_KEY_BIN_LEN] = np.asarray(arr, dtype=np.int8)
    return out


def compute_keypress_mouseclick_metrics(gen_json: Path, gt_json: Path) -> Dict[str, float]:
    gen_frames, gen_bin_ms, _ = _load_keypress_frames(gen_json)
    gt_frames, gt_bin_ms, _ = _load_keypress_frames(gt_json)
    if gen_bin_ms != gt_bin_ms:
        raise RuntimeError(f"bin_ms mismatch: {gen_bin_ms} vs {gt_bin_ms} for {gen_json} / {gt_json}")
    if len(gen_frames) != len(gt_frames):
        raise RuntimeError(f"frame count mismatch: {len(gen_frames)} vs {len(gt_frames)} for {gen_json} / {gt_json}")

    key_order = list(gt_frames[0].keys())
    if sorted(key_order) != sorted(gen_frames[0].keys()):
        raise RuntimeError("Key sets differ between GT and GEN in key_press.json")

    gt_mat = _frames_to_binary_matrix(gt_frames, key_order)
    gen_mat = _frames_to_binary_matrix(gen_frames, key_order)
    if gt_mat.shape != gen_mat.shape:
        raise RuntimeError(f"Key matrix shape mismatch: {gt_mat.shape} vs {gen_mat.shape}")

    # Existing micro-Hamming metric over all key x time positions.
    diffs = (gt_mat != gen_mat).astype(np.int32)
    hamming = int(diffs.sum())
    total_positions = int(np.prod(gt_mat.shape))
    normalized = hamming / total_positions if total_positions > 0 else 0.0
    acc = 1.0 - normalized

    if normalized < 0.0:
        normalized = 0.0
    elif normalized > 1.0:
        normalized = 1.0
    if acc < 0.0:
        acc = 0.0
    elif acc > 1.0:
        acc = 1.0

    # Pressed-event metrics, ignoring true negatives.
    # This avoids the misleading ~0.99 accuracy when only one key is wrong
    # but the other 78 inactive keys are correctly zero.
    gt_bool = gt_mat.astype(bool)
    gen_bool = gen_mat.astype(bool)

    tp = int(np.logical_and(gt_bool, gen_bool).sum())
    fp = int(np.logical_and(~gt_bool, gen_bool).sum())
    fn = int(np.logical_and(gt_bool, ~gen_bool).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float(gt_bool.sum() == 0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else float(gen_bool.sum() == 0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "hamming_distance": hamming,
        "total_positions": total_positions,
        "normalized_hamming": float(normalized),
        "accuracy": float(acc),

        "pressed_precision": float(precision),
        "pressed_recall": float(recall),
        "pressed_f1": float(f1),
        "true_positive_pressed": tp,
        "false_positive_pressed": fp,
        "false_negative_pressed": fn,
        "gt_pressed_count": int(gt_bool.sum()),
        "gen_pressed_count": int(gen_bool.sum()),
    }


def _load_mouse_movement_frames(path: Path) -> Tuple[List[Dict[str, List[int]]], int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "raw_decoded" not in obj or "frames" not in obj["raw_decoded"]:
        raise ValueError(f"Invalid mouse_movement.json schema: {path}")
    frames = obj["raw_decoded"]["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in mouse_movement.json: {path}")
    bin_ms = int(obj.get("bin_ms", BIN_MS))
    if bin_ms != BIN_MS:
        raise ValueError(f"BIN_MS mismatch in {path}: {bin_ms} != {BIN_MS}")

    for i, fr in enumerate(frames):
        if sorted(fr.keys()) != ["dx", "dy"]:
            raise ValueError(f"Frame {i} in {path} must contain 'dx' and 'dy'")
        dx, dy = fr["dx"], fr["dy"]
        if not (isinstance(dx, list) and len(dx) == EXPECTED_MM_SHAPE[1] and
                isinstance(dy, list) and len(dy) == EXPECTED_MM_SHAPE[1]):
            raise ValueError(f"Frame {i} dx/dy must be lists of length {EXPECTED_MM_SHAPE[1]} in {path}")
        for v in dx + dy:
            if not isinstance(v, int):
                raise ValueError(f"Non-int mouse movement value {v} in {path}")

    return frames, bin_ms


def _flatten_mouse_series(frames: List[Dict[str, List[int]]]) -> Tuple[np.ndarray, np.ndarray]:
    dx_all: List[int] = []
    dy_all: List[int] = []
    for fr in frames:
        dx_all.extend(fr["dx"])
        dy_all.extend(fr["dy"])
    return np.asarray(dx_all, dtype=np.int32), np.asarray(dy_all, dtype=np.int32)


def compute_mouse_movement_metrics(gen_json: Path, gt_json: Path) -> Dict[str, float]:
    gen_frames, gen_bin_ms = _load_mouse_movement_frames(gen_json)
    gt_frames, gt_bin_ms = _load_mouse_movement_frames(gt_json)
    if gen_bin_ms != gt_bin_ms:
        raise RuntimeError(f"bin_ms mismatch in mouse movement: {gen_bin_ms} vs {gt_bin_ms}")
    if len(gen_frames) != len(gt_frames):
        raise RuntimeError(f"frame count mismatch in mouse movement: {len(gen_frames)} vs {len(gt_frames)}")

    gen_dx, gen_dy = _flatten_mouse_series(gen_frames)
    gt_dx, gt_dy = _flatten_mouse_series(gt_frames)
    if gen_dx.shape != gt_dx.shape or gen_dy.shape != gt_dy.shape:
        raise RuntimeError("mouse dx/dy length mismatch between GT and GEN")

    gt_xy = np.stack([gt_dx, gt_dy], axis=1)
    gen_xy = np.stack([gen_dx, gen_dy], axis=1)

    errs = np.linalg.norm(gen_xy - gt_xy, axis=1)
    ide = float(errs[0])
    ade = float(errs.mean())
    fde = float(errs[-1])

    gt_segs = np.linalg.norm(gt_xy[1:] - gt_xy[:-1], axis=1)
    gen_segs = np.linalg.norm(gen_xy[1:] - gen_xy[:-1], axis=1)
    pld = float(abs(gen_segs.sum() - gt_segs.sum()))

    if ide < 0.0: ide = 0.0
    if ade < 0.0: ade = 0.0
    if fde < 0.0: fde = 0.0
    if pld < 0.0: pld = 0.0

    return {"IDE": ide, "ADE": ade, "FDE": fde, "PLD": pld}


def validate(sample_root: Path) -> Dict:
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = find_clip_paths(Path(sample_root))
    print(f"[metrics] Found {len(clips)} samples. Computing metrics...", file=sys.stderr)
    sys.stderr.flush()

    video_psnr: List[float] = []
    video_ssim: List[float] = []
    video_lpips: List[float] = []

    kp_hamming_sum = 0
    kp_total_positions_sum = 0
    kp_norm_per_clip: List[float] = []
    kp_acc_per_clip: List[float] = []

    mm_IDE: List[float] = []
    mm_ADE: List[float] = []
    mm_FDE: List[float] = []
    mm_PLD: List[float] = []

    individual_results: List[Dict] = []

    gen_ai_files: List[Path] = []
    gt_ai_files: List[Path] = []
    gen_ao_files: List[Path] = []
    gt_ao_files: List[Path] = []

    for cp in clips:
        v_metrics = compute_video_clip_metrics(cp.gen_video, cp.gt_video, device)
        video_psnr.append(v_metrics["psnr"])
        video_ssim.append(v_metrics["ssim"])
        video_lpips.append(v_metrics["lpips"])

        gen_ai_files.append(cp.gen_ai)
        gt_ai_files.append(cp.gt_ai)
        gen_ao_files.append(cp.gen_ao)
        gt_ao_files.append(cp.gt_ao)

        kp = compute_keypress_mouseclick_metrics(cp.gen_keys, cp.gt_keys)
        kp_hamming_sum += int(kp["hamming_distance"])
        kp_total_positions_sum += int(kp["total_positions"])
        kp_norm_per_clip.append(float(kp["normalized_hamming"]))
        kp_acc_per_clip.append(float(kp["accuracy"]))

        mm = compute_mouse_movement_metrics(cp.gen_mouse, cp.gt_mouse)
        mm_IDE.append(mm["IDE"]); mm_ADE.append(mm["ADE"]); mm_FDE.append(mm["FDE"]); mm_PLD.append(mm["PLD"])

        individual_results.append({
            "clip_dir": str(cp.base),
            "video": v_metrics,
            "keypress_mouseclick": kp,
            "mouse_movement": mm,
        })

    import sys
    print("[metrics] Computing FID...", file=sys.stderr)
    sys.stderr.flush()
    fid_value = compute_fid_over_all_videos(clips, device)
    if fid_value < FID_MIN:
        fid_value = FID_MIN

    print("[metrics] Computing FAD (speaking)...", file=sys.stderr)
    sys.stderr.flush()
    fad_speaking = compute_fad_wav2vec2(gen_ai_files, gt_ai_files, device)
    if fad_speaking < FAD_MIN:
        fad_speaking = FAD_MIN

    print("[metrics] Computing FAD (hearing)...", file=sys.stderr)
    sys.stderr.flush()
    fad_hearing = compute_fad_wav2vec2(gen_ao_files, gt_ao_files, device)
    if fad_hearing < FAD_MIN:
        fad_hearing = FAD_MIN

    psnr_avg = float(np.mean(video_psnr))
    if psnr_avg > PSNR_CAP_DB:
        psnr_avg = PSNR_CAP_DB
    ssim_avg = float(np.mean(video_ssim))
    if ssim_avg < SSIM_MIN:
        ssim_avg = SSIM_MIN
    elif ssim_avg > SSIM_MAX:
        ssim_avg = SSIM_MAX
    lpips_avg = float(np.mean(video_lpips))
    if lpips_avg < LPIPS_MIN:
        lpips_avg = LPIPS_MIN

    nh_global = float(kp_hamming_sum / kp_total_positions_sum) if kp_total_positions_sum > 0 else 0.0
    if nh_global < 0.0:
        nh_global = 0.0
    elif nh_global > 1.0:
        nh_global = 1.0
    acc_global = 1.0 - nh_global

    nh_clip_avg = float(np.mean(kp_norm_per_clip)) if kp_norm_per_clip else 0.0
    if nh_clip_avg < 0.0:
        nh_clip_avg = 0.0
    elif nh_clip_avg > 1.0:
        nh_clip_avg = 1.0
    acc_clip_avg = float(np.mean(kp_acc_per_clip)) if kp_acc_per_clip else 0.0
    if acc_clip_avg < 0.0:
        acc_clip_avg = 0.0
    elif acc_clip_avg > 1.0:
        acc_clip_avg = 1.0

    ide_mean = float(np.mean(mm_IDE)); ade_mean = float(np.mean(mm_ADE))
    fde_mean = float(np.mean(mm_FDE)); pld_mean = float(np.mean(mm_PLD))
    if ide_mean < 0.0: ide_mean = 0.0
    if ade_mean < 0.0: ade_mean = 0.0
    if fde_mean < 0.0: fde_mean = 0.0
    if pld_mean < 0.0: pld_mean = 0.0

    report: Dict = {
        "video": {
            "psnr": psnr_avg,
            "ssim": ssim_avg,
            "lpips": lpips_avg,
            "fid": float(fid_value),
        },
        "audio": {
            "backend": {"name": AUDIO_BACKEND, "sr": AUDIO_SR},
            "speaking": {"frechet_w2v2": float(fad_speaking)},
            "hearing":  {"frechet_w2v2": float(fad_hearing)},
        },
        "keypress_mouseclick": {
            "hamming_distance": int(kp_hamming_sum),
            "total_positions": int(kp_total_positions_sum),
            "normalized_hamming": nh_global,
            "accuracy": acc_global,
            "normalized_hamming_per_clip_avg": nh_clip_avg,
            "accuracy_per_clip_avg": acc_clip_avg,
        },
        "mouse_movement": {
            "IDE_mean": ide_mean, "IDE_min": float(np.min(mm_IDE)),
            "ADE_mean": ade_mean, "ADE_min": float(np.min(mm_ADE)),
            "FDE_mean": fde_mean, "FDE_min": float(np.min(mm_FDE)),
            "PLD_mean": pld_mean, "PLD_min": float(np.min(mm_PLD)),
        },
        "individual_results": individual_results,
        "meta": {
            "device": device.type,
            "bin_ms": BIN_MS,
            "keys_per_frame": EXPECTED_KEYS_PER_FRAME,
            "key_bins_per_frame": EXPECTED_KEY_BIN_LEN,
            "mouse_frame_shape": EXPECTED_MM_SHAPE,
            "num_clips": len(clips),
        }
    }
    return report


def main():
    p = argparse.ArgumentParser(description="Compute semantic evaluation metrics for Plaicraft samples folder.")
    p.add_argument("--sample_root", type=str, required=True, help="Root containing <session_id>/r_*/{generated,gt}")
    p.add_argument("--save_name", type=str, default="semantic_evaluation_report.json", help="Saved JSON inside sample_root.")
    args = p.parse_args()

    sample_root = Path(args.sample_root).resolve()
    report = validate(sample_root)

    out_path = sample_root / args.save_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote report → {out_path}")

    v = report["video"]
    print(
        f"Video: PSNR={v['psnr']:.4f} SSIM={v['ssim']:.4f} LPIPS={v['lpips']:.4f}  FID={v['fid']:.4f}\n"
        f"Audio (W2V2-FD): speaking={report['audio']['speaking']['frechet_w2v2']:.6f} "
        f"hearing={report['audio']['hearing']['frechet_w2v2']:.6f}\n"
        f"Keys+Clicks: norm_hamming={report['keypress_mouseclick']['normalized_hamming']:.6f} "
        f"acc={report['keypress_mouseclick']['accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
