#!/usr/bin/env python3
"""Database-driven semantic evaluation metrics computation.

This module computes metrics per-clip based on the semantic evaluation database,
rather than computing all metrics for all clips. Each clip has a designated metric
in the database (Acc_R, D_K, etc.), and only that metric is computed.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

# Constants
BIN_MS = 10
EXPECTED_KEYS_PER_FRAME = 79
EXPECTED_KEY_BIN_LEN = 10
EXPECTED_MM_SHAPE = (2, 10)

PSNR_SSIM_RANGE = 1.0
LPIPS_NORMALIZE = True

# Metric caps / clamps
PSNR_CAP_DB = 120.0
SSIM_MIN, SSIM_MAX = 0.0, 1.0
LPIPS_MIN = 0.0
FID_MIN = 0.0
FAD_MIN = 0.0

AUDIO_BACKEND = "wav2vec2-base"
AUDIO_SR = 16000

FID_DIMS = 2048
FID_BATCH = 64
FID_EPS_START = 1e-6
FID_EPS_MAX = 1e-1
FID_IMAG_TOL = 1e-6

# Supported metrics
METRIC_D_K = "D_K"  # Hamming distance on keyboard events
METRIC_ACC_R = "Acc_R"  # Accuracy on speech responses


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


@dataclass
class ClipMetadata:
    """Metadata for a clip from the semantic evaluation database."""
    num: str  # Prompt number
    test_type: str
    prompt: str
    response: str
    metric: str  # Which metric to compute (D_K, Acc_R, etc.)
    session_id: str
    r_start_ms: int
    r_duration_ms: int


def _require(p: Path, desc: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {desc}: {p}")


def find_clip_paths(sample_root: Path) -> List[ClipPaths]:
    """Find all clip directories with generated/ and gt/ subdirectories."""
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


def load_clip_metadata_from_db(db_path: Path, sample_root: Path) -> Dict[str, ClipMetadata]:
    """Load metadata for all clips from metadata.json files saved during evaluation.

    Each sample directory should contain a metadata.json file written by the evaluation
    pipeline. Prompt and Response are looked up from the database using the `num` field.
    """
    metadata: Dict[str, ClipMetadata] = {}

    db_prompts: Dict[str, Tuple[str, str]] = {}
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT Num, Prompt, Response
                FROM prompts
                """
            )
            for row in cursor.fetchall():
                num = str(row["Num"])
                prompt = str(row["Prompt"]) if row["Prompt"] else ""
                response = str(row["Response"]) if row["Response"] else ""
                db_prompts[num] = (prompt, response)
            conn.close()
        except Exception as e:
            print(f"[metrics] Warning: Could not load prompts/responses from database: {e}")

    samples_dir = Path(sample_root) / "samples"
    if not samples_dir.is_dir():
        samples_dir = Path(sample_root)

    for sample_dir in sorted(samples_dir.iterdir()) if samples_dir.is_dir() else []:
        if not sample_dir.is_dir():
            continue
        if not sample_dir.name.startswith("sample_"):
            continue

        metadata_file = sample_dir / "metadata.json"
        if not metadata_file.is_file():
            continue

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
        except Exception as e:
            print(f"[metrics] Warning: Failed to load metadata.json from {sample_dir}: {e}")
            continue

        try:
            num = str(meta_dict.get("num", ""))
            test_type = str(meta_dict.get("test_type", ""))
            metric = str(meta_dict.get("metric", ""))
            session_id = str(meta_dict.get("session_id", ""))
            r_start_ms = int(meta_dict.get("response_start_ms", meta_dict.get("r_start_ms", 0)))
            r_duration_ms = int(meta_dict.get("response_duration_ms", meta_dict.get("r_duration_ms", 0)))
            prompt, response = db_prompts.get(num, ("", ""))

            metadata[sample_dir.name] = ClipMetadata(
                num=num,
                test_type=test_type,
                prompt=prompt,
                response=response,
                metric=metric,
                session_id=session_id,
                r_start_ms=r_start_ms,
                r_duration_ms=r_duration_ms,
            )
        except Exception as e:
            print(f"[metrics] Warning: Failed to parse metadata from {metadata_file}: {e}")
            continue

    return metadata


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


def compute_acc_r_metric(gen_audio: Path, gt_audio: Path, device: torch.device) -> Dict[str, float]:
    """Compute accuracy on speech responses using Wav2Vec2 embeddings.
    
    Acc_R: Exact-match accuracy on automatically transcribed verbal responses.
    For now, we compute embedding similarity as a proxy.
    """
    gt_audio_data = _load_audio_mono_16k(gt_audio)
    gen_audio_data = _load_audio_mono_16k(gen_audio)
    
    gt_emb = _embed_wav2vec2(gt_audio_data, device)
    gen_emb = _embed_wav2vec2(gen_audio_data, device)
    
    # Compute cosine similarity between embeddings
    similarity = float(np.dot(gt_emb, gen_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(gen_emb) + 1e-8))
    similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    return {"accuracy": similarity}


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


def compute_clip_metrics(
    clip_paths: ClipPaths,
    metadata: ClipMetadata,
    device: torch.device,
    compute_full_metrics: bool = False,
) -> Dict[str, Any]:
    """Compute metrics for a single clip based on its designated metric.
    
    Args:
        clip_paths: Paths to clip files
        metadata: Metadata from database (including which metric to compute)
        device: Torch device
        compute_full_metrics: Whether to compute all available metrics (for analysis)
    
    Returns:
        Dictionary with results for the requested metric(s)
    """
    results: Dict[str, Any] = {
        "num": metadata.num,
        "test_type": metadata.test_type,
        "prompt": metadata.prompt,
        "response": metadata.response,
        "metric_requested": metadata.metric,
    }
    
    metric_result: Dict[str, float] = {}
    
    # Compute the requested metric
    if metadata.metric == METRIC_D_K:
        metric_result = compute_keypress_mouseclick_metrics(
            clip_paths.gen_keys, clip_paths.gt_keys
        )
    elif metadata.metric == METRIC_ACC_R:
        metric_result = compute_acc_r_metric(
            clip_paths.gen_ai, clip_paths.gt_ai, device
        )
    else:
        raise ValueError(f"Unknown metric type: {metadata.metric}")
    
    results["metric_result"] = metric_result
    
    # Optionally compute full metrics for analysis
    if compute_full_metrics:
        try:
            video_metrics = compute_video_clip_metrics(
                clip_paths.gen_video, clip_paths.gt_video, device
            )
            results["video_metrics"] = video_metrics
        except Exception as e:
            results["video_metrics_error"] = str(e)
    
    return results


def validate_db_driven(
    sample_root: Path,
    db_path: Path,
    device: Optional[torch.device] = None,
    compute_full_metrics: bool = False,
) -> Dict[str, Any]:
    """Database-driven semantic evaluation metrics.

    Args:
        sample_root: Root directory containing sample clips
        db_path: Path to semantic evaluation database
        device: Torch device (default: cuda if available, else cpu)
        compute_full_metrics: Whether to compute all modality metrics (default: False)

    Returns:
        Report dictionary with:
        - clip_results: List of per-clip results
        - summary: Aggregated statistics
        - metadata: Computation metadata
    """
    import sys
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sample_root = Path(sample_root)
    db_path = Path(db_path)
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    clips = find_clip_paths(sample_root)
    metadata_map = load_clip_metadata_from_db(db_path, sample_root)

    print(f"[metrics] Found {len(clips)} clips. Computing metrics...", file=sys.stderr)

    sys.stderr.flush()
    
    clip_results: List[Dict[str, Any]] = []
    metric_counts: Dict[str, int] = {}
    
    for clip_path in clips:
        clip_name = clip_path.base.name
        if clip_name not in metadata_map:
            print(f"[metrics] Warning: No metadata for clip {clip_name}", file=sys.stderr)
            continue
        
        meta = metadata_map[clip_name]
        metric_counts[meta.metric] = metric_counts.get(meta.metric, 0) + 1
        
        try:
            result = compute_clip_metrics(
                clip_path, meta, device, compute_full_metrics=compute_full_metrics
            )
            
            clip_results.append(result)
        except Exception as e:
            print(f"[metrics] Error computing metrics for clip {clip_name}: {e}", file=sys.stderr)
            sys.stderr.flush()
            clip_results.append({
                "num": meta.num,
                "test_type": meta.test_type,
                "prompt": meta.prompt,
                "response": meta.response,
                "metric_requested": meta.metric,
                "error": str(e),
            })
    
    # Summary statistics
    summary: Dict[str, Any] = {
        "total_clips": len(clip_results),
        "metric_distribution": metric_counts,
    }
    
    # Compute per-metric statistics
    for metric_type in set(metric_counts.keys()):
        results_for_metric = [
            r for r in clip_results 
            if r.get("metric_requested") == metric_type and "error" not in r
        ]
        if results_for_metric:
            summary[f"{metric_type}_count"] = len(results_for_metric)
    
    report: Dict[str, Any] = {
        "clip_results": clip_results,
        "summary": summary,
        "meta": {
            "device": device.type,
            "compute_full_metrics": compute_full_metrics,
            "database_path": str(db_path),
        }
    }
    
    return report


def main():
    p = argparse.ArgumentParser(
        description="Compute database-driven semantic evaluation metrics."
    )
    p.add_argument(
        "--sample_root",
        type=str,
        required=True,
        help="Root containing <session_id>/r_*/{generated,gt}"
    )
    p.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to semantic evaluation database"
    )
    p.add_argument(
        "--save_name",
        type=str,
        default="semantic_evaluation_report.json",
        help="Saved JSON inside sample_root."
    )
    p.add_argument(
        "--compute_full_metrics",
        action="store_true",
        help="Compute all modality metrics (not just requested)"
    )
    args = p.parse_args()

    sample_root = Path(args.sample_root).resolve()
    db_path = Path(args.db_path).resolve()
    
    report = validate_db_driven(
        sample_root,
        db_path,
        compute_full_metrics=args.compute_full_metrics,
    )

    out_path = sample_root / args.save_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote report → {out_path}")


if __name__ == "__main__":
    main()
