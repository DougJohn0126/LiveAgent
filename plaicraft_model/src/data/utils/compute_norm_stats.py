#!/usr/bin/env python3
"""
Compute masked, post-normalization stats for Plaicraft latents and suggest EDM sigma_data values.

Fixes:
- Uses boolean-broadcast + masked_select (no advanced indexing) to avoid IndexError.
- Writes an up-to-date JSON **after every batch** (configurable via --save_every).
- Matches training: builds latents like AbstractDenoiser.format_data_dict, applies scale_minmax,
  and supports both explicit masks and all-valid batches, with optional robust clipping to [-1, 1].

Usage example:
  python normalization_param.py --clip_for_std --save_every 10 --max_batches 20000
"""
import argparse
import json
import logging
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project imports (keep paths as in your repo)
from data.datasets.iterstyle import IterStyleDataset
from data.samplers.dynamic_length import DynamicLengthBatchSampler
from data.datasets.mapstyle import MapStyleDataset
from utils.normalization import scale_minmax, NORMALIZATIONS

# ---------------------------
# RunningStats
# ---------------------------
class RunningStats:
    """
    Tracks min, max, mean, std incrementally, plus an approximate reservoir
    for computing quantiles. Keeps memory bounded for large datasets.
    """
    def __init__(self, reservoir_size=10_000, samples_per_batch=100):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = float('inf')
        self.max_val = -float('inf')
        self.reservoir = []
        self.reservoir_size = reservoir_size
        self.samples_per_batch = samples_per_batch

    def update_np(self, arr: np.ndarray):
        if arr.size == 0:
            return
        arr = arr.astype(np.float64, copy=False)
        self.count += arr.size
        self.sum += arr.sum(dtype=np.float64)
        self.sum_sq += np.square(arr, dtype=np.float64).sum(dtype=np.float64)
        self.min_val = min(self.min_val, float(arr.min()))
        self.max_val = max(self.max_val, float(arr.max()))

        # reservoir sample
        if arr.size <= self.samples_per_batch:
            batch_samples = arr
        else:
            idx = np.random.choice(arr.size, self.samples_per_batch, replace=False)
            batch_samples = arr[idx]
        self.reservoir.extend(batch_samples.tolist())
        if len(self.reservoir) > self.reservoir_size:
            self.reservoir = list(
                np.random.choice(self.reservoir, self.reservoir_size, replace=False)
            )

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"min": None, "max": None, "mean": None, "std": None, "quantiles": None}
        mean = self.sum / self.count
        var = (self.sum_sq / self.count) - mean ** 2
        var = max(var, 0.0)
        std = float(np.sqrt(var))
        if self.reservoir:
            res = np.array(self.reservoir, dtype=np.float64)
            quantiles = {
                "25": float(np.percentile(res, 25)),
                "50": float(np.percentile(res, 50)),
                "75": float(np.percentile(res, 75)),
            }
        else:
            quantiles = None
        return {
            "min": float(self.min_val),
            "max": float(self.max_val),
            "mean": float(mean),
            "std": std,
            "quantiles": quantiles,
        }

# ---------------------------
# Helpers
# ---------------------------
def _masked_flat(x: torch.Tensor, valid_mask: torch.BoolTensor) -> torch.Tensor:
    """
    Return a 1D tensor of elements from x where valid_mask is True.
    valid_mask shape [B,T]; x shape [B,T,...].
    Uses boolean broadcasting + masked_select to avoid advanced indexing pitfalls.
    """
    # Ensure bool and on same device
    m = valid_mask.to(dtype=torch.bool, device=x.device)
    # Broadcast mask to x’s rank
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    m = m.expand_as(x)
    return x.masked_select(m)

def _format_data_dict(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    EXACTLY like AbstractDenoiser.format_data_dict (latent-key space).
    """
    frame_latent = batch["video"].flatten(2, 3)  # (B,T,F*C,H,W)
    audio_speak_latent = batch["audio_speak"]          # (B,T,L,D)
    audio_hear_latent = batch["audio_hear"]        # (B,T,L,D)
    keyboard_latent = batch["key_press"].flatten(2, 3)
    mouse_latent = batch["mouse_movement"].flatten(2, 3)
    return dict(
        frame_latent=frame_latent,
        audio_speak_latent=audio_speak_latent,
        audio_hear_latent=audio_hear_latent,
        keyboard_latent=keyboard_latent,
        mouse_latent=mouse_latent,
    )

def _build_sigma_suggestions(std_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Map latent keys to trainer arg names.
    """
    mapping = {
        "frame_latent": "sigma_data_video",
        "audio_speak_latent": "sigma_data_audio_speak",
        "audio_hear_latent": "sigma_data_audio_hear",
        "keyboard_latent": "sigma_data_key_press",
        "mouse_latent": "sigma_data_mouse_movement",
    }
    out = {}
    for k, v in std_dict.items():
        if v is None:
            continue
        arg = mapping.get(k)
        if arg:
            out[arg] = float(v)
    return out

def _dump_partial_json(path: str,
                       post_stats: Dict[str, RunningStats],
                       raw_stats: Dict[str, RunningStats] | None,
                       batches_done: int,
                       args) -> None:
    """
    Write the current snapshot to JSON so progress is never lost.
    """
    post = {k: v.finalize() for k, v in post_stats.items()}
    raw = {k: v.finalize() for k, v in raw_stats.items()} if raw_stats is not None else None
    std_post = {k: (post[k]["std"] if post[k]["std"] is not None else None) for k in post.keys()}
    sigma_suggestion = _build_sigma_suggestions(std_post)

    out = {
        "batches_processed": batches_done,
        "normalizations_used": NORMALIZATIONS,
        "post_normalization_stats_masked": post,
        "raw_stats_masked": raw,
        "sigma_data_suggestion": sigma_suggestion,
        "config": {
            "clip_for_std": args.clip_for_std,
            "max_batches": args.max_batches,
            "num_workers": args.num_workers,
            "save_every": args.save_every,
        }
    }
    # Atomic write: write to temp then replace
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    import os
    os.replace(tmp, path)

# ---------------------------
# Core computation
# ---------------------------
def compute_stats(args) -> Dict[str, Any]:
    logging.info("Initializing dataset...")
    dataset = MapStyleDataset(args, seed=args.seed)
    sampler = DynamicLengthBatchSampler(args, seed=args.seed)
    logging.info("Creating DataLoader...")
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=dataset.collate_fn,
    )

    # Stats in latent-key space, AFTER scale_minmax
    keys = ["frame_latent", "audio_speak_latent", "audio_hear_latent", "keyboard_latent", "mouse_latent"]
    post_stats = {k: RunningStats() for k in keys}

    # Optional raw stats BEFORE normalization (masked) for debugging
    if args.compute_raw:
        raw_keys = ["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"]
        raw_stats = {k: RunningStats() for k in raw_keys}
    else:
        raw_stats = None

    batches_done = 0
    with torch.no_grad():
        for batch in loader:
            batches_done += 1
            try:
                if "dataframe_indices" not in batch or batch["dataframe_indices"] is None:
                    raise ValueError("Expected dataframe_indices in batch.")
                vm = batch["dataframe_indices"].ge(0)

                # Build SAME latents and apply SAME normalization
                latents = _format_data_dict(batch)
                latents = scale_minmax(latents)

                # Optional robust clipping before std
                if args.clip_for_std:
                    for k in latents:
                        latents[k] = latents[k].clamp_(-1, 1)

                # Update post-normalization stats using ONLY valid timesteps
                for k, v in latents.items():
                    vals = _masked_flat(v, vm).float().cpu().numpy()
                    post_stats[k].update_np(vals)

                # Optional raw stats (masked)
                if raw_stats is not None:
                    raw_stats["video"].update_np(_masked_flat(batch["video"], vm).float().cpu().numpy())
                    raw_stats["audio_speak"].update_np(_masked_flat(batch["audio_speak"], vm).float().cpu().numpy())
                    raw_stats["audio_hear"].update_np(_masked_flat(batch["audio_hear"], vm).float().cpu().numpy())
                    kp = batch["key_press"]
                    mm = batch["mouse_movement"]
                    raw_stats["key_press"].update_np(_masked_flat(kp, vm).float().cpu().numpy())
                    raw_stats["mouse_movement"].update_np(_masked_flat(mm, vm).float().cpu().numpy())

                if batches_done % max(1, args.log_every) == 0:
                    logging.info(f"Processed {batches_done} batches...")

                # Save a snapshot every batch (or every N batches)
                if (batches_done % args.save_every) == 0:
                    _dump_partial_json(args.output_file, post_stats, raw_stats, batches_done, args)

                if batches_done >= args.max_batches:
                    logging.info(f"Reached max_batches={args.max_batches}, stopping early.")
                    break

            except Exception as e:
                logging.exception(f"Error on batch {batches_done}: {e}")
                # Always dump what we have so far
                _dump_partial_json(args.output_file, post_stats, raw_stats, batches_done, args)
                if args.stop_on_error:
                    raise
                # else continue to next batch

    # Final dump
    _dump_partial_json(args.output_file, post_stats, raw_stats, batches_done, args)

    # Return the final snapshot (already written)
    post = {k: v.finalize() for k, v in post_stats.items()}
    raw = {k: v.finalize() for k, v in raw_stats.items()} if raw_stats is not None else None
    std_post = {k: (post[k]["std"] if post[k]["std"] is not None else None) for k in post.keys()}
    return {
        "batches_processed": batches_done,
        "normalizations_used": NORMALIZATIONS,
        "post_normalization_stats_masked": post,
        "raw_stats_masked": raw,
        "sigma_data_suggestion": _build_sigma_suggestions(std_post),
    }

# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="Compute masked post-normalization stats and sigma_data suggestions.")
    # Dataset + Sampler arguments
    IterStyleDataset.add_command_line_options(p)
    DynamicLengthBatchSampler.add_command_line_options(p)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_file", type=str, default="norm_stats_postnorm_masked.json")
    p.add_argument("--clip_for_std", action="store_true",
                   help="Compute std on values clipped to [-1,1] (robust). Recommended.")
    p.add_argument("--compute_raw", action="store_true",
                   help="Also compute masked raw stats in original spaces (for debugging).")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=10,
                   help="Write JSON snapshot every N batches (1 = every batch).")
    p.add_argument("--stop_on_error", action="store_true",
                   help="If set, re-raise on batch error after writing a snapshot.")

    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Determinism for reservoir sampling
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = compute_stats(args)
    logging.info("==== POST-NORMALIZATION (masked) STD ====")
    for k, v in out["post_normalization_stats_masked"].items():
        logging.info(f"{k}: std={v['std']} min={v['min']} max={v['max']}")
    logging.info("==== SUGGESTED TRAINER FLAGS (sigma_data_*) ====")
    for argname, val in out["sigma_data_suggestion"].items():
        logging.info(f"--{argname} {val}")

if __name__ == "__main__":
    main()