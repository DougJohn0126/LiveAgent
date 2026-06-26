"""
Normalization utilities for multi-modal latent representations.

This module provides scaling functions for normalizing latent representations
across different modalities (video, audio, actions) during training and inference.
"""

import torch
from utils.constants import MODALITY_TO_LATENT


# Pre-computed normalization statistics for each modality's latent representation
NORMALIZATIONS = {
    'frame_latent': (-4.79296875, 5.098236083984375),
    'audio_speak_latent': (-28.728960037231445, 35.23261642456055),
    'audio_hear_latent': (-47.9345588684082, 50.30904006958008),
    'keyboard_latent': (-1.1530787944793701, 1.148032784461975),
    'mouse_latent': (-150.0, 150.0)
}

# Z-score statistics (mean, std) for each latent modality
ZSCORE_STATS = {
    'frame_latent': {'mean': -0.26, 'std': 1.03},
    'audio_speak_latent': {'mean': -0.53, 'std': 3.98},
    'audio_hear_latent': {'mean': -0.53, 'std': 4.08},
    'keyboard_latent': {'mean': -0.02, 'std': 0.13},
    'mouse_latent': {'mean': 0.05, 'std': 5.13},
}

TARGET_STD = 0.5
_EPS = 1e-8


def _get_scalar(t, x):
    """Create a scalar tensor (broadcastable) on x's device/dtype."""
    return torch.as_tensor(t, device=x.device, dtype=x.dtype)


def scale_zscore(x: dict, stats: dict = ZSCORE_STATS, target_std: float = TARGET_STD):
    """
    Zero-mean, fixed-std normalization:
        x' = (x - mean) * (target_std / std)
    Works per key; keys not in `stats` are returned unchanged.
    """
    out = {}
    for k, v in x.items():
        if v is None:
            out[k] = v
            continue
        s = stats.get(k, None)
        if s is None or ('mean' not in s) or ('std' not in s) or s['std'] is None:
            # no stats provided → passthrough
            out[k] = v
            continue
        mean = _get_scalar(s['mean'], v)
        std  = _get_scalar(s['std'],  v).clamp_min(_EPS)
        scale = _get_scalar(target_std, v) / std
        out[k] = (v - mean) * scale
    return out



def scale_minmax(x):
    """Min-max normalization to [-1, 1] range."""
    return {
        k: 2 * (v - NORMALIZATIONS[k][0]) / max(NORMALIZATIONS[k][1] - NORMALIZATIONS[k][0], _EPS) - 1
        if k in NORMALIZATIONS else v
        for k, v in x.items()
    }


def inverse_scale_zscore(x: dict, stats: dict = ZSCORE_STATS, target_std: float = TARGET_STD):
    """
    Inverse of scale_zscore:
        x = x_norm * (std / target_std) + mean
    Works per key; keys not in `stats` are returned unchanged.
    """
    out = {}
    for k, v in x.items():
        if v is None:
            out[k] = v
            continue
        s = stats.get(k, None)
        if s is None or ('mean' not in s) or ('std' not in s) or s['std'] is None:
            out[k] = v
            continue
        mean = _get_scalar(s['mean'], v)
        std = _get_scalar(s['std'], v).clamp_min(_EPS)
        tgt = _get_scalar(target_std, v).clamp_min(_EPS)
        out[k] = v * (std / tgt) + mean
    return out


def inverse_scale_minmax(x: dict):
    """
    Inverse of scale_minmax:
        x = ((x_norm + 1) / 2) * (max - min) + min
    Works per key; keys not in NORMALIZATIONS are returned unchanged.
    """
    out = {}
    for k, v in x.items():
        if v is None or k not in NORMALIZATIONS:
            out[k] = v
            continue
        min_v = _get_scalar(NORMALIZATIONS[k][0], v)
        max_v = _get_scalar(NORMALIZATIONS[k][1], v)
        out[k] = ((v + 1.0) / 2.0) * (max_v - min_v) + min_v
    return out



