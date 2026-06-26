"""Attention mask utilities backed by PyTorch BlockMask for flex attention."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


MASK_REGISTRY: Dict[str, Callable] = {}
DEFAULT_MASK_TYPE = "dataframe_level"


def register_mask(name: str):
    """Register a mask factory function under a human-readable name."""

    def decorator(fn: Callable) -> Callable:
        """Store the function in the global registry and return it unchanged."""
        MASK_REGISTRY[name] = fn
        return fn

    return decorator


def get_available_masks() -> List[str]:
    """Return all currently registered mask type names."""
    return list(MASK_REGISTRY.keys())


def get_default_mask_type() -> str:
    """Return the default mask type used when none is specified."""
    return DEFAULT_MASK_TYPE


def create_mask(mask_type: str, **kwargs) -> Optional[BlockMask]:
    """Create a mask instance from the registry using the given type and args."""
    if mask_type not in MASK_REGISTRY:
        available = ", ".join(get_available_masks())
        raise ValueError(f"Unknown mask type '{mask_type}'. Available: {available}")
    return MASK_REGISTRY[mask_type](**kwargs)


@register_mask("token_level")
def create_token_level_mask(
    timestamps: torch.Tensor,
    B: Optional[int] = None,
    Q_LEN: Optional[int] = None,
    KV_LEN: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> BlockMask:
    """Build a token-level causal mask where each query attends only to earlier timestamps."""

    if timestamps.ndim == 1:
        timestamps = timestamps.unsqueeze(0)

    if timestamps.ndim != 2:
        raise ValueError(f"timestamps must be rank-1 or rank-2, got shape {tuple(timestamps.shape)}")

    inferred_b, inferred_len = int(timestamps.shape[0]), int(timestamps.shape[1])
    B = inferred_b if B is None else int(B)
    Q_LEN = inferred_len if Q_LEN is None else int(Q_LEN)
    KV_LEN = inferred_len if KV_LEN is None else int(KV_LEN)
    device = timestamps.device if device is None else device

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        """Allow attention only when the key/value timestamp is not in the future."""
        del h
        q_idx = torch.clamp(q_idx, max=Q_LEN - 1)
        kv_idx = torch.clamp(kv_idx, max=KV_LEN - 1)
        return timestamps[b, kv_idx] <= timestamps[b, q_idx]

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device)


@register_mask("dataframe_level")
def create_dataframe_level_mask(
    modality_layout: List[Tuple[str, int]],
    num_timesteps: int,
    B: Optional[int] = None,
    Q_LEN: Optional[int] = None,
    KV_LEN: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> BlockMask:
    """Build a dataframe-level causal mask for timestep-interleaved multimodal sequences."""

    if not modality_layout:
        raise ValueError("modality_layout must be non-empty")

    tokens_per_timestep = sum(int(tokens_per_step) for _, tokens_per_step in modality_layout)
    if tokens_per_timestep <= 0:
        raise ValueError("modality_layout must contain at least one token per timestep")

    inferred_len = int(num_timesteps) * int(tokens_per_timestep)
    B = 1 if B is None else int(B)
    Q_LEN = inferred_len if Q_LEN is None else int(Q_LEN)
    KV_LEN = inferred_len if KV_LEN is None else int(KV_LEN)
    device = torch.device("cpu") if device is None else device

    def build_token_maps(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Map each token index to its timestep and modality id for interleaved [t][modality] layout."""
        token_to_timestep = torch.empty(seq_len, dtype=torch.long, device=device)
        token_to_modality = torch.empty(seq_len, dtype=torch.long, device=device)

        start = 0
        last_modality = len(modality_layout) - 1
        for timestep in range(int(num_timesteps)):
            for mod_idx, (_, tokens_per_step) in enumerate(modality_layout):
                span = int(tokens_per_step)
                if span <= 0:
                    continue
                end = min(start + span, seq_len)
                if end <= start:
                    break
                token_to_timestep[start:end] = timestep
                token_to_modality[start:end] = mod_idx
                start = end
            if start >= seq_len:
                break

        if start < seq_len:
            token_to_timestep[start:] = num_timesteps - 1
            token_to_modality[start:] = last_modality
        return token_to_timestep, token_to_modality

    q_to_timestep, q_to_modality = build_token_maps(Q_LEN)
    kv_to_timestep, kv_to_modality = build_token_maps(KV_LEN)

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        """Allow past attention and same-timestep attention across all modalities."""
        del b, h
        q_idx = torch.clamp(q_idx, max=Q_LEN - 1)
        kv_idx = torch.clamp(kv_idx, max=KV_LEN - 1)

        q_t = q_to_timestep[q_idx]
        kv_t = kv_to_timestep[kv_idx]
        q_m = q_to_modality[q_idx]
        kv_m = kv_to_modality[kv_idx]

        same_timestep = kv_t == q_t
        past_timestep = kv_t < q_t
        del q_m, kv_m
        return past_timestep | same_timestep

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device)


@register_mask("no_mask")
def create_fully_bidirectional_mask(
    B: Optional[int] = None,
    Q_LEN: Optional[int] = None,
    KV_LEN: Optional[int] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> BlockMask | None:
    """Disable masking so attention is fully bidirectional."""
    del B, Q_LEN, KV_LEN, device, kwargs
    return None
