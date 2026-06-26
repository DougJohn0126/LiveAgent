"""Positional encodings (Fourier and Learnable) for sequence models.

Adapted from DeepMind's Perceiver implementation with modifications for multimodal use cases.
Reference: https://github.com/deepmind/perceiver/blob/main/perceiver/position_encoding.py
"""

from abc import ABC, abstractmethod
from math import pi
from typing import List, Literal, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from torch.amp import autocast


def dataframe_indices_to_metric_seconds(
    dataframe_indices: torch.Tensor,
    unit_duration_seconds: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert dataframe indices to physical time in seconds.
    
    Args:
        dataframe_indices: Tensor of dataframe index values (typically integers or floats).
        unit_duration_seconds: Duration of each dataframe unit in seconds.
        dtype: Output dtype. If None, uses input dtype.
    
    Returns:
        Tensor of time values in seconds.
    """
    out_dtype = dataframe_indices.dtype if dtype is None else dtype
    return dataframe_indices.to(dtype=out_dtype) * float(unit_duration_seconds)


def build_linear_positions(
    index_dims: Sequence[int],
    output_range: Tuple[float, float] = (-1.0, 1.0),
    device: torch.device = None,
) -> torch.Tensor:
    """Generate position indices for an N-D input array.
    
    Args:
        index_dims: The spatial dimensions of the input array.
        output_range: Min and max values for normalized position coordinates.
        device: Torch device to place tensors on.
    
    Returns:
        Tensor of shape [*index_dims, len(index_dims)] containing normalized coordinates.
    """
    def _linspace(n_xels_per_dim):
        return torch.linspace(
            output_range[0], output_range[1],
            steps=n_xels_per_dim,
            dtype=torch.float32,
            device=device,
        )

    # Create coordinate grids for each dimension
    dim_ranges = [_linspace(n) for n in index_dims]
    grids = torch.meshgrid(*dim_ranges, indexing='ij')
    
    # Stack grids along the last dimension
    return torch.stack(grids, dim=-1)


def generate_fourier_features(
    pos: torch.Tensor,
    num_bands: int,
    min_resolution: Optional[Sequence[float]] = None,
    max_resolution: Optional[Sequence[float]] = None,
    concat_pos: bool = True,
    sine_only: bool = False,
) -> torch.Tensor:
    """Generate unit-based Fourier position encoding with linear spacing.
    
    Args:
        pos: Positions of shape [*spatial_dims, D] where D is number of coordinate dimensions.
        num_bands: Number of frequency bands (K) to use.
        min_resolution: Minimum frequency per coordinate unit for each dimension.
        max_resolution: Maximum frequency per coordinate unit for each dimension.
        concat_pos: Whether to concatenate raw positions to Fourier features.
        sine_only: Whether to use only sine (True) or both sin/cos (False).
    
    Returns:
        Encoded positions of shape [*spatial_dims, D + (2 or 1) * D * num_bands] if concat_pos,
        else [*spatial_dims, (2 or 1) * D * num_bands].
    """
    # Flatten all dimensions except the last (coordinate) dimension
    original_shape = pos.shape[:-1]
    d = pos.shape[-1]
    pos_flat = pos.reshape(-1, d)
    
    min_frequencies, max_frequencies = _resolve_frequency_bounds(
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        num_coord_dims=d,
    )
    freq_bands = _build_frequency_bands(
        min_frequencies=min_frequencies,
        max_frequencies=max_frequencies,
        num_bands=num_bands,
        device=pos.device,
        dtype=pos.dtype,
    )
    
    # Apply frequency bands to each coordinate dimension
    # pos_flat: [N, D], freq_bands: [D, num_bands]
    # Output: [N, D, num_bands]
    per_pos_features = pos_flat.unsqueeze(-1) * freq_bands.unsqueeze(0)  # [N, D, num_bands]
    
    # Flatten to [N, D * num_bands]
    per_pos_features = per_pos_features.reshape(per_pos_features.shape[0], -1)
    
    # Apply sin/cos using phase = 2π f x with frequency in real coordinate units.
    phase = 2.0 * math.pi * per_pos_features
    if sine_only:
        per_pos_features = torch.sin(phase)
    else:
        per_pos_features = torch.cat([
            torch.sin(phase),
            torch.cos(phase),
        ], dim=-1)
    
    # Concatenate raw positions if requested
    if concat_pos:
        per_pos_features = torch.cat([pos_flat, per_pos_features], dim=-1)
    
    # Reshape back to original spatial dimensions
    output_dim = per_pos_features.shape[-1]
    per_pos_features = per_pos_features.reshape(*original_shape, output_dim)
    
    return per_pos_features


def _resolve_frequency_bounds(
    min_resolution: Optional[Sequence[float]],
    max_resolution: Optional[Sequence[float]],
    num_coord_dims: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return one min/max frequency pair per coordinate dimension."""
    if min_resolution is None or max_resolution is None:
        raise ValueError(
            "Fourier frequency bounds must be explicitly configured: both min_resolution and max_resolution are required."
        )

    min_frequencies = tuple(float(v) for v in min_resolution)
    max_frequencies = tuple(float(v) for v in max_resolution)

    if len(min_frequencies) != len(max_frequencies):
        raise ValueError(
            f"min_resolution and max_resolution must have the same length, got {len(min_frequencies)} and {len(max_frequencies)}."
        )

    if len(max_frequencies) > num_coord_dims:
        min_frequencies = min_frequencies[:num_coord_dims]
        max_frequencies = max_frequencies[:num_coord_dims]
    elif len(max_frequencies) < num_coord_dims:
        pad = num_coord_dims - len(max_frequencies)
        min_frequencies = min_frequencies + tuple([min_frequencies[-1]] * pad)
        max_frequencies = max_frequencies + tuple([max_frequencies[-1]] * pad)

    return min_frequencies, max_frequencies


def _build_frequency_bands(
    min_frequencies: Sequence[float],
    max_frequencies: Sequence[float],
    num_bands: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build per-dimension Fourier bands in real coordinate units using log-linear spacing."""
    if len(min_frequencies) != len(max_frequencies):
        raise ValueError(
            f"min_frequencies and max_frequencies must have the same length, got {len(min_frequencies)} and {len(max_frequencies)}."
        )

    freq_bands = []
    for min_freq, max_freq in zip(min_frequencies, max_frequencies):
        min_f = float(min_freq)
        max_f = float(max_freq)
        if min_f <= 0.0 or max_f <= 0.0:
            raise ValueError(f"Frequency bounds must be positive, got min={min_f}, max={max_f}.")
        if min_f > max_f:
            raise ValueError(f"min frequency must be <= max frequency, got min={min_f}, max={max_f}.")

        if num_bands > 1:
            # Log-linear spacing for multi-scale positional awareness
            bands = torch.exp(
                torch.linspace(
                    math.log(min_f),
                    math.log(max_f),
                    steps=num_bands,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            bands = torch.tensor([max_f], dtype=dtype, device=device)
        freq_bands.append(bands)

    return torch.stack(freq_bands, dim=0)

def _apply_nyquist_mask(
    raw_features: torch.Tensor,
    fps: float,
    min_resolution: Sequence[float],
    max_resolution: Sequence[float],
    num_bands: int,
    concat_pos: bool,
    sine_only: bool,
) -> torch.Tensor:
    """Zero out temporal Fourier bands above the Nyquist limit while preserving spatial bands."""
    nyquist_limit = float(fps) / 2.0
    num_coord_dims = len(max_resolution)
    min_frequencies, max_frequencies = _resolve_frequency_bounds(
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        num_coord_dims=num_coord_dims,
    )
    temporal_freq_bands = _build_frequency_bands(
        min_frequencies=min_frequencies[:1],
        max_frequencies=max_frequencies[:1],
        num_bands=num_bands,
        device=raw_features.device,
        dtype=raw_features.dtype,
    ).reshape(-1)

    temporal_band_mask = (temporal_freq_bands <= nyquist_limit).to(raw_features.dtype)
    spatial_band_mask = torch.ones(
        max(num_coord_dims - 1, 0) * num_bands,
        device=raw_features.device,
        dtype=raw_features.dtype,
    )

    base_band_mask = torch.cat([temporal_band_mask, spatial_band_mask], dim=0)

    if sine_only:
        feature_mask = base_band_mask
    else:
        feature_mask = torch.cat([base_band_mask, base_band_mask], dim=0)

    if concat_pos:
        coord_mask = torch.ones(num_coord_dims, device=raw_features.device, dtype=raw_features.dtype)
        feature_mask = torch.cat([coord_mask, feature_mask], dim=0)

    if feature_mask.shape[-1] != raw_features.shape[-1]:
        raise RuntimeError(
            f"Mask dimension mismatch: mask has {feature_mask.shape[-1]} features, "
            f"but positional encoding has {raw_features.shape[-1]}."
        )

    return raw_features * feature_mask


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


@autocast('cuda', enabled=False)
def apply_rotary_emb(
    freqs,
    t,
    start_index=0,
    scale=1.,
    seq_dim=-2,
    freqs_seq_dim=None,
):
    dtype = t.dtype

    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        custom_freqs: torch.Tensor | None = None,
        freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        theta=10000,
        max_spatial_freq=1.0,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        interpolate_factor=1.,
        theta_rescale_factor=1.,
        seq_before_head_dim=False,
        cache_if_possible=True,
        cache_max_seq_len=8192,
        mrope_section: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for
        self.mrope_section = None if mrope_section is None else tuple(int(v) for v in mrope_section)

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            if self.mrope_section is None:
                raise ValueError(
                    "mrope_section must be explicitly provided for freqs_for='lang' to match VideoRoPE layout semantics."
                )

            if len(self.mrope_section) != 3:
                raise ValueError(f"mrope_section must have 3 values [t, x, y], got {self.mrope_section}")

            if any(v < 0 for v in self.mrope_section):
                raise ValueError(f"mrope_section entries must be >= 0, got {self.mrope_section}")

            half_dim = dim // 2
            if sum(self.mrope_section) != half_dim:
                raise ValueError(
                    f"sum(mrope_section) must equal dim//2 ({half_dim}) for dim={dim}, got {self.mrope_section}"
                )

            # Shared RoPE ladder across all axes, like Qwen/VideoRoPE-style MRoPE.
            # Axis-specific behavior is controlled by section allocation + VideoRoPE layout.
            freqs = theta ** -(torch.arange(half_dim, dtype=torch.float32) / half_dim)
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
            self.mrope_section = None
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
            self.mrope_section = None
        else:
            raise ValueError(f"Unknown freqs_for: {freqs_for}")

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent=False)
        self.cached_freqs_seq_len = 0

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.learned_freq = learned_freq

        self.register_buffer('dummy', torch.tensor(0), persistent=False)

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device=None, dtype=None, offset=0):
        device = default(device, self.device)
        dtype = default(dtype, self.cached_freqs.dtype)

        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, scale=default(scale, 1.), seq_dim=seq_dim)

    def _apply_videorope_axis_layout(self, axis_freqs: torch.Tensor) -> torch.Tensor:
        """
        VideoRoPE layout:
        1) use true mrope section sizes,
        2) merge spatial sections,
        3) duplicate and reverse to get [spatial, temporal, spatial, temporal],
        4) interleave spatial channels x/y channel-by-channel.
        """
        if axis_freqs.shape[0] != 3:
            raise ValueError(f"Expected 3 axes (t, x, y), got axis_freqs.shape[0]={axis_freqs.shape[0]}")

        rot_dim = axis_freqs.shape[-1]

        if self.mrope_section is None:
            raise ValueError("mrope_section is required for VideoRoPE axis layout.")

        temporal, spatial_x, spatial_y = (int(v) for v in self.mrope_section)
        merged = [temporal, spatial_x + spatial_y]
        # Match reference implementation: [t, x+y] -> [t, x+y, t, x+y] -> reversed.
        merged_sizes = (merged * 2)[::-1]  # [spatial, temporal, spatial, temporal]

        pieces = []
        index = 0
        for section_idx, section_len in enumerate(merged_sizes):
            if section_len <= 0:
                continue
            if section_idx % 2 == 0:
                # Spatial: interleave x/y channel-by-channel.
                for channel_offset in range(section_len):
                    axis = 1 if (channel_offset % 2 == 0) else 2
                    pieces.append(axis_freqs[axis, ..., index:index + 1])
                    index += 1
            else:
                # Temporal: contiguous temporal block at the end.
                pieces.append(axis_freqs[0, ..., index:index + section_len])
                index += section_len

        if index != rot_dim:
            raise AssertionError(
                f"VideoRoPE section layout consumed {index} channels, expected exactly {rot_dim}. "
                f"Check mrope_section={self.mrope_section}."
            )

        return torch.cat(pieces, dim=-1)

    def apply_multimodal_rotary_pos_emb(
        self,
        t,
        metric_pos,
        seq_dim=None,
    ):
        """
        Apply multimodal RoPE to query/key tensors. (https://qwenlm.github.io/blog/qwen2-vl/)

        Multimodal 3D RoPE extends 1D RoPE by assigning separate temporal, height,
        and width coordinates to vision tokens. For text-like / 1D tokens, the three
        coordinates are identical so behavior matches standard 1D RoPE.

        Args:
            t: Query or Key tensor [B, H, N, D]
            metric_pos: Explicit per-token VideoRoPE coordinates [B, N, 3]
            seq_dim: Sequence dimension (-2 for [B, H, N, D])

        Returns:
            Rotated tensor with same shape as t
        """
        seq_dim = default(seq_dim, self.default_seq_dim)

        if metric_pos.ndim != 3 or metric_pos.shape[-1] != 3:
            raise ValueError(
                f"metric_pos must have shape [B, N, 3], got {tuple(metric_pos.shape)}"
            )

        coords = metric_pos.to(self.freqs.dtype).clone()

        axis_freqs = torch.einsum('b n c, f -> c b n f', coords, self.freqs)
        axis_freqs = repeat(axis_freqs, 'c b n f -> c b n (f r)', r=2)
        freqs = self._apply_videorope_axis_layout(axis_freqs)

        if seq_dim == -2:
            freqs = rearrange(freqs, 'b n d -> b 1 n d')
        elif seq_dim == -3:
            freqs = rearrange(freqs, 'b n d -> n 1 d')

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    @autocast('cuda', enabled=False)
    def forward(self, t: torch.Tensor, seq_len: int | None = None, offset=0):
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and
            exists(self.cached_freqs) and
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs


class AbstractPositionEncoding(nn.Module, ABC):
    """Base class for position encodings."""
    
    @abstractmethod
    def forward(
        self,
        batch_size: int,
        pos: Optional[torch.Tensor] = None,
        nyquist_fps: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate position encodings.
        
        Args:
            batch_size: Batch size to broadcast encodings to.
            pos: Optional pre-computed positions to encode.
        
        Returns:
            Encoded positions of shape [batch_size, *spatial_dims, encoding_dim].
        """
        raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
    """Learnable position encoding via a lookup table."""
    
    def __init__(self, index_dim: int, num_channels: int = 128, init_scale: float = 0.02):
        super().__init__()
        self.index_dim = index_dim
        self.num_channels = num_channels
        self.pos_embs = nn.Parameter(
            torch.randn(index_dim, num_channels) * init_scale
        )
    
    def forward(
        self,
        batch_size: int,
        pos: Optional[torch.Tensor] = None,
        nyquist_fps: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size to broadcast to.
            pos: Unused; kept for interface compatibility.
        
        Returns:
            Learnable embeddings of shape [batch_size, index_dim, num_channels].
        """
        del nyquist_fps
        if pos is None:
            embs = self.pos_embs.unsqueeze(0)  # [1, index_dim, num_channels]
            if batch_size is not None:
                embs = embs.expand(batch_size, -1, -1)
            return embs

        # Dynamic shape path: map each position to a learned embedding index.
        # This preserves the leading shape of `pos` while providing trainable encodings.
        leading_shape = pos.shape[:-1]
        flat_count = int(np.prod(leading_shape))
        idx = torch.arange(flat_count, device=pos.device, dtype=torch.long) % self.index_dim
        out = self.pos_embs[idx].reshape(*leading_shape, self.num_channels)
        return out


class FourierPositionEncoding(AbstractPositionEncoding):
    """Fourier (sinusoidal) position encoding."""
    
    def __init__(
        self,
        index_dims: Sequence[int],
        num_bands: int,
        concat_pos: bool = True,
        min_resolution: Optional[Sequence[float]] = None,
        max_resolution: Optional[Sequence[int]] = None,
        sine_only: bool = False,
        apply_nyquist_mask: bool = False,
        nyquist_fps: Optional[float] = None,
    ):
        super().__init__()
        self.index_dims = tuple(index_dims)
        self.num_bands = num_bands
        self.concat_pos = concat_pos
        self.sine_only = sine_only
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution or self.index_dims
        self.apply_nyquist_mask = apply_nyquist_mask
        self.nyquist_fps = nyquist_fps
        self.register_buffer("device_ref", torch.tensor(0), persistent=False)
        
        # Pre-compute output dimension for convenience
        d = len(self.index_dims)
        pos_dim = d if concat_pos else 0
        freq_dim = (2 if not sine_only else 1) * d * num_bands
        self.output_dim = pos_dim + freq_dim
    
    def forward(
        self,
        batch_size: int,
        pos: Optional[torch.Tensor] = None,
        nyquist_fps: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size to broadcast encodings to.
            pos: Optional pre-computed positions of shape [batch_size, *index_dims, D].
                 If None, linear positions are generated.
        
        Returns:
            Fourier encoded positions of shape [batch_size, *index_dims, output_dim].
        """
        return self.encode_raw(batch_size=batch_size, pos=pos, nyquist_fps=nyquist_fps)

    def encode_raw(
        self,
        batch_size: int,
        pos: Optional[torch.Tensor] = None,
        nyquist_fps: Optional[float] = None,
    ) -> torch.Tensor:
        """Return the unprojected Fourier features for direct inspection or visualization."""
        if pos is None:
            device = self.device_ref.device
            pos = build_linear_positions(
                self.index_dims,
                device=device,
            )
            pos = pos.unsqueeze(0).expand(batch_size, *([1] * len(self.index_dims)), -1)

        features = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        )

        if self.apply_nyquist_mask:
            effective_nyquist_fps = self.nyquist_fps if nyquist_fps is None else nyquist_fps
            if effective_nyquist_fps is None:
                raise ValueError("nyquist_fps must be set when apply_nyquist_mask=True")
            features = _apply_nyquist_mask(
                raw_features=features,
                fps=float(effective_nyquist_fps),
                min_resolution=self.min_resolution,
                max_resolution=self.max_resolution,
                num_bands=self.num_bands,
                concat_pos=self.concat_pos,
                sine_only=self.sine_only,
            )

        return features


class PositionEncodingProjector(AbstractPositionEncoding):
    """Projects a position encoding to a target dimension."""
    
    def __init__(self, output_size: int, base_encoding: AbstractPositionEncoding):
        super().__init__()
        self.output_size = output_size
        self.base_encoding = base_encoding
        
        # Determine input dimension from base encoding output_dim if available
        if hasattr(base_encoding, 'output_dim'):
            input_dim = base_encoding.output_dim
        else:
            # Default fallback
            input_dim = 128
        
        self.projection = nn.Linear(input_dim, output_size, bias=False)
    
    def forward(
        self,
        batch_size: int,
        pos: Optional[torch.Tensor] = None,
        nyquist_fps: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size.
            pos: Optional positions to pass to base encoding.
        
        Returns:
            Projected encodings of shape [batch_size, ..., output_size].
        """
        base_encoding = self.base_encoding(batch_size, pos, nyquist_fps=nyquist_fps)
        
        # Reshape for linear projection: flatten all dims except output_dim
        original_shape = base_encoding.shape
        flat = base_encoding.reshape(-1, original_shape[-1])

        # Match projection parameter dtype/device (e.g. bf16 under mixed precision training)
        flat = flat.to(device=self.projection.weight.device, dtype=self.projection.weight.dtype)
        projected = self.projection(flat)
        
        return projected.reshape(*original_shape[:-1], self.output_size)


def build_position_encoding(
    encoding_type: str,
    index_dims: Sequence[int],
    output_dim: int,
    fourier_num_bands: Optional[int] = None,
    fourier_concat_pos: bool = True,
    fourier_sine_only: bool = False,
    fourier_min_resolution: Optional[Sequence[float]] = None,
    fourier_max_resolution: Optional[Sequence[int]] = None,
    fourier_apply_nyquist_mask: bool = False,
    fourier_nyquist_fps: Optional[float] = None,
    trainable_pos_init_scale: float = 0.02,
) -> AbstractPositionEncoding:
    """Builder function for position encodings.
    
    Args:
        encoding_type: 'fourier' or 'trainable'.
        index_dims: Spatial dimensions.
        output_dim: Target output dimension.
        fourier_num_bands: Number of frequency bands for Fourier encoding.
        fourier_concat_pos: Whether to concatenate raw positions in Fourier.
        fourier_sine_only: Whether to use only sine in Fourier.
        fourier_max_resolution: Max resolution for Fourier frequencies.
        trainable_pos_init_scale: Initialization scale for trainable encoding.
    
    Returns:
        A position encoding module.
    """
    if encoding_type == 'fourier':
        if fourier_num_bands is None:
            fourier_num_bands = 64

        base_encoding = FourierPositionEncoding(
            index_dims=index_dims,
            num_bands=fourier_num_bands,
            concat_pos=fourier_concat_pos,
            min_resolution=fourier_min_resolution,
            max_resolution=fourier_max_resolution,
            sine_only=fourier_sine_only,
            apply_nyquist_mask=fourier_apply_nyquist_mask,
            nyquist_fps=fourier_nyquist_fps,
        )
    elif encoding_type == 'trainable':
        index_dim = int(np.prod(index_dims))
        base_encoding = TrainablePositionEncoding(
            index_dim=index_dim,
            num_channels=output_dim,
            init_scale=trainable_pos_init_scale,
        )
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    # Project to output_dim if needed
    if encoding_type == 'fourier' and base_encoding.output_dim != output_dim:
        return PositionEncodingProjector(output_dim, base_encoding)
    
    return base_encoding
