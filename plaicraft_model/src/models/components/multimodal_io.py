from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask

from data.data_classes import FullData
from models.components.attention_masks import (
    create_token_level_mask,
    create_dataframe_level_mask,
)
from models.components.positional_encoding import (
    build_position_encoding,
    AbstractPositionEncoding,
)
from utils.constants import (
    AUDIO_TOKEN_FPS,
    KEYBOARD_FPS,
    MODALITY_SHAPES,
    MOUSE_FPS,
    UNIT_DURATION_SECONDS,
    VALID_MODALITIES,
    VIDEO_FPS,
)

# Enforce deterministic modality ordering for consistent encode/decode.
# Flattened target/STM sequences use timestep-interleaved order.
MODALITY_ORDER = (
    "audio_speak",
    "key_press",
    "mouse_movement",
    "audio_hear",
    "video",
)


class MultimodalIO(nn.Module):
    """
    Unified multimodal I/O adapter for context and target FullData.

    This module handles all packing, unpacking, patchification, and embedding logic.
    It takes raw datasets (video frames, audio waveforms, keypresses) and projects 
    them directly into a unified `model_dim` space with additive Fourier positional 
    and learned modality embeddings.
    """

    def __init__(
        self,
        patch_h: int,
        patch_w: int,
        model_dim: int,
        player_embed_dim: int,
        ltm_patch_h: Optional[int] = None,
        ltm_patch_w: Optional[int] = None,
        stm_context_length: int = 0,
        ltm_downsample_chunk_length: int = 10,
        ltm_drop_modalities: Optional[List[str]] = None,
        context_modalities: Optional[List[str]] = None,
        target_modalities: Optional[List[str]] = None,
        name_vocab_size: int = 4096,
        gender_vocab_size: int = 32,
        skill_vocab_size: int = 64,
        mask_type: str = "token_level",
        positional_encoding_type: str = "fourier",
        fourier_num_bands: int = 64,
        fourier_concat_pos: bool = True,
        fourier_sine_only: bool = False,
        temporal_fourier_min_resolution: Optional[float] = None,
        temporal_fourier_max_resolution: Optional[float] = None,
        video_spatial_fourier_min_resolution: Optional[Sequence[float]] = None,
        video_spatial_fourier_max_resolution: Optional[Sequence[float]] = None,
        trainable_pos_init_scale: float = 0.02,
        rope_temporal_scale_factor: float = 100.0,
        use_diagonal_layout_coord: bool = False,
    ) -> None:
        super().__init__()

        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"patch_h and patch_w must be > 0, got ({patch_h}, {patch_w})")

        video_shape = MODALITY_SHAPES["video"]
        video_channels = video_shape[1]
        video_height = video_shape[2]
        video_width = video_shape[3]

        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.ltm_patch_h = int(patch_h if ltm_patch_h is None else ltm_patch_h)
        self.ltm_patch_w = int(patch_w if ltm_patch_w is None else ltm_patch_w)
        
        self.model_dim = int(model_dim)
        self.player_embed_dim = int(player_embed_dim)
        self.stm_context_length = int(stm_context_length)
        self.ltm_downsample_chunk_length = int(ltm_downsample_chunk_length)
        self.ltm_drop_modalities = set(ltm_drop_modalities or [])
        self.context_modalities = set(context_modalities or VALID_MODALITIES)
        self.target_modalities = set(target_modalities or VALID_MODALITIES)
        self.mask_type = mask_type
        
        # Positional encoding configuration
        self.positional_encoding_type = str(positional_encoding_type)
        if self.positional_encoding_type not in {"fourier", "trainable", "rope"}:
            raise ValueError(
                f"Unsupported positional_encoding_type='{self.positional_encoding_type}'. "
                "Expected one of: 'fourier', 'trainable', 'rope'."
            )
        self.use_rope = self.positional_encoding_type == "rope"
        additive_positional_encoding_type = "fourier" if self.use_rope else self.positional_encoding_type
        self.fourier_num_bands = int(fourier_num_bands)
        self.fourier_concat_pos = bool(fourier_concat_pos)
        self.fourier_sine_only = bool(fourier_sine_only)
        self.temporal_fourier_min_resolution = (
            float(temporal_fourier_min_resolution)
            if temporal_fourier_min_resolution is not None
            else None
        )
        self.temporal_fourier_max_resolution = (
            float(temporal_fourier_max_resolution)
            if temporal_fourier_max_resolution is not None
            else None
        )
        self.video_spatial_fourier_min_resolution = (
            tuple(float(value) for value in video_spatial_fourier_min_resolution)
            if video_spatial_fourier_min_resolution is not None
            else None
        )
        self.video_spatial_fourier_max_resolution = (
            tuple(float(value) for value in video_spatial_fourier_max_resolution)
            if video_spatial_fourier_max_resolution is not None
            else None
        )
        self.rope_temporal_scale_factor = float(rope_temporal_scale_factor)
        self.use_diagonal_layout_coord = bool(use_diagonal_layout_coord)

        invalid_ltm_drop = self.ltm_drop_modalities - VALID_MODALITIES
        invalid_context = self.context_modalities - VALID_MODALITIES
        invalid_target = self.target_modalities - VALID_MODALITIES
        if invalid_ltm_drop:
            raise ValueError(
                f"Invalid ltm_drop_modalities: {sorted(invalid_ltm_drop)}. "
                f"Allowed: {sorted(VALID_MODALITIES)}"
            )
        if invalid_context:
            raise ValueError(
                f"Invalid context_modalities: {sorted(invalid_context)}. "
                f"Allowed: {sorted(VALID_MODALITIES)}"
            )
        if invalid_target:
            raise ValueError(
                f"Invalid target_modalities: {sorted(invalid_target)}. "
                f"Allowed: {sorted(VALID_MODALITIES)}"
            )
        if not self.context_modalities:
            raise ValueError("context_modalities must include at least one modality")
        if not self.target_modalities:
            raise ValueError("target_modalities must include at least one modality")

        if video_height % self.ltm_patch_h != 0 or video_width % self.ltm_patch_w != 0:
            raise ValueError(
                f"Invalid LTM patch size ({self.ltm_patch_h}, {self.ltm_patch_w}) for video latent shape "
                f"(C={video_channels}, H={video_height}, W={video_width})."
            )

        # ---------------------------------------------------------
        # Input Projectors (Raw Feature Space -> Unified model_dim)
        # ---------------------------------------------------------
        stm_video_patch_dim = int(video_channels * self.patch_h * self.patch_w)
        ltm_video_patch_dim = int(video_channels * self.ltm_patch_h * self.ltm_patch_w)
        
        self.stm_video_proj = nn.Linear(stm_video_patch_dim, self.model_dim)
        self.ltm_video_proj = nn.Linear(ltm_video_patch_dim, self.model_dim)
        self.audio_proj = nn.Linear(MODALITY_SHAPES["audio_hear"][1], self.model_dim)
        self.key_proj = nn.Linear(MODALITY_SHAPES["key_press"][1], self.model_dim)
        self.mouse_proj = nn.Linear(MODALITY_SHAPES["mouse_movement"][1], self.model_dim)
        
        # Positional encoding modules
        # For video: 3D coordinates (time, x, y)
        self.video_pos_encoding = build_position_encoding(
            encoding_type=additive_positional_encoding_type,
            index_dims=(1, 1, 1),  # Will be dynamic at runtime
            output_dim=self.model_dim,
            fourier_num_bands=fourier_num_bands,
            fourier_concat_pos=fourier_concat_pos,
            fourier_sine_only=fourier_sine_only,
            fourier_min_resolution=(
                [self.temporal_fourier_min_resolution, *self.video_spatial_fourier_min_resolution]
                if self.temporal_fourier_min_resolution is not None and self.video_spatial_fourier_min_resolution is not None
                else None
            ),
            fourier_max_resolution=(
                [self.temporal_fourier_max_resolution, *self.video_spatial_fourier_max_resolution]
                if self.temporal_fourier_max_resolution is not None and self.video_spatial_fourier_max_resolution is not None
                else None
            ),
            fourier_apply_nyquist_mask=True,
            fourier_nyquist_fps=VIDEO_FPS,
            trainable_pos_init_scale=trainable_pos_init_scale,
        )

        self.seq_pos_encoding = build_position_encoding(
            encoding_type=additive_positional_encoding_type,
            index_dims=(1,),
            output_dim=self.model_dim,
            fourier_num_bands=fourier_num_bands,
            fourier_concat_pos=fourier_concat_pos,
            fourier_sine_only=fourier_sine_only,
            fourier_min_resolution=(
                [self.temporal_fourier_min_resolution]
                if self.temporal_fourier_min_resolution is not None
                else None
            ),
            fourier_max_resolution=(
                [self.temporal_fourier_max_resolution]
                if self.temporal_fourier_max_resolution is not None
                else None
            ),
            fourier_apply_nyquist_mask=True,
            fourier_nyquist_fps=None,
            trainable_pos_init_scale=trainable_pos_init_scale,
        )
        
        self.modality_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(self.model_dim) * 0.02)
                for name in VALID_MODALITIES
            }
        )

        # ---------------------------------------------------------
        # Output Projectors (Unified model_dim -> Raw Feature Space)
        # ---------------------------------------------------------
        all_output_projectors = {
            "video": nn.Linear(self.model_dim, stm_video_patch_dim),
            "audio_hear": nn.Linear(self.model_dim, MODALITY_SHAPES["audio_hear"][1]),
            "audio_speak": nn.Linear(self.model_dim, MODALITY_SHAPES["audio_speak"][1]),
            "key_press": nn.Linear(self.model_dim, MODALITY_SHAPES["key_press"][1]),
            "mouse_movement": nn.Linear(self.model_dim, MODALITY_SHAPES["mouse_movement"][1]),
        }
        
        # Only keep projectors for target modalities
        self.output_projectors = nn.ModuleDict({
            name: proj for name, proj in all_output_projectors.items()
            if name in self.target_modalities
        })
        
        # Player metadata embeddings
        self.name_embedding = nn.Embedding(name_vocab_size, self.player_embed_dim)
        self.gender_embedding = nn.Embedding(gender_vocab_size, self.player_embed_dim)
        self.skill_embedding = nn.Embedding(skill_vocab_size, self.player_embed_dim)
        self.player_norm = nn.LayerNorm(self.player_embed_dim)
        self._mask_cache = {}

    def _apply_linear(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        """Applies a linear projection, safely coercing dtype and device."""
        return layer(x.to(device=layer.weight.device, dtype=layer.weight.dtype))

    def _get_relative_indices(self, full_data: FullData) -> torch.Tensor:
        """Extracts and formats the sequence dataframe indices for time computation."""
        relative_indices = getattr(full_data, "dataframe_indices", None)
        if relative_indices is None:
            raise ValueError("full_data.dataframe_indices is required for MultimodalIO processing")
        return relative_indices.to(dtype=torch.float32) * float(UNIT_DURATION_SECONDS)

    def _prepare_modalities(
        self,
        full_data: FullData,
        drop_modalities: Optional[set[str]] = None,
        include_modalities: Optional[set[str]] = None,
        local_time_period: Optional[int] = None,
        ltm_mode: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extracts modalities from FullData and prepares them for the model.
        Handles time chunking, modality dropping, and routing to the correct projection handlers.
        """
        resolved_relative_indices = self._get_relative_indices(full_data)
        
        # If processing LTM, wrap the timestamps so they reset every N chunks (e.g. -k...-1)
        if local_time_period is not None and local_time_period > 0:
            b, t = resolved_relative_indices.shape
            k = int(local_time_period)
            
            # Emit chunk-local negative indices for every retained frame.
            local_negative_indices = (torch.arange(t, device=resolved_relative_indices.device, dtype=torch.long) % k) - k
            resolved_relative_indices = local_negative_indices.view(1, t).expand(b, -1).to(torch.float32)

        modalities: Dict[str, Dict[str, torch.Tensor]] = {}
        drop_modalities = drop_modalities or set()
        include_modalities = include_modalities or VALID_MODALITIES

        patch_h = self.ltm_patch_h if ltm_mode else self.patch_h
        patch_w = self.ltm_patch_w if ltm_mode else self.patch_w
        video_proj = self.ltm_video_proj if ltm_mode else self.stm_video_proj

        # 1. Video Sequence
        if (
            full_data.video is not None
            and "video" in include_modalities
            and "video" not in drop_modalities
        ):
            modalities["video"] = self._prepare_video_sequence(
                x=full_data.video,
                relative_indices=resolved_relative_indices,
                modality="video",
                patch_h=patch_h,
                patch_w=patch_w,
                projection_layer=video_proj,
                ltm_mode=ltm_mode,
            )
            
        # 2. Unified 1D Sequences (Audio, Key, Mouse)
        configs = [
            ("audio_hear", full_data.audio_hear, AUDIO_TOKEN_FPS, self.audio_proj),
            ("audio_speak", full_data.audio_speak, AUDIO_TOKEN_FPS, self.audio_proj),
            ("key_press", full_data.key_press, KEYBOARD_FPS, self.key_proj),
            ("mouse_movement", full_data.mouse_movement, MOUSE_FPS, self.mouse_proj),
        ]
        
        for name, data, fps, proj in configs:
            if data is not None and name in include_modalities and name not in drop_modalities:
                modalities[name] = self._prepare_1d_sequence(
                    x=data,
                    relative_indices=resolved_relative_indices,
                    modality=name,
                    fps=fps,
                    projection_layer=proj,
                    ltm_mode=ltm_mode,
                )

        if not modalities:
            raise ValueError("No modalities found in FullData for multimodal I/O processing")

        return modalities

    def _prepare_video_sequence(
        self,
        x: torch.Tensor,
        relative_indices: torch.Tensor,
        modality: str,
        patch_h: int,
        patch_w: int,
        projection_layer: nn.Linear,
        ltm_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Splits video into spatial patches and processes them into model_dim tokens."""
        b, t, f, c, h, w = x.shape
        gh, gw = h // patch_h, w // patch_w
        patch_dim = c * patch_h * patch_w

        # Patchify: [B, T, F, C, H, W] -> [B, T, F*gh*gw, patch_dim]
        patches = x.reshape(b, t, f, c, gh, patch_h, gw, patch_w)
        patches = patches.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(b, t, f * gh * gw, patch_dim)
        projected = self._apply_linear(projection_layer, patches)
        
        # Calculate positional coordinates for each patch
        # Time: offset within frame, relative to dataframe start
        frame_offsets = (torch.arange(f, device=x.device, dtype=torch.float32) / float(VIDEO_FPS)).view(1, 1, f, 1)
        frame_offsets = frame_offsets.expand(b, t, f, gh * gw).reshape(b, t, f * gh * gw)

        # Spatial coordinates: preserve exact integer patch indices for stable spatial RoPE geometry.
        patch_x = torch.arange(gh, device=x.device, dtype=torch.long)
        patch_y = torch.arange(gw, device=x.device, dtype=torch.long)
        px, py = torch.meshgrid(patch_x, patch_y, indexing="ij")
        px = px.reshape(1, 1, 1, gh * gw).expand(b, t, f, gh * gw).reshape(b, t, f * gh * gw)
        py = py.reshape(1, 1, 1, gh * gw).expand(b, t, f, gh * gw).reshape(b, t, f * gh * gw)

        modality_enc = self.modality_embeddings[modality].to(device=projected.device, dtype=projected.dtype)
        modality_enc = modality_enc.view(1, 1, 1, -1).expand_as(projected)

        # Stack coordinates: [B, T, F*gh*gw, 3] with either:
        # - default layout: [tau, x, y]
        # - diagonal layout: [tau, tau + (x - cx), tau + (y - cy)]
        rope_time = (relative_indices.unsqueeze(-1).expand(b, t, f * gh * gw) + frame_offsets) * self.rope_temporal_scale_factor
        if self.use_diagonal_layout_coord:
            center_x = (float(gh) - 1.0) / 2.0
            center_y = (float(gw) - 1.0) / 2.0
            rope_x = rope_time + (px.to(dtype=rope_time.dtype) - center_x)
            rope_y = rope_time + (py.to(dtype=rope_time.dtype) - center_y)
        else:
            rope_x = px.to(dtype=rope_time.dtype)
            rope_y = py.to(dtype=rope_time.dtype)
        rope_pos = torch.stack([
            rope_time,
            rope_x,
            rope_y,
        ], dim=-1)

        # Keep current additive Fourier positional encoding for LTM.
        if (not self.use_rope) or ltm_mode:
            pos_encodings = self._generate_pos_encoding_for_video(rope_pos, b, t, f * gh * gw)
            projected = projected + pos_encodings.to(device=projected.device, dtype=projected.dtype)

        projected = projected + modality_enc

        # Preserve raw physical time for masking and downstream alignment.
        time_values = relative_indices.unsqueeze(-1).expand(b, t, f * gh * gw) + frame_offsets

        return {"tokens": projected, "time": time_values, "rope_pos": rope_pos}

    def _prepare_1d_sequence(
        self,
        x: torch.Tensor,
        relative_indices: torch.Tensor,
        modality: str,
        fps: float,
        projection_layer: nn.Linear,
        ltm_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generic processor for 1D sequences (audio, keyboard, mouse)."""
        b, t, l, _ = x.shape
        
        # Calculate time offsets
        offsets = (torch.arange(l, device=x.device, dtype=torch.float32) / float(fps)).view(1, 1, l)
        time_values = relative_indices.unsqueeze(-1).expand(b, t, l) + offsets.expand(b, t, l)
        
        # Build coordinates (just time for 1D sequences)
        # Reshape to [B*T*L, 1] for positional encoding
        coords = time_values.reshape(b * t * l, 1)
        
        # Project raw features first, then inject positional and modality embeddings additively
        projected = self._apply_linear(projection_layer, x)
        modality_enc = self.modality_embeddings[modality].to(device=projected.device, dtype=projected.dtype)
        modality_enc = modality_enc.view(1, 1, 1, -1).expand_as(projected)

        if (not self.use_rope) or ltm_mode:
            pos_encodings = self._generate_pos_encoding_for_sequence(coords, b, t, l, fps, self.seq_pos_encoding)
            projected = projected + pos_encodings.to(device=projected.device, dtype=projected.dtype)

        projected = projected + modality_enc

        # Text-like / 1D modalities: expand the temporal coordinate across all three axes.
        rope_time = time_values * self.rope_temporal_scale_factor
        rope_pos = torch.stack([rope_time, rope_time, rope_time], dim=-1)
        
        return {"tokens": projected, "time": time_values, "rope_pos": rope_pos}

    def _augment_sequence_features(
        self,
        projected_tokens: torch.Tensor,
        time_values: torch.Tensor,
        modality_name: str,
        pos_x: Optional[torch.Tensor] = None,
        pos_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deprecated: additive embedding injection now happens in the input preparation path."""
        raise NotImplementedError(
            "Additive augmentation is handled directly in the input preparation path."
        )

    def _generate_pos_encoding_for_video(
        self,
        coords: torch.Tensor,
        batch_size: int,
        time_steps: int,
        num_patches: int,
    ) -> torch.Tensor:
        """Generate positional encodings for video."""
        del time_steps, num_patches
        pos_encodings = self.video_pos_encoding(batch_size=batch_size, pos=coords)
        return pos_encodings.to(device=coords.device, dtype=coords.dtype)
    
    def _generate_pos_encoding_for_sequence(
        self,
        coords: torch.Tensor,
        batch_size: int,
        time_steps: int,
        seq_len: int,
        fps: float,
        pos_encoding: AbstractPositionEncoding,
    ) -> torch.Tensor:
        """Generate positional encodings for 1D sequences (time only)."""
        coords_reshaped = coords.reshape(batch_size, time_steps, seq_len, 1)
        pos_encodings = pos_encoding(batch_size=batch_size, pos=coords_reshaped, nyquist_fps=fps)
        return pos_encodings.to(device=coords.device, dtype=coords.dtype)

    def _flatten_sequence(
        self,
        modalities: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int], Dict[str, Tuple[int, int]]]:
        """Flattens modality tensors into decoder order: timestep-major with per-timestep modality interleaving.
        
        Enforces deterministic modality ordering based on MODALITY_ORDER to guarantee
        consistent encoding/decoding across all forward passes.
        """
        active_modality_names: List[str] = []
        modality_lengths: List[int] = []
        modality_shapes: Dict[str, Tuple[int, int]] = {}
        interleaved_tokens: List[torch.Tensor] = []
        interleaved_times: List[torch.Tensor] = []
        interleaved_rope_pos: List[torch.Tensor] = []

        # Sort modalities according to MODALITY_ORDER for deterministic output
        sorted_names = [name for name in MODALITY_ORDER if name in modalities]
        if not sorted_names:
            raise ValueError("No modalities available for flattening")

        first_name = sorted_names[0]
        timesteps = int(modalities[first_name]["tokens"].shape[1])
        
        for name in sorted_names:
            mod_data = modalities[name]
            tokens = mod_data["tokens"]
            times = mod_data["time"]
            _, current_timesteps, token_len, _ = tokens.shape
            if int(current_timesteps) != timesteps:
                raise ValueError(
                    "All modalities must share the same timestep count for timestep-interleaved flattening. "
                    f"Expected {timesteps}, got {current_timesteps} for modality '{name}'."
                )

            active_modality_names.append(name)
            modality_lengths.append(timesteps * token_len)
            modality_shapes[name] = (timesteps, token_len)

        for timestep in range(timesteps):
            for name in active_modality_names:
                mod_data = modalities[name]
                interleaved_tokens.append(mod_data["tokens"][:, timestep, :, :])
                interleaved_times.append(mod_data["time"][:, timestep, :])
                interleaved_rope_pos.append(mod_data["rope_pos"][:, timestep, :, :])

        x_flat = torch.cat(interleaved_tokens, dim=1)
        target_time = torch.cat(interleaved_times, dim=1)
        target_rope_pos = torch.cat(interleaved_rope_pos, dim=1)
        return x_flat, target_time, target_rope_pos, active_modality_names, modality_lengths, modality_shapes

    def fulldata_to_context_embedder_input(self, full_data: FullData) -> Dict[str, torch.Tensor]:
        """
        Creates distinct, decoupled LTM and STM streams for the ContextEmbedder.
        Applies modality dropping and temporal chunking logic to the LTM.
        
        Returns:
            - ltm_tokens: [B, T_ltm, L_ltm, D] - LTM stream with concatenated modalities
            - stm_tokens: [B, T_stm, L_stm, D] - STM stream with concatenated modalities (not flattened)
                        - stm_flat_tokens: [B, N_stm, D] - STM stream flattened in the same timestep-major
                            token ordering used by the target decoder input path
            - stm_dataframe_indices: [B, T_stm] - Start time indices for STM dataframes (for temporal embeddings)
        """
        t = FullData.infer_time_length(full_data)

        # 1. Build Long Term Memory (LTM) Stream
        ltm_modalities = self._prepare_modalities(
            full_data,
            drop_modalities=self.ltm_drop_modalities,
            include_modalities=self.context_modalities,
            local_time_period=self.ltm_downsample_chunk_length,
            ltm_mode=True,
        )
        # Concatenate in deterministic order to match encode/decode consistency
        ltm_sorted_names = [name for name in MODALITY_ORDER if name in ltm_modalities]
        ltm_tokens = torch.cat([ltm_modalities[name]["tokens"] for name in ltm_sorted_names], dim=2)

        # 2. Build Short Term Memory (STM) Stream — last stm_context_length units of context.
        stm_start = max(0, t - self.stm_context_length)
        stm_source = FullData.slice_time(full_data, stm_start, t)
        stm_modalities = self._prepare_modalities(
            stm_source,
            include_modalities=self.context_modalities,
            ltm_mode=False,
        )
        stm_flat_tokens, stm_time, stm_rope_pos, _, _, _ = self._flatten_sequence(stm_modalities)

        # Concatenate STM modalities in deterministic order (consistent with _flatten_sequence order)
        stm_sorted_names = [name for name in MODALITY_ORDER if name in stm_modalities]
        stm_tokens = torch.cat([stm_modalities[name]["tokens"] for name in stm_sorted_names], dim=2)
        
        # Extract STM dataframe indices for temporal embeddings (use start index of each dataframe)
        stm_dataframe_indices = self._get_relative_indices(stm_source)

        return {
            "ltm_tokens": ltm_tokens,
            "stm_tokens": stm_tokens,
            "stm_flat_tokens": stm_flat_tokens,
            "stm_time": stm_time,
            "stm_rope_pos": stm_rope_pos,
            "stm_dataframe_indices": stm_dataframe_indices,
        }
    
    def fulldata_to_moe_decoder_input(
        self,
        full_data: FullData,
        mask_type: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Packs a FullData object into a 1D token sequence [B, N, D] for the DiT decoder, 
        generating the appropriate sequence-layout attention mask.
        """
        modalities = self._prepare_modalities(
            full_data,
            include_modalities=self.target_modalities,
            ltm_mode=False,
        )

        x_flat, target_time, target_rope_pos, active_modality_names, _, modality_shapes = (
            self._flatten_sequence(modalities)
        )

        # Extract context safely post-flattening
        device = x_flat.device
        B, Q_LEN = target_time.shape
        player_emb = self._build_player_embedding(full_data.metadata, device, B)

        effective_mask_type = self.mask_type if mask_type is None else mask_type
        block_mask = self._build_attention_mask(
            target_time=target_time,
            active_modality_names=active_modality_names,
            modality_shapes=modality_shapes,
            mask_type=effective_mask_type,
            B=B,
            Q_LEN=Q_LEN,
            KV_LEN=Q_LEN,
        )

        return {
            "x_flat": x_flat,
            "target_time": target_time,
            "target_rope_pos": target_rope_pos,
            "active_modality_names": active_modality_names,
            "modality_shapes": modality_shapes,
            "block_mask": block_mask,
            "player_emb": player_emb,
        }

    def moe_decoder_output_to_fulldata(
        self,
        x_flat: torch.Tensor,
        active_modality_names: List[str],
        modality_shapes: Dict[str, Tuple[int, int]],
        reference_full_data: FullData,
    ) -> FullData:
        """
        Converts flat MoE decoder predictions back into original FullData structures
        by running them through the specific output projection layers.
        
        Args:
            x_flat: Flattened decoder predictions [B, N, D]
            active_modality_names: List of modality names in output order
            modality_shapes: Dict mapping modality names to (timesteps, token_len) tuples
            reference_full_data: Reference FullData for batch structure
        
        Returns:
            FullData with decoded modalities
        """
        if not active_modality_names:
            return reference_full_data

        first_name = active_modality_names[0]
        timesteps = int(modality_shapes[first_name][0])
        tokens_per_modality = [int(modality_shapes[name][1]) for name in active_modality_names]

        for name in active_modality_names:
            if int(modality_shapes[name][0]) != timesteps:
                raise ValueError(
                    "All active modalities must share timestep count for timestep-interleaved decode. "
                    f"Expected {timesteps}, got {modality_shapes[name][0]} for modality '{name}'."
                )

        tokens_per_timestep = int(sum(tokens_per_modality))
        expected_tokens = int(timesteps * tokens_per_timestep)
        if int(x_flat.shape[1]) != expected_tokens:
            raise ValueError(
                "x_flat length does not match timestep-interleaved modality layout. "
                f"Expected {expected_tokens}, got {x_flat.shape[1]}."
            )

        timestep_chunks = torch.split(x_flat, tokens_per_timestep, dim=1)
        gathered: Dict[str, List[torch.Tensor]] = {name: [] for name in active_modality_names}
        for step_chunk in timestep_chunks:
            per_mod_chunks = torch.split(step_chunk, tokens_per_modality, dim=1)
            for name, mod_chunk in zip(active_modality_names, per_mod_chunks):
                gathered[name].append(mod_chunk)

        outputs = {
            name: torch.cat(chunks, dim=1)
            for name, chunks in gathered.items()
        }

        decoded = {}
        for name, tokens in outputs.items():
            if name not in self.output_projectors or name not in modality_shapes:
                continue
            
            # Project from model_dim back to raw feature shapes
            raw_tokens = self._apply_linear(self.output_projectors[name], tokens)  
            b, _, d = raw_tokens.shape
            T, L_mod = modality_shapes[name]
            
            raw_tokens_reshaped = raw_tokens.reshape(b, T, L_mod, d)
            
            if name == "video":
                # Unpatchify video back to [B, T, F, C, H, W]
                v_shape = MODALITY_SHAPES["video"]
                F, C, H, W = v_shape[0], v_shape[1], v_shape[2], v_shape[3]
                gh, gw = H // self.patch_h, W // self.patch_w
                
                reshaped = raw_tokens_reshaped.reshape(b, T, F, gh, gw, C, self.patch_h, self.patch_w)
                video_frames = reshaped.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(b, T, F, C, H, W)
                decoded[name] = video_frames
            else:
                # 1D Modalities are already perfectly shaped [B, T, L_mod, Raw_Dim]!
                decoded[name] = raw_tokens_reshaped
        
        pred_batch = reference_full_data.to_dict()
        for modality, value in decoded.items():
            pred_batch[modality] = value

        return FullData(batch=pred_batch)

    # --------------------------------------------------------------------------------
    # Internal Helpers (Masks, Time, Fourier)
    # --------------------------------------------------------------------------------

    def _build_attention_mask(
        self,
        target_time: torch.Tensor,
        active_modality_names: List[str],
        modality_shapes: Dict[str, Tuple[int, int]],
        mask_type: str,
        B: int,
        Q_LEN: int,
        KV_LEN: int,
    ) -> Optional[BlockMask]:
        """Builds or retrieves a cached attention mask based on sequence topology."""
        if mask_type == "no_mask":
            return None

        # 1. Construct a rigorous, hashable key representing the sequence layout
        cache_key = (
            mask_type,
            B,
            Q_LEN,
            KV_LEN,
            tuple(active_modality_names),
            tuple(sorted(modality_shapes.items()))
        )

        # 2. Return cached instance if sequence topology matches
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # 3. Cache miss: build the mask
        device = target_time.device
        mask = None

        if mask_type == "token_level":
            mask = create_token_level_mask(
                timestamps=target_time,
                B=B,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=device,
            )
        elif mask_type == "dataframe_level":
            if not active_modality_names:
                raise ValueError("dataframe_level mask requires non-empty active_modality_names")

            first_name = active_modality_names[0]
            num_timesteps = int(modality_shapes[first_name][0])
            modality_layout: List[Tuple[str, int]] = [
                (name, int(modality_shapes[name][1])) for name in active_modality_names
            ]

            mask = create_dataframe_level_mask(
                modality_layout=modality_layout,
                num_timesteps=num_timesteps,
                B=B,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=device,
            )
        else:
            raise ValueError(f"Unknown mask_type '{mask_type}'")

        # 4. Store and return
        self._mask_cache[cache_key] = mask
        return mask
    
    def _build_player_embedding(self, metadata, device: torch.device, batch_size: int) -> torch.Tensor:
        items = FullData._unwrap_non_tensor(metadata)
        if not isinstance(items, list):
            items = [items] * batch_size

        names, genders, skills = [], [], []
        for item in items:
            entry = FullData._unwrap_non_tensor(item)
            entry = FullData._unwrap_non_tensor(entry[0]) if isinstance(entry, list) and entry else entry
            
            names.append(self._hash_to_bucket(entry.get("player_name", ""), self.name_embedding.num_embeddings))
            genders.append(self._hash_to_bucket(entry.get("player_gender", ""), self.gender_embedding.num_embeddings))
            skills.append(self._hash_to_bucket(entry.get("player_skill_level", ""), self.skill_embedding.num_embeddings))

        emb = (
            self.name_embedding(torch.tensor(names, device=device, dtype=torch.long))
            + self.gender_embedding(torch.tensor(genders, device=device, dtype=torch.long))
            + self.skill_embedding(torch.tensor(skills, device=device, dtype=torch.long))
        )
        return self.player_norm(emb)

    @staticmethod
    def _hash_to_bucket(value, buckets: int) -> int:
        return int(hashlib.md5(str(value).encode("utf-8")).hexdigest(), 16) % int(buckets)