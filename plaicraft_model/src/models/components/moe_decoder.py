"""
Modality-Aware Mixture of Experts (MoE) Transformer

Architecture:
- Dictionary input with multiple modalities
- Modality-specific projection MLPs
- Sinusoidal positional encodings per modality
- Modality type embeddings
- Shared attention across all modalities
- Deterministic modality-based routing to expert MLPs
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask

from models.components.attention import Attention


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply adaptive layer normalization modulation.
    
    Args:
        x: Input tensor [B, N, D]
        shift: Shift parameter [B, D]
        scale: Scale parameter [B, D]
    
    Returns:
        Modulated tensor [B, N, D]
    """
    return x * (1 + rearrange(scale, 'b d -> b 1 d')) + rearrange(shift, 'b d -> b 1 d')


@dataclass
class ModalityConfig:
    """Configuration for a single modality."""
    input_dim: int
    hidden_dim: int


class ModalityProjector(nn.Module):
    """Projects a modality from its input dimension to the shared embedding dimension."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project modality input to shared embedding dimension.
        
        Args:
            x: Input tensor [..., input_dim]
        
        Returns:
            Projected tensor [..., output_dim]
        """
        return self.mlp(x)


class ModalityExpert(nn.Module):
    """MLP expert for a specific modality."""
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process tokens through expert MLP.
        
        Args:
            x: Input tensor [..., embed_dim]
        
        Returns:
            Output tensor [..., embed_dim]
        """
        return self.mlp(x)



class ModalityMoEBlock(nn.Module):
    """
    Transformer block with shared attention, optional cross-attention, and modality-specific experts.
    Capable of handling unified temporal sequences with mixed modalities.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        modality_names: List[str],
        modality_hidden_dims: Dict[str, int],
        modality_map: Dict[str, int],
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        use_weightnorm: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_max_spatial_freq: float = 1.0,
        rope_mrope_section: Optional[List[int]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_names = modality_names
        self.modality_map = modality_map
        self.use_cross_attention = use_cross_attention
        self.use_rope = use_rope
        
        # AdaLN-Zero: LayerNorm without affine parameters (scale/shift applied via modulation)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        
        # Determine modulation output size: 6 base + 1 for cross-attn gate if enabled
        modulation_output_dim = (7 if use_cross_attention else 6) * embed_dim
        
        # AdaLN modulation network
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, modulation_output_dim, bias=True)
        )
        # Zero-initialize the modulation layers for training stability
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        
        # Shared attention (RoPE enabled/disabled based on use_rope flag)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            use_weightnorm=use_weightnorm,
            rope_theta=rope_theta,
            rope_max_spatial_freq=rope_max_spatial_freq,
            rope_mrope_section=rope_mrope_section,
        )
        
        # Cross-attention (optional)
        if use_cross_attention:
            self.cross_attn = Attention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=use_rope,
                use_weightnorm=use_weightnorm,
                rope_theta=rope_theta,
                rope_max_spatial_freq=rope_max_spatial_freq,
                rope_mrope_section=rope_mrope_section,
            )
            self.cross_attn_norm = nn.LayerNorm(embed_dim)
            self.cross_attn_dropout = nn.Dropout(dropout)
        
        # Modality-specific experts
        self.experts = nn.ModuleDict({
            name: ModalityExpert(
                embed_dim=embed_dim,
                hidden_dim=modality_hidden_dims[name],
                dropout=dropout,
            )
            for name in modality_names
        })
        
        self.dropout = nn.Dropout(dropout)
        self._route_index_cache: Dict[Tuple[str, int, Tuple[int, ...], int], List[torch.Tensor]] = {}

    def _get_route_indices(
        self,
        num_timesteps: int,
        tokens_per_modality: List[int],
        seq_len: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Build cached token index lists for timestep-interleaved modality routing."""
        device_key = f"{device.type}:{device.index}"
        cache_key = (device_key, int(num_timesteps), tuple(int(v) for v in tokens_per_modality), int(seq_len))
        cached = self._route_index_cache.get(cache_key)
        if cached is not None:
            return cached

        tokens_per_timestep = int(sum(tokens_per_modality))
        expected_seq_len = int(num_timesteps * tokens_per_timestep)
        if expected_seq_len != int(seq_len):
            raise ValueError(
                "Token count does not match interleaved modality routing layout. "
                f"Expected {expected_seq_len}, got {seq_len}."
            )

        route_indices: List[torch.Tensor] = []
        timestep_bases = torch.arange(num_timesteps, device=device, dtype=torch.long) * tokens_per_timestep
        start = 0
        for tokens_per_step in tokens_per_modality:
            local = torch.arange(tokens_per_step, device=device, dtype=torch.long)
            indices = (timestep_bases[:, None] + start + local[None, :]).reshape(-1)
            route_indices.append(indices)
            start += tokens_per_step

        self._route_index_cache[cache_key] = route_indices
        return route_indices
    
    def forward(
        self,
        x: torch.Tensor,
        adaln_emb: torch.Tensor,
        modality_shapes: Dict[str, Tuple[int, int]],
        active_modality_names: List[str],
        block_mask: Optional[BlockMask] = None,
        history: Optional[torch.Tensor] = None,
        history_time: Optional[torch.Tensor] = None,
        target_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with shared attention, optional cross-attention, and modality-specific experts.
        
        Args:
            x: Input tokens [B, N, D]
            adaln_emb: AdaLN modulation embedding [B, D] (can include player metadata + diffusion time)
            modality_shapes: Dict mapping modality names to (num_tokens, token_dim) tuples
            active_modality_names: List of active modality names to process
            block_mask: Optional flex BlockMask for self-attention
            history: Optional context for cross-attention [B, K, D]
            history_time: Optional RoPE coordinates [B, K] or [B, K, C] for cross-attention
            target_time: Optional RoPE coordinates [B, N] or [B, N, C] for self-attention
        """
        missing_experts = [name for name in active_modality_names if name not in self.experts]
        if missing_experts:
            raise ValueError(
                "Active modalities are missing corresponding MoE experts. "
                f"missing={missing_experts}, configured_experts={list(self.experts.keys())}"
            )

        missing_shapes = [name for name in active_modality_names if name not in modality_shapes]
        if missing_shapes:
            raise ValueError(
                "modality_shapes is missing active modalities required for routing. "
                f"missing={missing_shapes}, provided={list(modality_shapes.keys())}"
            )

        if not active_modality_names:
            raise ValueError("active_modality_names must be non-empty")

        first_name = active_modality_names[0]
        num_timesteps = int(modality_shapes[first_name][0])
        tokens_per_modality = [int(modality_shapes[name][1]) for name in active_modality_names]
        for name in active_modality_names:
            if int(modality_shapes[name][0]) != num_timesteps:
                raise ValueError(
                    "All active modalities must share timestep count for interleaved routing. "
                    f"Expected {num_timesteps}, got {modality_shapes[name][0]} for modality '{name}'."
                )

        route_indices = self._get_route_indices(
            num_timesteps=num_timesteps,
            tokens_per_modality=tokens_per_modality,
            seq_len=int(x.shape[1]),
            device=x.device,
        )
        
        # Generate adaptive modulation parameters from adaln_emb
        modulation_outputs = self.adaLN_modulation(adaln_emb).chunk(7 if self.use_cross_attention else 6, dim=1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation_outputs[:6]
        gate_cross = modulation_outputs[6] if self.use_cross_attention else None
        
        # 1. Shared attention across all modalities
        # Cast to FP32 for LayerNorm (AMP-safe with elementwise_affine=False)
        x_dtype = x.dtype
        x_norm = self.norm1(x.float()).to(x_dtype)
        x_modulated = modulate(x_norm, shift_attn, scale_attn)
        
        attn_out = self.attn(
            q_input=x_modulated,
            k_input=x_modulated,
            v_input=x_modulated,
            block_mask=block_mask,
            q_pos=target_time if self.use_rope else None,
            k_pos=target_time if self.use_rope else None,
        )
        x = x + rearrange(gate_attn, 'b d -> b 1 d') * self.dropout(attn_out)
        
        # 2. Route interleaved tokens to modality experts and scatter back.
        # Cast to FP32 for LayerNorm (AMP-safe with elementwise_affine=False)
        x_norm2 = self.norm2(x.float()).to(x_dtype)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        expert_out = torch.zeros_like(x_modulated2)
        for name, token_indices in zip(active_modality_names, route_indices):
            selected_tokens = x_modulated2.index_select(1, token_indices)
            processed_tokens = self.experts[name](selected_tokens)
            expert_out.index_copy_(1, token_indices, processed_tokens)
        
        # 3. Residual connection for experts
        x = x + rearrange(gate_mlp, 'b d -> b 1 d') * self.dropout(expert_out)
        
        # 4. Cross-attention (optional)
        if self.use_cross_attention and history is not None and history.shape[1] > 0:
            x_norm_cross = self.cross_attn_norm(x)
            cross_out = self.cross_attn(
                q_input=x_norm_cross,
                k_input=history,
                v_input=history,
                block_mask=None,
                q_pos=target_time if self.use_rope else None,
                k_pos=history_time if self.use_rope else None,
            )
            x = x + rearrange(gate_cross, 'b d -> b 1 d') * self.cross_attn_dropout(cross_out)
        
        return x


class MoEDecoder(nn.Module):
    """
    MoE Decoder math engine for a pre-packed flat multimodal sequence.

    Sequence packing/unpacking and attention-mask construction are handled by
    MultimodalIO. This module only applies transformer blocks and expert MLPs
    over [B, N, D] tensors.
    """
    

    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        modality_configs: Dict[str, Dict[str, int]],
        target_modalities: Optional[List[str]] = None,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        ltm_conditioning_mode: str = "cross_attention",
        use_weightnorm: bool = False,
        positional_encoding_type: str = "fourier",
        use_stm_perceiver: bool = True,
        rope_theta: float = 10000.0,
        rope_max_spatial_freq: float = 1.0,
        rope_mrope_section: Optional[List[int]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        if target_modalities is not None:
            self.modality_names = [name for name in modality_configs.keys() if name in target_modalities]
            self.modality_hidden_dims = {
                name: modality_configs[name]['hidden_dim']
                for name in self.modality_names
            }
        else:
            self.modality_names = list(modality_configs.keys())
            self.modality_hidden_dims = {
                name: config['hidden_dim']
                for name, config in modality_configs.items()
            }
        self.use_cross_attention = use_cross_attention
        self.ltm_conditioning_mode = str(ltm_conditioning_mode)
        if self.ltm_conditioning_mode not in {"cross_attention", "adaln"}:
            raise ValueError(
                f"Unsupported ltm_conditioning_mode='{self.ltm_conditioning_mode}'. "
                "Expected one of: 'cross_attention', 'adaln'."
            )
        self.cond_token_count = 3 if self.ltm_conditioning_mode == "adaln" else 2
        
        self.positional_encoding_type = str(positional_encoding_type)
        if self.positional_encoding_type not in {"fourier", "trainable", "rope"}:
            raise ValueError(
                f"Unsupported positional_encoding_type='{self.positional_encoding_type}'. "
                "Expected one of: 'fourier', 'trainable', 'rope'."
            )

        # RoPE is selected exclusively via positional_encoding_type.
        self.use_rope = self.positional_encoding_type == "rope"
        if self.use_rope and not (not use_stm_perceiver and self.ltm_conditioning_mode == "adaln"):
            raise ValueError(
                f"positional_encoding_type='rope' requires raw STM (use_stm_perceiver=False) and "
                f"adaln mode (ltm_conditioning_mode='adaln'), got use_stm_perceiver={use_stm_perceiver}, "
                f"ltm_conditioning_mode='{self.ltm_conditioning_mode}'"
            )
        
        # Create map for modality indices
        self.modality_map = {name: i for i, name in enumerate(self.modality_names)}
        
        # Transformer blocks (cross-attention now integrated into blocks)
        self.blocks = nn.ModuleList([
            ModalityMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                modality_names=self.modality_names,
                modality_hidden_dims=self.modality_hidden_dims,
                modality_map=self.modality_map,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
                use_weightnorm=use_weightnorm,
                use_rope=self.use_rope,
                rope_theta=rope_theta,
                rope_max_spatial_freq=rope_max_spatial_freq,
                rope_mrope_section=rope_mrope_section,
            )
            for _ in range(num_layers)
        ])
        
        # Normalize history once at the start if using cross-attention
        if use_cross_attention:
            self.history_norm = nn.LayerNorm(embed_dim)
        
        # Final layer norm (keep affine=True for final norm, it's not part of AdaLN)
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.adaln_proj = nn.Sequential(
            nn.Linear(self.cond_token_count * embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_flat: torch.Tensor,
        cond_emb: torch.Tensor,
        modality_shapes: Dict[str, Tuple[int, int]],
        active_modality_names: List[str],
        block_mask: Optional[BlockMask] = None,
        history: Optional[torch.Tensor] = None,
        history_time: Optional[torch.Tensor] = None,
        target_time: Optional[torch.Tensor] = None,
        gradient_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass over a pre-packed flat multimodal sequence.
        
        Args:
            x_flat: Packed tokens [B, N, D]
            cond_emb: Combined timestep and player embeddings [B, 2, D]
            modality_shapes: Dict mapping modality names to (num_tokens, token_dim) tuples
            active_modality_names: Ordered modality names matching modality_shapes
            block_mask: Optional flex BlockMask for self-attention
            history: Unified memory [B, K, D] from context encoder
        
        Returns:
            Flat predictions [B, N, D]
        """
        # Project combined embeddings: reshape [B, K, D] -> [B, K*D] -> [B, D]
        B, K, D = cond_emb.shape
        if K != self.cond_token_count:
            raise ValueError(
                f"Expected cond_emb with {self.cond_token_count} conditioning tokens in "
                f"ltm_conditioning_mode='{self.ltm_conditioning_mode}', got {K}"
            )
        cond_emb_flat = cond_emb.reshape(B, -1)  # [B, 2*D]
        adaln_emb = self.adaln_proj(cond_emb_flat)  # [B, D]

        # Forward through blocks
        x = x_flat

        if self.use_cross_attention and history is not None and history.shape[1] > 0:
            history = self.history_norm(history)

        if self.use_rope and target_time is None:
            raise ValueError("positional_encoding_type='rope' requires target_time for decoder attention")
        if self.use_rope and self.use_cross_attention and history is not None and history.shape[1] > 0 and history_time is None:
            raise ValueError("positional_encoding_type='rope' with cross-attention requires history_time")

        use_checkpoint = gradient_checkpoint and torch.is_grad_enabled()
        
        for i, block in enumerate(self.blocks):
            # Define layer function with explicit argument passing for checkpointing
            def _run_layer(x_in, emb, hist, t_target, idx=i, b=block):
                return b(
                    x=x_in,
                    adaln_emb=emb,
                    modality_shapes=modality_shapes,
                    active_modality_names=active_modality_names,
                    block_mask=block_mask,
                    history=hist,
                    history_time=history_time,
                    target_time=t_target,
                )

            if use_checkpoint:
                x = checkpoint(_run_layer, x, adaln_emb, history, target_time, use_reentrant=False)
            else:
                x = _run_layer(x, adaln_emb, history, target_time)
        
        x = self.final_norm(x)
        return x

    @classmethod
    def from_config(cls, config):
        """Helper to init from config object."""
        if hasattr(config, '__dataclass_fields__'):
            modality_configs = {
                name: {'hidden_dim': mod_cfg.hidden_dim}
                for name, mod_cfg in config.modalities.items()
            }
            return cls(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                modality_configs=modality_configs,
                target_modalities=getattr(config, 'target_modalities', None),
                dropout=config.dropout,
                use_cross_attention=getattr(config, 'use_cross_attention', False),
                ltm_conditioning_mode=getattr(config, 'ltm_conditioning_mode', 'cross_attention'),
                use_weightnorm=getattr(config, 'use_weightnorm', False),
                positional_encoding_type=getattr(config, 'positional_encoding_type', 'fourier'),
                use_stm_perceiver=getattr(config, 'use_stm_perceiver', True),
                rope_theta=getattr(config, 'rope_theta', 10000.0),
                rope_max_spatial_freq=getattr(config, 'rope_max_spatial_freq', 1.0),
                rope_mrope_section=getattr(config, 'rope_mrope_section', None),
            )
        else:
            return cls(**config)
