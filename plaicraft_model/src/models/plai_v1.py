import torch
import torch.nn as nn
from typing import Optional

from data.data_classes import FullData
from models.components.timestep_embedder import TimestepEmbedder


class PlaiV1Model(nn.Module):
    """
    Plai v1 model backbone with pluggable generative noise_scheduler.
    
    Handles:
    - Multi-modal context encoding via Perceiver (Phase 1-2)
    - LTM/STM split with recurrent summarization
    - DiT-based generative modeling with cross-attention (Phase 3)
    - Multi-modal decoder heads
    """
    
    def __init__(self, cfg=None, context_embedder=None, moe_decoder=None, multimodal_io=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        params = dict(cfg) if cfg is not None else {}
        params.update(kwargs)
        
        self.h_dim = params.get('h_dim', 512)
        checkpointing_cfg = params.get('checkpointing', {}) or {}
        self.checkpointing = {
            'perceiver': bool(checkpointing_cfg.get('perceiver', True)),
            'rnn': bool(checkpointing_cfg.get('rnn', True)),
            'moe_decoder': bool(checkpointing_cfg.get('moe_decoder', True)),
        }
        self.context_embedder = context_embedder
        self.moe_decoder = moe_decoder
        self.multimodal_io = multimodal_io
        self.ltm_conditioning_mode = getattr(
            self.context_embedder,
            "ltm_conditioning_mode",
            "cross_attention",
        )

        self.timestep_embedder = TimestepEmbedder(self.h_dim)
        # Keep conditioning branches on comparable scale before AdaLN mixing.
        self.time_cond_norm = nn.LayerNorm(self.h_dim)
        self.ltm_cond_norm = nn.LayerNorm(self.h_dim)

    def forward(
        self,
        x_tau: FullData,
        tau: torch.Tensor,
        context: Optional[FullData] = None,
        memory: Optional[dict[str, torch.Tensor]] = None,
    ) -> FullData:
        """
        Forward pass for generative plai_v1 prediction.

        Args:
            x_tau: Noisy target in original data space.
            tau: Diffusion timestep tensor [B, 1].
            context: Clean conditioning context in original data space.
            memory: Dict with `stm` and `ltm` tensors used for conditioning.

        Returns:
            FullData containing predicted velocities in original data space.
        """
        # 1. Get timestep embedding.
        time_emb = self.time_cond_norm(self.timestep_embedder(tau.squeeze(-1)))
        target = self.multimodal_io.fulldata_to_moe_decoder_input(x_tau)
        player_emb = target.get("player_emb", torch.zeros_like(time_emb))
        # Get LTM and STM from precomputed memory or encode context online
        if memory is None:
            if context is None:
                raise ValueError("Must provide either 'context' or 'memory'.")
            context_dict = self.multimodal_io.fulldata_to_context_embedder_input(context)
            output = self.context_embedder(
                context_dict,
                gradient_checkpoint_perceiver=self.checkpointing['perceiver'],
                gradient_checkpoint_rnn=self.checkpointing['rnn'],
            )
            ltm = output["ltm"]
            stm = output["stm"]
            history_time = output.get("stm_rope_pos", output.get("stm_time", None))
        else:
            ltm = memory.get("ltm", None)
            stm = memory.get("stm", None)
            history_time = memory.get("stm_rope_pos", memory.get("stm_time", None))

        if stm is None:
            raise ValueError("Could not resolve STM from memory/context")

        # Prepare history tokens for decoder
        if self.ltm_conditioning_mode == "cross_attention":
            history_tokens = torch.cat([ltm, stm], dim=1)
        else:
            history_tokens = stm

        # Build conditioning embedding based on mode
        cond_parts = [time_emb, player_emb]
        if self.ltm_conditioning_mode == "adaln":
            if ltm is None:
                raise ValueError("adaln mode requires ltm in memory/context")
            cond_parts.append(self.ltm_cond_norm(ltm))
        cond_emb = torch.stack(cond_parts, dim=1)
        
        # 5. MoE Decoder forward pass.
        predictions = self.moe_decoder(
            x_flat=target["x_flat"],
            cond_emb=cond_emb,
            modality_shapes=target["modality_shapes"],
            active_modality_names=target["active_modality_names"],
            block_mask=target["block_mask"],
            history=history_tokens,
            history_time=history_time,
            target_time=target.get("target_rope_pos", target.get("target_time", None)),
            gradient_checkpoint=self.checkpointing['moe_decoder'],
        )

        return self.multimodal_io.moe_decoder_output_to_fulldata(
            x_flat=predictions,
            active_modality_names=target["active_modality_names"],
            modality_shapes=target["modality_shapes"],
            reference_full_data=x_tau,
        )