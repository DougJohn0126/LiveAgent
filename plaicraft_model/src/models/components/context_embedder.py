"""ContextEmbedder - build unified memory from decoupled LTM and STM streams."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from models.components.positional_encoding import (
    build_position_encoding,
    dataframe_indices_to_metric_seconds,
)
from models.components.perceiver import PerceiverIO
from models.components.recurrent_encoders.min_gru import MinGRUEncoder
from models.components.recurrent_encoders.xlstm import xLSTMEncoder
from utils.constants import UNIT_DURATION_SECONDS


class FactorizedSequenceCompressor(nn.Module):
    """
    Compresses a sequence of latents into a single vector using factorized projections.
    
    This module reduces parameters by using a bottleneck dimension, significantly reducing
    memory and computation compared to a full linear projection from seq_len * in_dim to out_dim.
    Parameter reduction: O(seq_len * in_dim * out_dim) -> O(in_dim * r + seq_len * r * out_dim)
    where r is the bottleneck dimension.
    """
    def __init__(
        self,
        seq_len: int,
        in_dim: int,
        out_dim: int,
        bottleneck_dim: int = 64
    ) -> None:
        """
        Initialize the factorized sequence compressor.
        
        Args:
            seq_len: Length of the sequence dimension to compress over
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            bottleneck_dim: Intermediate bottleneck dimension for parameter reduction (default: 64)
        """
        super().__init__()
        # Channel-wise feature extraction (shared across sequence)
        self.norm = nn.LayerNorm(in_dim)
        self.channel_compress = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.GELU()
        
        # Spatial integration mapping to the final vector dimension
        self.spatial_combine = nn.Linear(seq_len * bottleneck_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress a sequence of latents to a single vector.
        
        First applies channel-wise feature compression with normalization and activation,
        then flattens spatial dimensions and applies spatial integration to produce output.
        
        Args:
            x: Input tensor of shape [B, T, N, in_dim] or [B, N, in_dim]
        
        Returns:
            Compressed tensor of shape [B, T, out_dim] or [B, out_dim]
        """
        # Feature compression
        x = self.norm(x)
        x = self.channel_compress(x)
        x = self.act(x)
        
        # Flatten the spatial dimension (N)
        x = rearrange(x, "b t n d -> b t (n d)")

        # Spatial integration
        return self.spatial_combine(x)


class FactorizedSequenceExpander(nn.Module):
    """
    Expands a single vector into a sequence of latents using factorized projections.
    
    This module reconstructs a sequence of outputs from a compressed representation,
    using a bottleneck dimension to reduce parameters compared to a full linear projection.
    Parameter reduction: O(in_dim * seq_len * out_dim) -> O(in_dim * seq_len * r + r * out_dim)
    where r is the bottleneck dimension.
    """
    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        out_dim: int,
        bottleneck_dim: int = 128
    ) -> None:
        """
        Initialize the factorized sequence expander.
        
        Args:
            in_dim: Input feature dimension (compressed representation)
            seq_len: Length of the sequence to expand to
            out_dim: Output feature dimension for each element in the sequence
            bottleneck_dim: Intermediate bottleneck dimension for parameter reduction (default: 128)
        """
        super().__init__()
        self.seq_len = seq_len
        self.norm = nn.LayerNorm(in_dim)
        # Spatial expansion generating sequence seeds
        self.spatial_expand = nn.Linear(in_dim, seq_len * bottleneck_dim)
        
        # Channel-wise up-projection (shared across sequence)
        self.channel_project = nn.Linear(bottleneck_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand a compressed vector into a sequence of latents.
        
        Applies normalization, spatial expansion to generate sequence seeds,
        then further expansion in the feature dimension to produce the output sequence.
        
        Args:
            x: Compressed input tensor of shape [B, in_dim]
        
        Returns:
            Expanded sequence tensor of shape [B, seq_len, out_dim]
        """
        
        # Normalization
        x = self.norm(x)
        
        # Spatial expansion
        x = self.spatial_expand(x)
        x = rearrange(x, "b (k d) -> b k d", k=self.seq_len)
        
        return self.channel_project(x)

class ContextEmbedder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        stm_context_length: Optional[int] = None,
        ltm_downsample_chunk_length: int = 1,
        chunk_len: int = 100,
        k_ltm: int = 64,
        rnn_config: Dict[str, Any] = None,
        # LTM Perceiver parameters
        ltm_num_latents: int = 32,
        ltm_perceiver_depth: int = 8,
        ltm_perceiver_cross_heads: int = 4,
        ltm_perceiver_latent_heads: int = 8,
        ltm_perceiver_cross_dim_head: int = 64,
        ltm_perceiver_latent_dim_head: int = 64,
        ltm_perceiver_seq_dropout: float = 0.0,
        # STM Perceiver parameters
        stm_num_latents: int = 32,
        stm_perceiver_depth: int = 8,
        stm_perceiver_cross_heads: int = 4,
        stm_perceiver_latent_heads: int = 8,
        stm_perceiver_cross_dim_head: int = 64,
        stm_perceiver_latent_dim_head: int = 64,
        stm_perceiver_seq_dropout: float = 0.0,
        use_stm_perceiver: bool = True,
        ltm_conditioning_mode: str = "cross_attention",
        temporal_fourier_min_resolution: Optional[float] = None,
        temporal_fourier_max_resolution: Optional[float] = None,
        stm_fourier_num_bands: Optional[int] = None,
        use_weightnorm: bool = False,

    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.ltm_num_latents = ltm_num_latents
        self.stm_num_latents = stm_num_latents
        self.use_stm_perceiver = bool(use_stm_perceiver)
        self.ltm_conditioning_mode = str(ltm_conditioning_mode)
        if self.ltm_conditioning_mode not in {"cross_attention", "adaln"}:
            raise ValueError(
                f"Unsupported ltm_conditioning_mode='{self.ltm_conditioning_mode}'. "
                "Expected one of: 'cross_attention', 'adaln'."
            )
        self.stm_context_length = int(stm_context_length) if stm_context_length is not None else 0
        self.ltm_downsample_chunk_length = int(ltm_downsample_chunk_length)
        self.chunk_len = chunk_len
        self.k_ltm = k_ltm
        self.rnn_config = rnn_config or {}
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
        self.stm_fourier_num_bands = int(stm_fourier_num_bands) if stm_fourier_num_bands is not None else max(1, model_dim // 2)
        self.stm_temporal_pos_encoding = None
        if self.use_stm_perceiver:
            if self.temporal_fourier_min_resolution is None or self.temporal_fourier_max_resolution is None:
                raise ValueError(
                    "Fourier STM positional encoding requires temporal_fourier_min_resolution and temporal_fourier_max_resolution to be configured."
                )
            self.stm_temporal_pos_encoding = build_position_encoding(
                encoding_type="fourier",
                index_dims=(1,),
                output_dim=model_dim,
                fourier_num_bands=self.stm_fourier_num_bands,
                fourier_concat_pos=True,
                fourier_sine_only=False,
                fourier_min_resolution=[self.temporal_fourier_min_resolution],
                fourier_max_resolution=[self.temporal_fourier_max_resolution],
                fourier_apply_nyquist_mask=True,
                fourier_nyquist_fps=(1.0 / UNIT_DURATION_SECONDS),
            )

        self.ltm_perceiver = PerceiverIO(
            depth=ltm_perceiver_depth,
            dim=model_dim,
            queries_dim=model_dim,
            num_latents=ltm_num_latents,
            latent_dim=model_dim,
            cross_heads=ltm_perceiver_cross_heads,
            latent_heads=ltm_perceiver_latent_heads,
            cross_dim_head=ltm_perceiver_cross_dim_head,
            latent_dim_head=ltm_perceiver_latent_dim_head,
            seq_dropout_prob=ltm_perceiver_seq_dropout,
            use_weightnorm=use_weightnorm,
        )

        self.stm_perceiver = None
        if self.use_stm_perceiver:
            self.stm_perceiver = PerceiverIO(
                depth=stm_perceiver_depth,
                dim=model_dim,
                queries_dim=model_dim,
                num_latents=stm_num_latents,
                latent_dim=model_dim,
                cross_heads=stm_perceiver_cross_heads,
                latent_heads=stm_perceiver_latent_heads,
                cross_dim_head=stm_perceiver_cross_dim_head,
                latent_dim_head=stm_perceiver_latent_dim_head,
                seq_dropout_prob=stm_perceiver_seq_dropout,
                use_weightnorm=use_weightnorm,
            )

        # Instantiate the appropriate recurrent encoder based on rnn_type
        rnn_type = self.rnn_config.get('rnn_type', 'mingru')
        rnn_type_lower = rnn_type.lower()
        if rnn_type_lower == 'mingru':
            self.recurrent_encoder = MinGRUEncoder(
                embedding_dim=self.rnn_config.get('embedding_dim', 512),
                num_layers=self.rnn_config.get('num_layers', 4),
                num_heads=self.rnn_config.get('num_heads', 4),
                mlp_multiplier=self.rnn_config.get('mlp_multiplier', 4),
            )
        elif rnn_type_lower == 'xlstm':
            self.recurrent_encoder = xLSTMEncoder(
                embedding_dim=self.rnn_config.get('embedding_dim', 512),
                num_heads=self.rnn_config.get('num_heads', 4),
                num_blocks=self.rnn_config.get('num_blocks', self.rnn_config.get('num_layers', 4)),
                dropout=self.rnn_config.get('dropout', 0.0),
                context_length=self.rnn_config.get('context_length', 8192),
                conv1d_kernel_size=self.rnn_config.get('conv1d_kernel_size', 4),
                qkv_proj_blocksize=self.rnn_config.get('qkv_proj_blocksize', 4),
                slstm_at=tuple(self.rnn_config.get('slstm_at', ())),
                mlstm_chunk_size=self.rnn_config.get('mlstm_chunk_size', 64),
                chunkwise_kernel=self.rnn_config.get('chunkwise_kernel', 'chunkwise--triton_xl_chunk'),
                sequence_kernel=self.rnn_config.get('sequence_kernel', 'native_sequence__triton'),
                step_kernel=self.rnn_config.get('step_kernel', 'triton'),
                backend_mode=self.rnn_config.get('backend_mode', 'train_with_padding'),
            )
        else:
            raise ValueError(
                f"Unknown rnn_type: {rnn_type}. "
                f"Supported types: 'mingru', 'xlstm'"
            )
        
        rnn_dim = self.rnn_config.get('embedding_dim', 512)
        in_bottleneck_dim = self.rnn_config.get('ltm_in_bottleneck_dim', 64)
        self.ltm_in_proj = FactorizedSequenceCompressor(
            seq_len=ltm_num_latents,
            in_dim=model_dim,
            out_dim=rnn_dim,
            bottleneck_dim=in_bottleneck_dim
        )
        self.ltm_state_to_model_dim = nn.Linear(rnn_dim, model_dim)
        self.ltm_projector = None
        if self.ltm_conditioning_mode == "cross_attention":
            out_bottleneck_dim = self.rnn_config.get('ltm_out_bottleneck_dim', 128)
            self.ltm_projector = FactorizedSequenceExpander(
                in_dim=rnn_dim,
                seq_len=k_ltm,
                out_dim=model_dim,
                bottleneck_dim=out_bottleneck_dim
            )
        
        # Type embeddings are only needed when both LTM and STM are mixed in cross-attn history.
        self.ltm_type_emb = None
        self.stm_type_emb = None
        if self.ltm_conditioning_mode == "cross_attention":
            self.ltm_type_emb = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
            self.stm_type_emb = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Internal streaming cache
        self.use_streaming_cache = False
        self._context_cache: Optional[Dict[str, Any]] = None

    def reset_streaming_cache(self) -> None:
        """
        Clear the internal streaming cache.
        
        Used to reset cached RNN states when starting a new streaming inference sequence.
        """
        self._context_cache = None

    def enable_streaming_cache(self, reset_cache: bool = True) -> None:
        """
        Enable streaming cache for inference.
        
        When enabled, the RNN states are cached between forward passes, allowing
        for efficient streaming inference where context is processed frame-by-frame.
        
        Args:
            reset_cache: If True, clear any existing cached state (default: True)
        """
        self.use_streaming_cache = True
        if reset_cache:
            self.reset_streaming_cache()

    def disable_streaming_cache(self, reset_cache: bool = True) -> None:
        """
        Disable streaming cache for standard batch inference.
        
        When disabled, each forward pass processes independently without relying on
        cached RNN states from previous frames.
        
        Args:
            reset_cache: If True, clear any existing cached state (default: True)
        """
        self.use_streaming_cache = False
        if reset_cache:
            self.reset_streaming_cache()

    def _perceive_in_chunks(
        self,
        tokens: torch.Tensor,
        perceiver: Optional[nn.Module] = None,
        num_latents: Optional[int] = None,
        gradient_checkpoint_perceiver: bool = False,
    ) -> torch.Tensor:
        """
        Process tokens through a perceiver in chunks for memory efficiency.
        
        Handles processing of sequences through a PerceiverIO module, optionally
        splitting the batch into chunks and using gradient checkpointing to reduce
        memory usage during backward pass.
        
        Args:
            tokens: Input tokens of shape [B, T, L, D] where B is batch size,
                    T is time steps, L is sequence length, D is feature dimension
            perceiver: The perceiver module to use (defaults to self.ltm_perceiver)
            num_latents: Number of latent outputs from the perceiver (defaults to self.ltm_num_latents)
            gradient_checkpoint_perceiver: If True, enable gradient checkpointing to save memory
        
        Returns:
            Perceiver output of shape [B, T, N, D] where N is num_latents
        """
        b, t, _, d = tokens.shape
        if t == 0:
            return tokens.new_zeros((b, 0, num_latents, d))

        use_checkpoint = gradient_checkpoint_perceiver and torch.is_grad_enabled()

        def _run_perceiver(flat_tokens: torch.Tensor) -> torch.Tensor:
            if use_checkpoint:
                return checkpoint(
                    lambda x: perceiver(x, gradient_checkpoint=False),
                    flat_tokens,
                    use_reentrant=False,
                )
            return perceiver(flat_tokens, gradient_checkpoint=False)

        flat_tokens = rearrange(tokens, "b t m d -> (b t) m d")
        
        # Process in chunks if chunk_len > 0
        if self.chunk_len > 0:
            chunk_size = self.chunk_len
            num_tokens = flat_tokens.shape[0]
            outputs = []
            
            for i in range(0, num_tokens, chunk_size):
                chunk = flat_tokens[i:i + chunk_size]
                chunk_out = _run_perceiver(chunk)
                outputs.append(chunk_out)
            
            out_flat = torch.cat(outputs, dim=0)
        else:
            # Process all at once (original behavior)
            out_flat = _run_perceiver(flat_tokens)
        
        return rearrange(out_flat, "(b t) n d -> b t n d", b=b, t=t)

    def _extract_ltm_state_vector(
        self,
        rnn_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the final state vector from RNN output sequence.
        
        Takes the last timestep's output from the RNN, which contains the compressed
        representation of all prior LTM tokens.
        
        Args:
            rnn_output: RNN output of shape [B, T, D] where T is sequence length
        
        Returns:
            Final state vector of shape [B, D]
        """
        if rnn_output.shape[1] == 0:
            return rnn_output.new_zeros((rnn_output.shape[0], rnn_output.shape[-1]))
        return rnn_output[:, -1]

    def _downsample_ltm_tokens(self, ltm_tokens: torch.Tensor) -> torch.Tensor:
        """
        Downsample LTM tokens by grouping consecutive frames into chunks.
        
        Reduces the temporal sequence length while preserving information by grouping
        frames into chunks and concatenating their latent representations.
        
        Args:
            ltm_tokens: LTM tokens of shape [B, T, L, D]
        
        Returns:
            Downsampled tokens of shape [B, T//k, L*k, D] where k is ltm_downsample_chunk_length
        
        Raises:
            ValueError: If ltm_downsample_chunk_length is not >= 1
        """
        k = self.ltm_downsample_chunk_length
        if k <= 0:
            raise ValueError(f"ltm_downsample_chunk_length must be >= 1, got {k}")

        b, t, l, d = ltm_tokens.shape
        if t == 0:
            return ltm_tokens

        # Keep short contexts as a single chunk instead of dropping all frames.
        if t < k:
            return rearrange(ltm_tokens, "b t l d -> b 1 (t l) d")

        # Right-truncate the newest frames to keep only full historical chunks.
        remainder = t % k
        if remainder > 0:
            ltm_tokens = ltm_tokens[:, : t - remainder, :, :]

        grouped_tokens = rearrange(ltm_tokens, "b (tg k) l d -> b tg (k l) d", k=k)
        return grouped_tokens

    def _generate_stm_temporal_embeddings(
        self,
        stm_perceiver_output: torch.Tensor,
        stm_dataframe_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate Fourier temporal embeddings for STM perceiver output tokens.
        
        Creates temporal positional embeddings based on the metric time (in seconds)
        of each dataframe in the STM. All latents within a dataframe share the same
        temporal embedding, which is added to the perceiver output.
        
        Args:
            stm_perceiver_output: STM perceiver output of shape [B, T_stm, N_stm, D]
                                  where T_stm is number of dataframes, N_stm is number of latents
            stm_dataframe_indices: Start time indices of each dataframe, shape [B, T_stm]
        
        Returns:
            Temporal Fourier embeddings of shape [B, T_stm, N_stm, D] to add to STM tokens
        """
        b, t_stm, n_stm, d = stm_perceiver_output.shape
        metric_time = dataframe_indices_to_metric_seconds(
            stm_dataframe_indices,
            unit_duration_seconds=UNIT_DURATION_SECONDS,
            dtype=stm_perceiver_output.dtype,
        )
        coords = metric_time.unsqueeze(-1).unsqueeze(-1).expand(b, t_stm, n_stm, 1)
        if self.stm_temporal_pos_encoding is None:
            raise RuntimeError("stm_temporal_pos_encoding is only available when use_stm_perceiver=True")
        return self.stm_temporal_pos_encoding(batch_size=b, pos=coords).to(stm_perceiver_output)

    def _build_context_cache(
        self,
        rnn_states: Any,
    ) -> Dict[str, Any]:
        """
        Build the internal cache dictionary for streaming inference.
        
        Packages the RNN states into a cache structure that can be saved and restored
        for resuming streaming inference across multiple forward passes.
        
        Args:
            rnn_states: RNN states from the recurrent encoder
        
        Returns:
            Dictionary containing cached RNN states for later restoration
        """
        return {
            "ltm_state": self.recurrent_encoder.cache_state(rnn_states),
        }

    def _process_ltm(
        self,
        ltm: torch.Tensor,
        b: int,
        gradient_checkpoint_perceiver: bool = False,
        gradient_checkpoint_rnn: bool = False,
    ) -> torch.Tensor:
        """
        Process LTM tokens through Perceiver and RNN encoder.
        
        Handles empty sequences, downsampling, perception, compression via RNN,
        and optional expansion to tokens for cross-attention or state for adaln mode.
        
        Args:
            ltm: LTM tokens of shape [B, T_ltm, L_ltm, D]
            b: Batch size
            gradient_checkpoint_perceiver: If True, use gradient checkpointing in Perceiver
            gradient_checkpoint_rnn: If True, use gradient checkpointing in RNN
        
        Returns:
            Processed LTM of shape [B, K_ltm, D] (cross_attention) or [B, D] (adaln)
        """
        if ltm.shape[1] == 0:
            # Empty LTM sequence
            ltm = ltm.new_zeros((b, self.k_ltm, self.model_dim)) if self.ltm_projector is not None else ltm.new_zeros((b, self.model_dim))
            if self.ltm_type_emb is not None:
                ltm = ltm + self.ltm_type_emb
            return ltm

        # Process LTM through Perceiver and RNN
        ltm = self._downsample_ltm_tokens(ltm)
        context_latents = self._perceive_in_chunks(
            ltm,
            perceiver=self.ltm_perceiver,
            num_latents=self.ltm_num_latents,
            gradient_checkpoint_perceiver=gradient_checkpoint_perceiver,
        )

        ltm_proj = self.ltm_in_proj(context_latents)

        streaming_mode = self.use_streaming_cache and (not self.training)
        cached_state = self._context_cache if streaming_mode else None
        if streaming_mode:
            cached_state_data = cached_state.get("ltm_state", None) if cached_state is not None else None
            init_state = (
                self.recurrent_encoder.restore_state(cached_state_data)
                if cached_state_data is not None
                else self.recurrent_encoder.get_initial_state(ltm_proj.shape[0], ltm_proj.device, ltm_proj.dtype)
            )
        else:
            init_state = self.recurrent_encoder.get_initial_state(ltm_proj.shape[0], ltm_proj.device, ltm_proj.dtype)

        rnn_output, rnn_states = self.recurrent_encoder(
            ltm_proj,
            initial_state=init_state,
            gradient_checkpoint=gradient_checkpoint_rnn,
        )
        ltm_state_rnn = self._extract_ltm_state_vector(rnn_output)
        ltm_state_model = self.ltm_state_to_model_dim(ltm_state_rnn)

        # Expand to tokens for cross-attention or keep as state for adaln
        if self.ltm_projector is not None:
            ltm_out = self.ltm_projector(ltm_state_rnn)
            if self.ltm_type_emb is not None:
                ltm_out = ltm_out + self.ltm_type_emb
        else:
            ltm_out = ltm_state_model

        if streaming_mode:
            self._context_cache = self._build_context_cache(rnn_states)

        return ltm_out

    def _process_stm(
        self,
        stm: torch.Tensor,
        b: int,
        stm_flat: Optional[torch.Tensor] = None,
        stm_dataframe_indices: Optional[torch.Tensor] = None,
        gradient_checkpoint_perceiver: bool = False,
    ) -> torch.Tensor:
        """
        Process STM tokens through Perceiver and optional temporal embeddings.
        
        Handles empty sequences, optional pre-computed flattened STM, perception,
        temporal embeddings, and flattening to sequence form.
        
        Args:
            stm: STM tokens of shape [B, T_stm, L_stm, D]
            b: Batch size
            stm_flat: Optional pre-computed flattened STM of shape [B, N_stm, D]
            stm_dataframe_indices: Optional time indices for temporal embeddings of shape [B, T_stm]
            gradient_checkpoint_perceiver: If True, use gradient checkpointing in Perceiver
        
        Returns:
            Processed STM of shape [B, T_stm*N_stm, D]
        """
        if stm.shape[1] == 0:
            return stm.new_zeros((b, 0, self.model_dim))
        
        if not self.use_stm_perceiver:
            if stm_flat is not None:
                stm_out = stm_flat
            else:
                stm_out = rearrange(stm, "b t l d -> b (t l) d")
        else:
            stm_latents = self._perceive_in_chunks(
                stm,
                perceiver=self.stm_perceiver,
                num_latents=self.stm_num_latents,
                gradient_checkpoint_perceiver=gradient_checkpoint_perceiver,
            )
            
            # Generate temporal Fourier embeddings for STM
            if stm_dataframe_indices is not None:
                stm_temporal_emb = self._generate_stm_temporal_embeddings(stm_latents, stm_dataframe_indices)
            else:
                stm_temporal_emb = stm_latents.new_zeros_like(stm_latents)
            
            # Flatten STM: [B, T_stm, N_stm, D] -> [B, T_stm*N_stm, D]
            stm_out = rearrange(stm_latents + stm_temporal_emb, "b t n d -> b (t n) d")
        
        if self.stm_type_emb is not None:
            stm_out = stm_out + self.stm_type_emb
        
        return stm_out

    def forward(
        self,
        context: Dict[str, torch.Tensor],
        ltm_override: Optional[torch.Tensor] = None,
        gradient_checkpoint_perceiver: bool = False,
        gradient_checkpoint_rnn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Build unified memory representation from decoupled LTM and STM streams.
        
        Processes LTM and STM through separate Perceiver encoders and a recurrent encoder,
        returning the LTM state and processed STM tokens for use in downstream models.
        
        Args:
            context: Dictionary containing:
                - ltm_tokens [B, T_ltm, L_ltm, D]: Long-term memory tokens
                - stm_tokens [B, T_stm, L_stm, D]: Short-term memory tokens
                - stm_flat_tokens [B, N_stm, D] (optional): Pre-computed flattened STM
                - stm_dataframe_indices [B, T_stm] (optional): Time indices for STM temporal embeddings
            ltm_override: Optional precomputed LTM representation. If provided,
                LTM processing is skipped and this tensor is used directly.
            gradient_checkpoint_perceiver: If True, use gradient checkpointing in Perceiver
            gradient_checkpoint_rnn: If True, use gradient checkpointing in recurrent encoder
        
        Returns:
            Dictionary with:
            - ltm [B, K_ltm, D] or [B, D]: LTM representation (tokens for cross_attention, state for adaln)
            - stm [B, T_stm*N_stm, D]: Processed STM tokens
        
        Raises:
            KeyError: If ltm_tokens or stm_tokens not in context
            ValueError: If tensor shapes don't match model configuration
        """
        ltm = context["ltm_tokens"]
        stm = context["stm_tokens"]
        stm_flat = context.get("stm_flat_tokens", None)
        stm_dataframe_indices = context.get("stm_dataframe_indices", None)
        
        b = ltm.shape[0]

        # Process LTM and STM through separate helper functions
        if ltm_override is None:
            ltm = self._process_ltm(ltm, b, gradient_checkpoint_perceiver, gradient_checkpoint_rnn)
        else:
            ltm = ltm_override
        stm = self._process_stm(stm, b, stm_flat, stm_dataframe_indices, gradient_checkpoint_perceiver)

        return {
            "ltm": ltm,
            "stm": stm,
            "stm_time": context.get("stm_time", None),
            "stm_rope_pos": context.get("stm_rope_pos", None),
        }
