"""xLSTM recurrent encoder implementation."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.components.recurrent_encoders.base_recurrent_encoder import BaseRecurrentEncoder
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)


class _mLSTMKernelAdapter(nn.Module):
    """
    Adapter that matches xlstm's backend_fn signature:
      backend_fn(queries=..., keys=..., values=..., igate_preact=..., fgate_preact=..., ...)
    and routes it into mlstm-kernels' mLSTMBackend (TFLA chunkwise Triton kernel).
    """

    def __init__(
        self,
        chunk_size: int = 128,
        chunkwise_kernel: str = "chunkwise--triton_xl_chunk",
        sequence_kernel: str = "native_sequence__triton",
        step_kernel: str = "triton",
        mode: str = "train_with_padding",
    ):
        super().__init__()

        try:
            from mlstm_kernels.torch.backend_module import mLSTMBackend, mLSTMBackendConfig
        except Exception as e:
            raise RuntimeError(
                "mlstm-kernels is required for the fast mLSTM backend, but it could not be imported. "
                "Install it (and ensure it matches your CUDA/PyTorch build), then retry."
            ) from e

        # Train-time safety: pad to multiples of chunk_size so arbitrary S works.
        # return_last_states must be False in train_with_padding.
        cfg = mLSTMBackendConfig(
            chunkwise_kernel=chunkwise_kernel,
            sequence_kernel=sequence_kernel,
            step_kernel=step_kernel,
            chunk_size=chunk_size,
            return_last_states=False,
            mode=mode,
        )
        self.backend = mLSTMBackend(cfg)

    def forward(
        self,
        *,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # xlstm provides gates as (B, NH, S, 1); mlstm-kernels expects (B, NH, S)
        if igate_preact.dim() != 4 or fgate_preact.dim() != 4:
            raise RuntimeError(
                f"Expected igate_preact/fgate_preact to be 4D (B, NH, S, 1), got "
                f"{igate_preact.shape} and {fgate_preact.shape}."
            )

        i = igate_preact.squeeze(-1)
        f = fgate_preact.squeeze(-1)

        out = self.backend(q=queries, k=keys, v=values, i=i, f=f, return_last_states=False)

        # mLSTMBackend may return (H, (c,n,m)) depending on config/callsite; normalize to H.
        if isinstance(out, tuple):
            out = out[0]

        if not torch.is_tensor(out) or out.dim() != 4:
            raise RuntimeError(
                f"mLSTM backend returned unexpected output. Expected 4D tensor (B, NH, S, DH), got: "
                f"{type(out)} with shape {getattr(out, 'shape', None)}."
            )

        return out


class xLSTMEncoder(BaseRecurrentEncoder):
    """
    Wrapper around xLSTM for use as a recurrent encoder.

    Uses the official xLSTM implementation from the xlstm package.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 4,
        num_blocks: int = 6,
        dropout: float = 0.0,
        context_length: int = 8192,
        conv1d_kernel_size: int = 4,
        qkv_proj_blocksize: int = 4,
        slstm_at: Optional[Tuple[int, ...]] = (),
        mlstm_chunk_size: int = 64,
        chunkwise_kernel: str = "chunkwise--triton_xl_chunk",
        sequence_kernel: str = "native_sequence__triton",
        step_kernel: str = "triton",
        backend_mode: str = "train_with_padding",
    ):
        """
        Initialize xLSTM encoder using xLSTMBlockStack.

        Args:
            embedding_dim: Dimension of the embeddings (feature_dim)
            num_heads: Number of attention heads
            num_blocks: Number of xLSTM blocks
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # Configure mLSTM blocks for the stack
        mlstm_block_config = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=qkv_proj_blocksize,
                num_heads=num_heads,
            ),
        )

        # Configure xLSTM Block Stack
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block_config,
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            dropout=dropout,
            slstm_at=list(slstm_at),
        )

        # Initialize the xLSTM Block Stack
        self.xlstm = xLSTMBlockStack(xlstm_config)

        # Install the fast kernel backend for ALL mLSTMCell instances inside this stack.
        # If this patch is not applied, xlstm defaults to parallel_stabilized_simple (quadratic, OOM-prone).
        self._mlstm_kernel_adapter = _mLSTMKernelAdapter(
            chunk_size=mlstm_chunk_size,
            chunkwise_kernel=chunkwise_kernel,
            sequence_kernel=sequence_kernel,
            step_kernel=step_kernel,
            mode=backend_mode,
        )
        self._patch_mlstm_cells_or_raise()

    def _patch_mlstm_cells_or_raise(self) -> None:
        patched = 0
        for m in self.xlstm.modules():
            if m.__class__.__name__ == "mLSTMCell":
                m.backend_fn = self._mlstm_kernel_adapter
                patched += 1
        if patched == 0:
            raise RuntimeError(
                "Failed to patch xLSTM mLSTMCell backend_fn (no mLSTMCell modules found). "
                "The xlstm package internals may have changed; update the patch logic accordingly."
            )

    def _clone_state_tree(self, state: Any, detach: bool = True) -> Any:
        if torch.is_tensor(state):
            return state.detach().clone() if detach else state.clone()
        if isinstance(state, dict):
            return {key: self._clone_state_tree(value, detach=detach) for key, value in state.items()}
        if isinstance(state, tuple):
            return tuple(self._clone_state_tree(value, detach=detach) for value in state)
        if isinstance(state, list):
            return [self._clone_state_tree(value, detach=detach) for value in state]
        return state

    def _normalize_initial_state(self, initial_state: Optional[Any]) -> Dict[str, Dict[str, Tuple[torch.Tensor, ...]]]:
        if initial_state is None:
            return {}
        if isinstance(initial_state, dict):
            return self._clone_state_tree(initial_state, detach=False)
        raise TypeError(
            f"Unsupported xLSTM initial_state type: {type(initial_state)}. "
            f"Expected dict or None."
        )

    def get_initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Get the initial recurrent state for xLSTM.

        For xLSTM we use an empty dict, which signals xLSTMBlockStack.step() to
        initialize internal recurrent state lazily.
        """
        return {}

    def _checkpoint_full_forward(self, x: torch.Tensor) -> torch.Tensor:
        def _run(inp: torch.Tensor) -> torch.Tensor:
            return self.xlstm(inp)

        try:
            return checkpoint(_run, x, use_reentrant=False)
        except TypeError:
            return checkpoint(_run, x)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[Any] = None,
        gradient_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, Tuple[torch.Tensor, ...]]]]:
        """
        Forward pass through xLSTM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            initial_state: Optional xLSTM state dict from previous chunk
            gradient_checkpoint: Whether to use gradient checkpointing

        Returns:
            output: Output tensor of shape (batch_size, seq_len, embedding_dim)
            final_state: Final xLSTM state dict for streaming inference
        """
        state = self._normalize_initial_state(initial_state)

        if torch.is_grad_enabled():
            if state:
                raise ValueError(
                    "xLSTM streaming state with gradients is not supported due to inplace state updates in xlstm.step(). "
                    "Run streaming in inference/no_grad mode, or train with fresh state."
                )

            if not gradient_checkpoint:
                output = self.xlstm(x)
                return output, {}

            output = self._checkpoint_full_forward(x)
            return output, {}

        # Inference / no_grad streaming path
        _, seq_len, _ = x.shape
        outputs = []
        for t in range(seq_len):
            y_t, state = self.xlstm.step(x[:, t : t + 1, :], state=state)
            outputs.append(y_t)

        output = torch.cat(outputs, dim=1) if outputs else x[:, :0, :]
        return output, state

    def cache_state(self, state: Any):
        """Prepare the recurrent state for caching."""
        return self._clone_state_tree(state, detach=True)

    def restore_state(self, cached_state: Any):
        """Restore a cached recurrent state."""
        return self._clone_state_tree(cached_state, detach=False)
