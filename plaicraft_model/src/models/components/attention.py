from typing import Optional, Sequence, Tuple
import logging
import os
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask, flex_attention
 
from models.components.positional_encoding import RotaryEmbedding
from models.components.weightnorm_modules import MPConv, normalize
 
logger = logging.getLogger(__name__)
 
flex_attention = torch.compile(flex_attention, fullgraph=False)
 
# When PLAICRAFT_SDPA_ATTN=1, the masked attention path materializes the block
# mask into a dense boolean [1,1,Q,KV] tensor ONCE per topology and runs through
# F.scaled_dot_product_attention (Ampere mem-efficient kernel) instead of the
# torch.compile'd flex_attention, which can generate a pathologically slow kernel
# on consumer Ampere (e.g. RTX 3060). Same math, same weights -- a runtime kernel
# swap only. Falls back to flex_attention automatically if the dense build fails.
_SDPA_ATTN = os.environ.get("PLAICRAFT_SDPA_ATTN", "0").lower() in ("1", "true", "yes")
_DENSE_MASK_CACHE: dict = {}
_PATH_ANNOUNCED = False
 
 
def _announce_path(name: str) -> None:
    global _PATH_ANNOUNCED
    if not _PATH_ANNOUNCED:
        _PATH_ANNOUNCED = True
        print(f"[attention] masked path = {name}", flush=True)
 
 
def _dense_mask_from_blockmask(block_mask: BlockMask, n_q: int, n_kv: int, device) -> Optional[torch.Tensor]:
    """Materialize a flex BlockMask into a dense boolean [1,1,n_q,n_kv] mask
    (True = attend), cached by (n_q, n_kv, device). The dataframe_level mask_mod
    ignores batch/head, so a single [1,1,Q,KV] mask broadcasts across both.
    Returns None on any failure so the caller can fall back to flex_attention."""
    key = (int(n_q), int(n_kv), str(device))
    cached = _DENSE_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from torch.nn.attention.flex_attention import create_mask
        mask = create_mask(block_mask.mask_mod, 1, 1, int(n_q), int(n_kv), device=device)
        mask = mask.to(torch.bool)
        _DENSE_MASK_CACHE[key] = mask
        return mask
    except Exception as exc:  # API drift, missing mask_mod, etc. -> caller falls back.
        logger.warning("SDPA dense-mask build failed (%s); using flex_attention.", exc)
        return None
 
class Attention(nn.Module):
    """Multi-head attention with optional RoPE and BlockMask-based flex attention."""
 
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        use_weightnorm: bool = False,
        rope_theta: float = 10000.0,
        rope_max_spatial_freq: float = 1.0,
        rope_mrope_section: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
 
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.use_weightnorm = use_weightnorm
 
        self.q_proj = nn.Linear(embed_dim, embed_dim) if not use_weightnorm else MPConv(embed_dim, embed_dim, kernel=[]) # The weight param stored inside is a 2D tensor.
        self.k_proj = nn.Linear(embed_dim, embed_dim) if not use_weightnorm else MPConv(embed_dim, embed_dim, kernel=[])
        self.v_proj = nn.Linear(embed_dim, embed_dim) if not use_weightnorm else MPConv(embed_dim, embed_dim, kernel=[])
        self.out_proj = nn.Linear(embed_dim, embed_dim) if not use_weightnorm else MPConv(embed_dim, embed_dim, kernel=[])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.rope = (
            RotaryEmbedding(
                dim=self.head_dim,
                theta=rope_theta,
                max_spatial_freq=rope_max_spatial_freq,
                mrope_section=rope_mrope_section,
            )
            if use_rope
            else None
        )
 
    def forward(
        self,
        q_input: torch.Tensor,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
        block_mask: Optional[BlockMask] = None,
        q_pos: Optional[torch.Tensor] = None,
        k_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
 
        q = self.q_proj(q_input) # BND -> BND.
        k = self.k_proj(k_input)
        v = self.v_proj(v_input)
 
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads) # BND -> BhNd.
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
 
        if self.use_weightnorm:
            # Pixel normalization. Normalize along the feature dimension (RMSNorm per token per head).
            q = normalize(q, dim=-1) # BhNd -> BhNd, where norm tensor is BhN1.
            k = normalize(k, dim=-1)
            v = normalize(v, dim=-1)
 
        if self.rope is not None:
            # Use provided metric coordinates if available, else fallback to sequence index RoPE.
            if q_pos is not None and k_pos is not None:
                q = self.rope.apply_multimodal_rotary_pos_emb(
                    q,
                    q_pos,
                    seq_dim=-2,
                )
                k = self.rope.apply_multimodal_rotary_pos_emb(
                    k,
                    k_pos,
                    seq_dim=-2,
                )
            else:
                q = self.rope.rotate_queries_or_keys(q, seq_dim=-2)
                k = self.rope.rotate_queries_or_keys(k, seq_dim=-2)
 
        if block_mask is None:
            # Natively invokes pre-compiled C++ FlashAttention-2 / Mem-Efficient Attention
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(
                    q, k, v, 
                    dropout_p=self.dropout_p if self.training else 0.0, 
                    scale=self.scale # scale is sqrt(d).
                )
        else:
            dense = _dense_mask_from_blockmask(block_mask, q.shape[-2], k.shape[-2], q.device) if _SDPA_ATTN else None
            if dense is not None:
                _announce_path("SDPA dense-mask")
                # Same masked softmax as flex, but on the mem-efficient SDPA kernel.
                # Let PyTorch pick the backend that supports a boolean attn_mask
                # (mem-efficient on Ampere; math as a safe fallback).
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=dense,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    scale=self.scale,
                )
            else:
                _announce_path("flex_attention")
                # flex_attention execution for complex BlockMasks (requires compilation for speed)
                out = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
                if self.training and self.dropout is not None:
                    out = self.dropout(out)
 
        out = rearrange(out, "b h n d -> b n (h d)") # BhNd -> BND.
        return self.out_proj(out)