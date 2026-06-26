from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from einops import rearrange, repeat

# Import unified Attention class
from models.components.attention import Attention

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)



# main class

class PerceiverAttention(nn.Module):
    """Adapter to use unified Attention with Perceiver's interface."""
    
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, use_weightnorm=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        # Use unified Attention
        self.attn = Attention(
            embed_dim=inner_dim,
            num_heads=heads,
            dropout=0.0,
            use_rope=False,
            use_weightnorm=use_weightnorm,
        )
        
        # Input projections to match expected dimensions
        self.q_in = nn.Linear(query_dim, inner_dim, bias=False)
        self.kv_in = nn.Linear(context_dim, inner_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(inner_dim, query_dim)
    
    def forward(self, x, context=None):
        """Forward with Perceiver-style interface."""
        context = default(context, x)
        
        # Project inputs
        q = self.q_in(x)
        k = self.kv_in(context)
        v = k  # In Perceiver, K and V come from same input
        
        out = self.attn(q_input=q, k_input=k, v_input=v, block_mask=None)
        
        # Project output
        return self.out_proj(out)



class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        use_decoder = False,
        seq_dropout_prob = 0.,
        use_weightnorm = False
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.use_decoder = use_decoder

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, PerceiverAttention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head, use_weightnorm=use_weightnorm), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim, PerceiverAttention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, use_weightnorm=use_weightnorm))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # Only instantiate decoder if queries are expected to be provided
        if use_decoder:
            self.decoder_cross_attn = PreNorm(queries_dim, PerceiverAttention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head, use_weightnorm=use_weightnorm), context_dim = latent_dim)
            self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        else:
            self.decoder_cross_attn = None
            self.decoder_ff = None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        data,
        queries = None,
        gradient_checkpoint: bool = False,
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b = b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # cross attention only happens once for Perceiver IO

        if gradient_checkpoint and torch.is_grad_enabled():
            def _cross_attn(inp):
                return cross_attn(inp, context=data)

            x = checkpoint(_cross_attn, x, use_reentrant=False) + x
            x = checkpoint(cross_ff, x, use_reentrant=False) + x
        else:
            x = cross_attn(x, context = data) + x
            x = cross_ff(x) + x

        # layers

        for self_attn, self_ff in self.layers:
            if gradient_checkpoint and torch.is_grad_enabled():
                x = checkpoint(self_attn, x, use_reentrant=False) + x
                x = checkpoint(self_ff, x, use_reentrant=False) + x
            else:
                x = self_attn(x) + x
                x = self_ff(x) + x

        if not exists(queries):
            return x

        # Use decoder only if enabled (queries provided and decoder was instantiated)
        if not self.use_decoder or self.decoder_cross_attn is None:
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        
        if gradient_checkpoint and torch.is_grad_enabled():
            def _decoder_cross_attn(inp):
                return self.decoder_cross_attn(inp, context=x)

            latents = checkpoint(_decoder_cross_attn, queries, use_reentrant=False)
        else:
            latents = self.decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            if gradient_checkpoint and torch.is_grad_enabled():
                latents = latents + checkpoint(self.decoder_ff, latents, use_reentrant=False)
            else:
                latents = latents + self.decoder_ff(latents)

        # final linear out

        return self.to_logits(latents)

# Perceiver LM example

class PerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIO(
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )

    def forward(
        self,
        x,
        mask = None
    ):
        del mask
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        logits = self.perceiver_io(x, queries = x)
        return logits