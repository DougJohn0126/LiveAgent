"""Unit tests for Attention classes.

This module contains comprehensive tests for:
- Attention: Basic attention with optional RoPE
- ModalityAwareRoPEAttention: Attention with modality-aware RoPE support

Tests include functional coverage, equivalence verification against manual
implementations, and edge case handling.
"""

import pytest
import torch
import torch.nn.functional as F

from models.components.attention import Attention, ModalityAwareRoPEAttention


def manual_attention_forward(module, q_input, k_input, v_input, attn_mask=None, key_padding_mask=None):
    """
    Manual implementation of attention forward pass (reference implementation).
    Uses the weights from the provided module but performs manual calculation.
    """
    B, N_q, D = q_input.shape
    _, N_k, _ = k_input.shape
    
    # Project Q, K, V
    q = module.q_proj(q_input)
    k = module.k_proj(k_input)
    v = module.v_proj(v_input)
    
    # Reshape to multi-head: [B, N, D] -> [B, num_heads, N, head_dim]
    q = q.reshape(B, N_q, module.num_heads, module.head_dim).transpose(1, 2)
    k = k.reshape(B, N_k, module.num_heads, module.head_dim).transpose(1, 2)
    v = v.reshape(B, N_k, module.num_heads, module.head_dim).transpose(1, 2)
    
    # Apply RoPE if enabled
    if hasattr(module, 'rope') and module.rope is not None:
        q = module.rope.rotate_queries_or_keys(q, seq_dim=-2)
        k = module.rope.rotate_queries_or_keys(k, seq_dim=-2)
    
    # Compute attention scores: [B, num_heads, N_q, N_k]
    attn_scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * module.scale
    
    # Apply attention mask
    if attn_mask is not None:
        # Handle different mask dimensions
        if attn_mask.dim() == 2:  # [N_q, N_k]
            mask_to_apply = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_q, N_k]
        elif attn_mask.dim() == 3:  # [B, N_q, N_k]
            mask_to_apply = attn_mask.unsqueeze(1)  # [B, 1, N_q, N_k]
        else:
            mask_to_apply = attn_mask

        # Convert boolean mask to additive mask
        if mask_to_apply.dtype == torch.bool:
             attn_scores = attn_scores.masked_fill(mask_to_apply, float("-inf"))
        else:
             attn_scores = attn_scores + mask_to_apply
    
    # Apply key padding mask
    if key_padding_mask is not None:
        # [B, N_k] -> [B, 1, 1, N_k]
        mask_to_apply = key_padding_mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(mask_to_apply, float("-inf"))
    
    # Compute attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Apply dropout to attention weights (only if training)
    if module.dropout is not None and module.training:
        attn_weights = module.dropout(attn_weights)
    
    # Apply attention to values: [B, num_heads, N_q, head_dim]
    out = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)
    
    # Reshape back: [B, num_heads, N_q, head_dim] -> [B, N_q, D]
    out = out.transpose(1, 2).reshape(B, N_q, D)
    
    # Output projection
    out = module.out_proj(out)
    
    return out


class TestAttention:
    """Test suite for unified Attention class."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for tests."""
        return {
            "embed_dim": 256,
            "num_heads": 8,
            "batch_size": 2,
            "seq_len_q": 10,
            "seq_len_k": 15,
        }
    
    def test_self_attention(self, config):
        """Test self-attention (Q, K, V from same input)."""
        attn = Attention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=0.1,
            use_rope=False,
        )
        attn.eval()
        
        x = torch.randn(config["batch_size"], config["seq_len_q"], config["embed_dim"])
        
        with torch.no_grad():
            output = attn(q_input=x, k_input=x, v_input=x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_cross_attention(self, config):
        """Test cross-attention (Q from one source, K/V from another)."""
        attn = Attention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=0.0,
            use_rope=False,
        )
        attn.eval()
        
        # Query from decoder
        q = torch.randn(config["batch_size"], config["seq_len_q"], config["embed_dim"])
        # Key/Value from encoder
        kv = torch.randn(config["batch_size"], config["seq_len_k"], config["embed_dim"])
        
        with torch.no_grad():
            output = attn(q_input=q, k_input=kv, v_input=kv)
        
        # Output shape matches query shape
        assert output.shape == (config["batch_size"], config["seq_len_q"], config["embed_dim"])
        assert not torch.isnan(output).any()
    
    def test_attention_with_rope(self, config):
        """Test attention with Rotary Position Embeddings."""
        attn = Attention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=0.0,
            use_rope=True,
        )
        attn.eval()
        
        x = torch.randn(config["batch_size"], config["seq_len_q"], config["embed_dim"])
        
        with torch.no_grad():
            output = attn(q_input=x, k_input=x, v_input=x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_attention_mask(self, config):
        """Test attention with mask."""
        attn = Attention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=0.0,
        )
        attn.eval()
        
        x = torch.randn(config["batch_size"], config["seq_len_q"], config["embed_dim"])
        
        # Causal mask: upper triangle masked
        attn_mask = torch.triu(
            torch.ones(config["seq_len_q"], config["seq_len_q"], dtype=torch.bool),
            diagonal=1
        )
        
        with torch.no_grad():
            output = attn(q_input=x, k_input=x, v_input=x, attn_mask=attn_mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_key_padding_mask(self, config):
        """Test attention with key padding mask."""
        attn = Attention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=0.0,
        )
        attn.eval()
        
        q = torch.randn(config["batch_size"], config["seq_len_q"], config["embed_dim"])
        kv = torch.randn(config["batch_size"], config["seq_len_k"], config["embed_dim"])
        
        # Mask out last 5 positions in keys
        key_padding_mask = torch.zeros(config["batch_size"], config["seq_len_k"], dtype=torch.bool)
        key_padding_mask[:, -5:] = True
        
        with torch.no_grad():
            output = attn(
                q_input=q,
                k_input=kv,
                v_input=kv,
                key_padding_mask=key_padding_mask
            )
        
        assert output.shape == (config["batch_size"], config["seq_len_q"], config["embed_dim"])
        assert not torch.isnan(output).any()
    
    def test_gradient_flow(self, config):
        """Test gradient flow through attention."""
        attn = Attention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=0.0,
        )
        
        x = torch.randn(
            config["batch_size"],
            config["seq_len_q"],
            config["embed_dim"],
            requires_grad=True
        )
        
        output = attn(q_input=x, k_input=x, v_input=x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check all parameters have gradients
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"
    
    def test_different_batch_sizes(self, config):
        """Test with different batch sizes."""
        attn = Attention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
        )
        attn.eval()
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, config["seq_len_q"], config["embed_dim"])
            
            with torch.no_grad():
                output = attn(q_input=x, k_input=x, v_input=x)
            
            assert output.shape == (batch_size, config["seq_len_q"], config["embed_dim"])
    
    def test_different_sequence_lengths(self, config):
        """Test with varying sequence lengths."""
        attn = Attention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
        )
        attn.eval()
        
        for seq_len in [5, 20, 50]:
            x = torch.randn(config["batch_size"], seq_len, config["embed_dim"])
            
            with torch.no_grad():
                output = attn(q_input=x, k_input=x, v_input=x)
            
            assert output.shape == (config["batch_size"], seq_len, config["embed_dim"])
    
    def test_head_dim_divisibility(self):
        """Test that embed_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            Attention(embed_dim=256, num_heads=7)  # 256 not divisible by 7
    
    def test_equivalence_no_mask(self, config):
        """Test SDPA matches manual attention without masks."""
        # Use eval mode to disable dropout for deterministic comparison
        attn = Attention(
            embed_dim=config["embed_dim"], 
            num_heads=config["num_heads"],
            dropout=0.0
        ).eval()
        
        q = torch.randn(config["batch_size"], config["seq_len_q"], config["embed_dim"])
        k = torch.randn(config["batch_size"], config["seq_len_k"], config["embed_dim"])
        v = torch.randn(config["batch_size"], config["seq_len_k"], config["embed_dim"])
        
        with torch.no_grad():
            expected = manual_attention_forward(attn, q, k, v)
            actual = attn(q, k, v)
            
        # Check closeness. SDPA might have small numerical differences.
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_equivalence_with_causal_mask(self, config):
        """Test SDPA matches manual attention with causal mask."""
        attn = Attention(
            embed_dim=config["embed_dim"], 
            num_heads=config["num_heads"],
            dropout=0.0
        ).eval()
        
        q = torch.randn(config["batch_size"], config["seq_len_q"], config["embed_dim"])
        
        # Causal mask
        mask = torch.triu(torch.ones(config["seq_len_q"], config["seq_len_q"], dtype=torch.bool), diagonal=1)
        
        with torch.no_grad():
            expected = manual_attention_forward(attn, q, q, q, attn_mask=mask)
            actual = attn(q, q, q, attn_mask=mask)
            
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_equivalence_with_padding_mask(self, config):
        """Test SDPA matches manual attention with key padding mask."""
        attn = Attention(
            embed_dim=config["embed_dim"], 
            num_heads=config["num_heads"],
            dropout=0.0
        ).eval()
        
        q = torch.randn(config["batch_size"], config["seq_len_q"], config["embed_dim"])
        kv = torch.randn(config["batch_size"], config["seq_len_k"], config["embed_dim"])
        
        # Padding mask (last 5 tokens masked)
        pad_mask = torch.zeros(config["batch_size"], config["seq_len_k"], dtype=torch.bool)
        pad_mask[:, -5:] = True
        
        with torch.no_grad():
            expected = manual_attention_forward(attn, q, kv, kv, key_padding_mask=pad_mask)
            actual = attn(q, kv, kv, key_padding_mask=pad_mask)
            
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


class TestModalityAwareRoPEAttention:
    """Test suite for ModalityAwareRoPEAttention class."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for tests."""
        return {
            "embed_dim": 256,
            "num_heads": 8,
            "batch_size": 2,
            "seq_len": 20,
        }
    
    def test_equivalence_no_rope(self, config):
        """Test ModalityAwareRoPEAttention matches manual attention (without RoPE applied).
        
        This isolates mask handling correctness by not passing RoPE positions.
        """
        torch.manual_seed(42)
        
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        head_dim = embed_dim // num_heads
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        
        attn = ModalityAwareRoPEAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
        )
        attn.eval()
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Create a causal mask (True = mask out)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1
        )
        
        # === Get SDPA output (no RoPE positions) ===
        with torch.no_grad():
            sdpa_output = attn(
                q_input=x, k_input=x, v_input=x,
                rope_pos_t=None,  # No RoPE
                attn_mask=causal_mask
            )
        
        # === Compute manually ===
        with torch.no_grad():
            q = attn.q_proj(x)
            k = attn.k_proj(x)
            v = attn.v_proj(x)
            
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask_expanded, float("-inf"))
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            manual_out = torch.matmul(attn_weights, v)
            manual_out = manual_out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
            manual_output = attn.out_proj(manual_out)
        
        assert torch.allclose(sdpa_output, manual_output, atol=1e-5), \
            f"ModalityAwareRoPEAttention differs from manual! Max diff: {(sdpa_output - manual_output).abs().max()}"
    
    def test_equivalence_with_rope(self, config):
        """Test ModalityAwareRoPEAttention with RoPE matches manual computation."""
        torch.manual_seed(42)
        
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        head_dim = embed_dim // num_heads
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        
        attn = ModalityAwareRoPEAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
        )
        attn.eval()
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Temporal positions (sequential)
        rope_pos_t = torch.arange(seq_len, dtype=torch.float).unsqueeze(0).expand(batch_size, -1)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1
        )
        
        # === Get SDPA output with RoPE ===
        with torch.no_grad():
            sdpa_output = attn(
                q_input=x, k_input=x, v_input=x,
                rope_pos_t=rope_pos_t,
                attn_mask=causal_mask
            )
        
        # === Compute manually with RoPE ===
        with torch.no_grad():
            q = attn.q_proj(x)
            k = attn.k_proj(x)
            v = attn.v_proj(x)
            
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Apply RoPE using the same method
            q_rot, k_rot = attn.apply_unified_rope(q, k, rope_pos_t, None, None)
            
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * scale
            
            mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask_expanded, float("-inf"))
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            manual_out = torch.matmul(attn_weights, v)
            manual_out = manual_out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
            manual_output = attn.out_proj(manual_out)
        
        assert torch.allclose(sdpa_output, manual_output, atol=1e-5), \
            f"ModalityAwareRoPEAttention with RoPE differs! Max diff: {(sdpa_output - manual_output).abs().max()}"
    
    def test_rope_changes_output(self, config):
        """Test that applying RoPE changes the attention output."""
        torch.manual_seed(42)
        
        attn = ModalityAwareRoPEAttention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=0.0,
        )
        attn.eval()
        
        x = torch.randn(config["batch_size"], config["seq_len"], config["embed_dim"])
        rope_pos_t = torch.arange(config["seq_len"], dtype=torch.float).unsqueeze(0).expand(config["batch_size"], -1)
        
        with torch.no_grad():
            out_no_rope = attn(q_input=x, k_input=x, v_input=x, rope_pos_t=None)
            out_with_rope = attn(q_input=x, k_input=x, v_input=x, rope_pos_t=rope_pos_t)
        
        # Outputs should differ when RoPE is applied
        diff = (out_no_rope - out_with_rope).abs().mean()
        assert diff > 0.01, f"RoPE should change output, but diff is only {diff}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-p", "no:warnings"]))
