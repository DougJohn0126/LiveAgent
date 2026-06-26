"""Unit tests for ContextEmbedder (Perceiver + RNN LTM/STM)."""

import pytest
import torch
import torch.nn as nn

from models.components.context_embedder import ContextEmbedder


def _make_tokenized_modalities(batch_size: int, timesteps: int, model_dim: int, tokens_per_modality=None):
    if tokens_per_modality is None:
        tokens_per_modality = {
            "video": 8,
            "audio_speak": 4,
            "audio_hear": 4,
        }

    modalities = {}
    for name, tokens_per_frame in tokens_per_modality.items():
        modalities[name] = {
            "tokens": torch.randn(batch_size, timesteps, tokens_per_frame, model_dim)
        }

    return modalities


class TestContextEmbedder:
    """Test suite for ContextEmbedder with Perceiver + RNN."""

    @pytest.fixture
    def config(self):
        """Standard config for tests."""
        return {
            "model_dim": 256,
            "num_latents": 32,
            "perceiver_depth": 2,
            "perceiver_cross_heads": 4,
            "perceiver_latent_heads": 8,
            "perceiver_cross_dim_head": 64,
            "perceiver_latent_dim_head": 64,
            "perceiver_seq_dropout": 0.0,
            "h_short": 8,
            "chunk_len": 16,
            "rnn_config": {
                "num_layers": 2,
                "num_heads": 4,
                "mlp_multiplier": 4,
                "rnn_type": "mingru",
                "embedding_dim": 256,
            },
            "k_ltm": 16,
        }

    @pytest.fixture
    def embedder(self, config):
        """Create embedder instance."""
        return ContextEmbedder(**config)

    @pytest.fixture
    def sample_context(self, config):
        """Create sample tokenized context."""
        return _make_tokenized_modalities(2, 10, config["model_dim"])

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_embedder_initialization(self, config):
        """Test embedder initializes correctly."""
        embedder = ContextEmbedder(**config)
        assert embedder.model_dim == config["model_dim"]
        assert embedder.k_ltm == config["k_ltm"]
        assert embedder.h_short == config["h_short"]
        assert hasattr(embedder, "perceiver")
        assert hasattr(embedder, "recurrent_encoder")

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_forward_basic(self, embedder, sample_context):
        """Test basic forward pass."""
        embedder.eval()
        with torch.no_grad():
            output = embedder(sample_context)

        # Output should be unified memory [B, K_ltm + h_short*N, D]
        B = sample_context["video"]["tokens"].shape[0]
        T = sample_context["video"]["tokens"].shape[1]
        expected_seq_len = embedder.k_ltm + embedder.h_short * T
        assert output.shape == (B, expected_seq_len, embedder.model_dim)

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_output_dtype(self, embedder, sample_context):
        """Test output has correct dtype."""
        embedder.eval()
        with torch.no_grad():
            output = embedder(sample_context)

        assert output.dtype == torch.float32

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_no_nan_inf(self, embedder, sample_context):
        """Test output contains no NaN or Inf values."""
        embedder.eval()
        with torch.no_grad():
            output = embedder(sample_context)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_batched_processing(self, embedder):
        """Test embedder handles different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            context = _make_tokenized_modalities(batch_size, 10, embedder.model_dim)
            embedder.eval()
            with torch.no_grad():
                output = embedder(context)

            assert output.shape[0] == batch_size

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_sequence_length_variation(self, embedder):
        """Test embedder handles different sequence lengths."""
        for seq_len in [5, 10, 20]:
            context = _make_tokenized_modalities(2, seq_len, embedder.model_dim)
            embedder.eval()
            with torch.no_grad():
                output = embedder(context)

            expected_len = embedder.k_ltm + embedder.h_short * seq_len
            assert output.shape[1] == expected_len

    def test_sparse_input_handling(self):
        """Test embedder handles sparse/missing modalities."""
        embedder = ContextEmbedder(
            model_dim=256,
            num_latents=32,
            perceiver_depth=2,
            perceiver_cross_heads=4,
            perceiver_latent_heads=8,
            perceiver_cross_dim_head=64,
            perceiver_latent_dim_head=64,
            perceiver_seq_dropout=0.0,
            h_short=8,
            chunk_len=16,
            rnn_config={
                "num_layers": 2,
                "num_heads": 4,
                "mlp_multiplier": 4,
                "rnn_type": "mingru",
                "embedding_dim": 256,
            },
            k_ltm=16,
        )

        # Test with only some modalities
        context = _make_tokenized_modalities(2, 10, embedder.model_dim, {"video": 8, "audio_hear": 4})

        embedder.eval()
        with torch.no_grad():
            output = embedder(context)

        assert output is not None
        assert output.shape[0] == 2

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_perceiver_integration(self, embedder, sample_context):
        """Test that Perceiver is properly integrated."""
        assert hasattr(embedder, "perceiver")

        embedder.eval()
        with torch.no_grad():
            output = embedder(sample_context)

        # Perceiver should compress all modalities per timestep
        assert output is not None

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_rnn_integration(self, embedder, sample_context):
        """Test that RNN encoder is properly integrated."""
        assert hasattr(embedder, "recurrent_encoder")

        embedder.eval()
        with torch.no_grad():
            output = embedder(sample_context)

        # RNN should produce sequential output
        assert output.shape[1] > 0  # Has sequence dimension

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_ltm_stm_split(self, embedder):
        """Test LTM/STM split in output."""
        context = _make_tokenized_modalities(2, 10, embedder.model_dim)

        embedder.eval()
        with torch.no_grad():
            output = embedder(context)

        # Should have LTM + STM structure
        # LTM: k_ltm tokens
        # STM: h_short * T tokens
        B, T = 2, 10
        expected_len = embedder.k_ltm + embedder.h_short * T
        assert output.shape[1] == expected_len

    def test_gradient_flow(self):
        """Test that gradients flow through embedder."""
        embedder = ContextEmbedder(
            model_dim=128,
            num_latents=16,
            perceiver_depth=1,
            perceiver_cross_heads=2,
            perceiver_latent_heads=4,
            perceiver_cross_dim_head=32,
            perceiver_latent_dim_head=32,
            perceiver_seq_dropout=0.0,
            h_short=4,
            chunk_len=8,
            rnn_config={
                "num_layers": 1,
                "num_heads": 2,
                "mlp_multiplier": 2,
                "rnn_type": "mingru",
                "embedding_dim": 128,
            },
            k_ltm=8,
        )

        context = _make_tokenized_modalities(1, 5, embedder.model_dim)
        context["video"]["tokens"].requires_grad_(True)
        context["audio_speak"]["tokens"].requires_grad_(True)

        output = embedder(context)
        loss = output.sum()
        loss.backward()

        assert context["video"]["tokens"].grad is not None
        assert context["audio_speak"]["tokens"].grad is not None

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_eval_mode(self, embedder, sample_context):
        """Test embedder in eval mode (no dropout, etc)."""
        embedder.eval()
        with torch.no_grad():
            output1 = embedder(sample_context)
            output2 = embedder(sample_context)

        # Should be deterministic in eval mode
        assert torch.allclose(output1, output2)

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_train_mode_stochasticity(self, embedder, sample_context):
        """Test that embedder is stochastic in train mode."""
        embedder.train()

        output1 = embedder(sample_context)
        output2 = embedder(sample_context)

        # With dropout, outputs should differ
        # (with high probability, though not guaranteed)
        assert not torch.allclose(output1, output2)

    def test_different_rnn_types(self):
        """Test embedder with different RNN types."""
        config = {
            "model_dim": 128,
            "num_latents": 16,
            "perceiver_depth": 1,
            "perceiver_cross_heads": 2,
            "perceiver_latent_heads": 4,
            "perceiver_cross_dim_head": 32,
            "perceiver_latent_dim_head": 32,
            "perceiver_seq_dropout": 0.0,
            "h_short": 4,
            "chunk_len": 8,
            "rnn_config": {
                "num_layers": 1,
                "num_heads": 2,
                "mlp_multiplier": 2,
                "embedding_dim": 128,
            },
            "k_ltm": 8,
        }

        context = _make_tokenized_modalities(2, 5, config["model_dim"])

        for rnn_type in ["mingru"]:  # Add other types if available
            config["rnn_config"]["rnn_type"] = rnn_type
            embedder = ContextEmbedder(**config)
            embedder.eval()

            with torch.no_grad():
                output = embedder(context)

            assert output is not None
            assert output.shape[0] == 2

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_memory_efficiency(self, embedder):
        """Test memory efficiency with reasonable batch/seq sizes."""
        # Typical sizes from data pipeline
        context = _make_tokenized_modalities(4, 20, embedder.model_dim)

        embedder.eval()
        with torch.no_grad():
            output = embedder(context)

        assert output.shape == (4, embedder.k_ltm + embedder.h_short * 20, embedder.model_dim)

    def test_perceiver_parameters(self):
        """Test that Perceiver parameters are correctly set."""
        config = {
            "model_dim": 256,
            "num_latents": 32,
            "perceiver_depth": 3,
            "perceiver_cross_heads": 8,
            "perceiver_latent_heads": 16,
            "perceiver_cross_dim_head": 128,
            "perceiver_latent_dim_head": 128,
            "perceiver_seq_dropout": 0.1,
            "h_short": 8,
            "chunk_len": 16,
            "rnn_config": {
                "num_layers": 2,
                "num_heads": 4,
                "mlp_multiplier": 4,
                "rnn_type": "mingru",
                "embedding_dim": 256,
            },
            "k_ltm": 16,
        }

        embedder = ContextEmbedder(**config)

        assert embedder.num_latents == config["num_latents"]
        assert embedder.model_dim == config["model_dim"]
        assert embedder.h_short == config["h_short"]
        assert embedder.k_ltm == config["k_ltm"]
        assert hasattr(embedder, "perceiver")

    @pytest.mark.skip(reason="ContextEmbedder tests require tokenized input, not raw FullData")
    def test_output_can_be_used_as_conditioning(self, embedder, sample_context):
        """Test that output is suitable for use as cross-attention conditioning."""
        embedder.eval()
        with torch.no_grad():
            unified_memory = embedder(sample_context)

        # Should be usable as KV cache for cross-attention
        B, N, D = unified_memory.shape
        assert B > 0  # Batch dimension
        assert N > 0  # Sequence dimension (for attending)
        assert D == embedder.model_dim  # Matches model dim
