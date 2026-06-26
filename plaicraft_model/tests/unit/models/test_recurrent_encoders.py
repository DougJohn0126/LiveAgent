"""Unit tests for Recurrent Encoders."""

import pytest
import torch

from models.components.recurrent_encoders.min_gru import MinGRUCell, MinGRUEncoder

try:
    from models.components.recurrent_encoders.xlstm import xLSTMEncoder
except ImportError:
    xLSTMEncoder = None


class TestMinGRUCell:
    """Test suite for MinGRUCell."""

    @pytest.fixture
    def mingru_cell(self):
        """Create MinGRUCell."""
        return MinGRUCell(units=256, input_shape=256)

    def test_mingru_cell_initialization(self, mingru_cell):
        """Test MinGRUCell initializes correctly."""
        assert mingru_cell.units == 256
        assert mingru_cell.input_shape == 256

    def test_mingru_cell_forward(self, mingru_cell):
        """Test MinGRUCell forward pass."""
        x = torch.randn(2, 1, 256)
        h = torch.randn(2, 1, 256)

        h_new = mingru_cell(x, h)

        assert h_new.shape == (2, 1, 256)
        assert not torch.isnan(h_new).any()
        assert not torch.isinf(h_new).any()


class TestMinGRUEncoder:
    """Test suite for MinGRUEncoder."""

    @pytest.fixture
    def encoder_config(self):
        """Standard encoder config."""
        return {
            "embedding_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_multiplier": 4,
        }

    @pytest.fixture
    def encoder(self, encoder_config):
        """Create minGRU encoder."""
        return MinGRUEncoder(**encoder_config)

    def test_encoder_initialization(self, encoder, encoder_config):
        """Test encoder initializes correctly."""
        assert encoder.embedding_dim == encoder_config["embedding_dim"]
        assert encoder.num_layers == encoder_config["num_layers"]

    def test_encoder_forward_sequence(self, encoder):
        """Test encoder forward pass on sequence."""
        seq_len = 10
        batch_size = 2
        x = torch.randn(batch_size, seq_len, 256)

        init_state = encoder.get_initial_state(batch_size, x.device, x.dtype)
        output, rnn_states = encoder(x, initial_state=init_state)

        assert output.shape == (batch_size, seq_len, 256)
        assert rnn_states.shape[0] == encoder.num_layers
        assert not torch.isnan(output).any()

    def test_encoder_preserves_sequence_length(self, encoder):
        """Test that encoder preserves sequence length."""
        for seq_len in [5, 10, 20, 50]:
            x = torch.randn(2, seq_len, 256)
            init_state = encoder.get_initial_state(2, x.device, x.dtype)
            output, _ = encoder(x, initial_state=init_state)
            assert output.shape[1] == seq_len

    def test_encoder_with_different_batch_sizes(self, encoder):
        """Test encoder with various batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 10, 256)
            init_state = encoder.get_initial_state(batch_size, x.device, x.dtype)
            output, _ = encoder(x, initial_state=init_state)
            assert output.shape == (batch_size, 10, 256)

    def test_encoder_gradient_flow(self, encoder):
        """Test gradients flow through encoder."""
        x = torch.randn(2, 10, 256, requires_grad=True)
        init_state = encoder.get_initial_state(2, x.device, x.dtype)

        output, _ = encoder(x, initial_state=init_state)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_encoder_eval_mode(self, encoder):
        """Test encoder in eval mode (deterministic)."""
        encoder.eval()
        x = torch.randn(2, 10, 256)
        init_state = encoder.get_initial_state(2, x.device, x.dtype)

        with torch.no_grad():
            output1, _ = encoder(x, initial_state=init_state)
            output2, _ = encoder(x, initial_state=init_state)

        assert torch.allclose(output1, output2)


@pytest.mark.skipif(xLSTMEncoder is None, reason="xLSTM dependencies are not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="xLSTM requires CUDA GPU")
class TestxLSTMEncoder:
    """Test suite for xLSTMEncoder."""

    @pytest.fixture
    def encoder(self):
        return xLSTMEncoder(
            embedding_dim=64,
            num_heads=4,
            num_blocks=2,
            dropout=0.0,
            context_length=256,
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
        ).eval().cuda()

    def test_stateful_streaming_matches_chunked_processing(self, encoder):
        batch_size = 2
        seq_len = 7
        x = torch.randn(batch_size, seq_len, encoder.embedding_dim, device="cuda")

        with torch.no_grad():
            init_state = encoder.get_initial_state(batch_size, x.device, x.dtype)
            out_full, final_state_full = encoder(x, initial_state=init_state)

            x_first = x[:, :3, :]
            x_second = x[:, 3:, :]
            out_first, state_after_first = encoder(x_first, initial_state=init_state)
            out_second, final_state_stream = encoder(x_second, initial_state=state_after_first)
            out_stream = torch.cat([out_first, out_second], dim=1)

        assert out_full.shape == out_stream.shape
        assert torch.allclose(out_full, out_stream, atol=1e-5, rtol=1e-5)
        assert isinstance(final_state_full, dict)
        assert isinstance(final_state_stream, dict)

    def test_cache_and_restore_clone_nested_state_empty(self, encoder):
        x = torch.randn(2, 4, encoder.embedding_dim, device="cuda")
        init_state = encoder.get_initial_state(2, x.device, x.dtype)
        _, state = encoder(x, initial_state=init_state)

        cached = encoder.cache_state(state)
        restored = encoder.restore_state(cached)

        assert isinstance(cached, dict)
        assert isinstance(restored, dict)
        assert len(cached) == 0
        assert len(restored) == 0

    def test_cache_and_restore_clone_nested_state_with_slstm(self):
        encoder_slstm = xLSTMEncoder(
            embedding_dim=64,
            num_heads=4,
            num_blocks=2,
            dropout=0.0,
            context_length=256,
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            slstm_at=(0,)
        ).eval().cuda()

        x = torch.randn(2, 4, encoder_slstm.embedding_dim, device="cuda")
        init_state = encoder_slstm.get_initial_state(2, x.device, x.dtype)
        _, state = encoder_slstm(x, initial_state=init_state)

        cached = encoder_slstm.cache_state(state)
        restored = encoder_slstm.restore_state(cached)

        assert isinstance(cached, dict)
        assert isinstance(restored, dict)
        assert set(cached.keys()) == set(restored.keys())
        assert len(cached) > 0  # Should be populated since we have an sLSTM

        block_key = next(iter(cached.keys()))
        cached_block = cached[block_key]
        restored_block = restored[block_key]
        assert set(cached_block.keys()) == set(restored_block.keys())

        for state_key in cached_block.keys():
            cached_tensors = cached_block[state_key]
            restored_tensors = restored_block[state_key]
            assert len(cached_tensors) == len(restored_tensors)
            for cached_tensor, restored_tensor in zip(cached_tensors, restored_tensors):
                assert cached_tensor is not restored_tensor
                assert torch.allclose(cached_tensor, restored_tensor)

    def test_gradient_flow_uses_full_forward_path(self, encoder):
        encoder.train()
        x = torch.randn(2, 5, encoder.embedding_dim, device="cuda", requires_grad=True)

        output, state = encoder(x, initial_state=encoder.get_initial_state(2, x.device, x.dtype))
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert isinstance(state, dict)
        assert len(state) == 0
