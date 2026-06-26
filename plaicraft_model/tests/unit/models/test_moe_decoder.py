"""Unit tests for MoE Decoder with actual data shapes."""

import pytest
import torch

from models.components.moe_decoder import MoEDecoder


def _prepare_model_inputs(batch: dict, timestep: torch.Tensor, embed_dim: int, modality_tokens_per_frame: dict = None):
    """Convert simple [B, N, D_in] inputs to MoEDecoder's expected [B, T, L, embed_dim] format."""
    device = next(iter(batch.values())).device
    batch_size = next(iter(batch.values())).shape[0]

    # Default token counts per dataframe if not provided
    if modality_tokens_per_frame is None:
        modality_tokens_per_frame = {
            "audio_speak": 15,
            "keyboard": 10,
            "mouse": 20,
            "video": 32,
        }

    # Project each modality to embed_dim and reshape [B, N, D_in] to [B, T, L, embed_dim]
    x_noisy = {}
    num_timesteps = None
    for mod_name, tensor in batch.items():
        B, N, D_in = tensor.shape

        if D_in > embed_dim:
            tensor = tensor[..., :embed_dim]
        elif D_in < embed_dim:
            pad = torch.zeros(B, N, embed_dim - D_in, device=device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], dim=-1)

        L = modality_tokens_per_frame.get(mod_name, N)
        if N % L != 0:
            raise ValueError(f"Modality {mod_name}: Cannot reshape {N} tokens with {L} tokens per dataframe")

        T = N // L
        if num_timesteps is None:
            num_timesteps = T
        elif num_timesteps != T:
            raise ValueError(f"Modality {mod_name}: expected T={num_timesteps}, got T={T}")

        x_noisy[mod_name] = tensor.view(B, T, L, embed_dim)

    # Create positional embeddings with time information
    positional_embeddings = {}
    for mod_name, tensor in x_noisy.items():
        _, T, L, _ = tensor.shape
        time = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        time = time.expand(batch_size, T, L)
        positional_embeddings[mod_name] = {"time": time}

    # Create embeddings: [B] timestep -> [B, 2, embed_dim]
    t_emb = timestep.view(-1, 1).expand(batch_size, embed_dim).unsqueeze(1)
    player_emb = torch.zeros(batch_size, 1, embed_dim, device=device)
    emb = torch.cat([t_emb, player_emb], dim=1)

    return x_noisy, positional_embeddings, emb


class TestMoEDecoder:
    """Test suite for MoE Decoder with real data pipeline shapes."""
    
    @pytest.fixture
    def config(self):
        """Configuration matching actual data pipeline."""
        return {
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 3,
            "modality_configs": {
                'audio_speak': {'input_dim': 128, 'hidden_dim': 256},
                'keyboard': {'input_dim': 160, 'hidden_dim': 256},
                'mouse': {'input_dim': 40, 'hidden_dim': 128},
            },
            "batch_size": 2,
        }
    
    def test_decoder_basic_forward(self, config):
        """Test basic forward pass with actual data shapes."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            modality_configs=config["modality_configs"],
            use_cross_attention=False,
            mask_type="token_level",
        )
        model.eval()
        
        # Input in [B, N, D] format and will be reshaped to [B, T, L, D]
        batch = {
            'audio_speak': torch.randn(config["batch_size"], 150, 128),
            'keyboard': torch.randn(config["batch_size"], 100, 160),
            'mouse': torch.randn(config["batch_size"], 200, 40),
        }
        timestep = torch.rand(config["batch_size"])
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        
        # Check outputs
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {'audio_speak', 'keyboard', 'mouse'}
        
        # Check shapes - outputs are [B, T*L, D] matching flattened inputs
        assert outputs['audio_speak'].shape == (config["batch_size"], 150, 128)
        assert outputs['keyboard'].shape == (config["batch_size"], 100, config["embed_dim"])
        assert outputs['mouse'].shape == (config["batch_size"], 200, config["embed_dim"])
        
        # Check no NaN
        for name, tensor in outputs.items():
            assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    
    def test_decoder_with_cross_attention(self, config):
        """Test forward pass with cross-attention and conditioning."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            modality_configs=config["modality_configs"],
            use_cross_attention=True,
            mask_type="token_level",
        )
        model.eval()
        
        batch = {
            'audio_speak': torch.randn(config["batch_size"], 150, 128),
            'keyboard': torch.randn(config["batch_size"], 100, 160),
        }
        timestep = torch.zeros(config["batch_size"])
        
        # Conditioning signal
        conditioning = torch.randn(config["batch_size"], 15, config["embed_dim"])
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb, history=conditioning)
        
        # Check outputs
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {'audio_speak', 'keyboard'}
        assert outputs['audio_speak'].shape == (config["batch_size"], 150, 128)
        assert outputs['keyboard'].shape == (config["batch_size"], 100, config["embed_dim"])
        
        for name, tensor in outputs.items():
            assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    
    def test_cross_attention_is_optional(self, config):
        """Test that cross-attention model works without conditioning."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            use_cross_attention=True,
            mask_type="token_level",
        )
        model.eval()
        
        batch = {
            'audio_speak': torch.randn(config["batch_size"], 150, 128),
        }
        timestep = torch.zeros(config["batch_size"])
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb, history=None)
        
        assert outputs['audio_speak'].shape == (config["batch_size"], 150, 128)
        assert not torch.isnan(outputs['audio_speak']).any()
    
    def test_cross_attention_changes_output(self, config):
        """Test that conditioning affects output."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            use_cross_attention=True,
            mask_type="token_level",
        )
        model.eval()
        
        batch = {
            'audio_speak': torch.randn(config["batch_size"], 150, 128),
        }
        timestep = torch.zeros(config["batch_size"])
        
        conditioning1 = torch.randn(config["batch_size"], 5, config["embed_dim"])
        conditioning2 = torch.randn(config["batch_size"], 5, config["embed_dim"])
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        
        with torch.no_grad():
            out_no_cond = model(x_noisy, pos_emb, emb, history=None)
            out_cond1 = model(x_noisy, pos_emb, emb, history=conditioning1)
            out_cond2 = model(x_noisy, pos_emb, emb, history=conditioning2)
        
        diff_no_vs_cond = (out_no_cond['audio_speak'] - out_cond1['audio_speak']).abs().mean()
        diff_cond1_vs_cond2 = (out_cond1['audio_speak'] - out_cond2['audio_speak']).abs().mean()
        
        assert diff_no_vs_cond > 0.01, "Conditioning should affect output"
        assert diff_cond1_vs_cond2 > 0.01, "Different conditioning should produce different outputs"
    
    def test_gradient_flow_with_cross_attention(self, config):
        """Test gradient flow through cross-attention."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            use_cross_attention=True,
            mask_type="token_level",
        )
        
        batch = {
            'audio_speak': torch.randn(config["batch_size"], 75, 128, requires_grad=True),
        }
        timestep = torch.rand(config["batch_size"], requires_grad=True)
        conditioning = torch.randn(config["batch_size"], 3, config["embed_dim"], requires_grad=True)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb, history=conditioning)
        loss = outputs['audio_speak'].sum()
        loss.backward()
        
        assert batch['audio_speak'].grad is not None
        assert conditioning.grad is not None
        assert not torch.isnan(batch['audio_speak'].grad).any()
        assert not torch.isnan(conditioning.grad).any()
    
    def test_causal_mask_prevents_future_leakage(self, config):
        """Test that causal masking prevents future tokens from influencing past tokens."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            mask_type="token_level",
        )
        
        batch = {'audio_speak': torch.randn(1, 45, 128, requires_grad=True)}
        timestep = torch.zeros(1)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        
        loss_t0 = outputs['audio_speak'][:, :15, :].sum()
        loss_t0.backward()
        
        grad = batch['audio_speak'].grad
        grad_t0 = grad[:, :15, :].abs().sum().item()
        grad_t1 = grad[:, 15:30, :].abs().sum().item()
        grad_t2 = grad[:, 30:45, :].abs().sum().item()
        
        assert grad_t0 > 0, "Dataframe 0 should have gradients"
        assert grad_t1 == 0, f"Future dataframe 1 should have zero gradient, got {grad_t1}"
        assert grad_t2 == 0, f"Future dataframe 2 should have zero gradient, got {grad_t2}"
    
    def test_bidirectional_within_dataframe(self, config):
        """Test that tokens within the same dataframe can influence each other."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            mask_type="dataframe_level",
        )

        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[1].weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(block.adaLN_modulation[1].bias, mean=0.0, std=0.02)
        
        batch = {'audio_speak': torch.randn(1, 15, 128, requires_grad=True)}
        timestep = torch.zeros(1)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        
        loss = outputs['audio_speak'][:, 0, :].sum()
        loss.backward()
        
        grad = batch['audio_speak'].grad
        
        for token_idx in range(15):
            token_grad = grad[:, token_idx, :].abs().sum().item()
            assert token_grad > 0, f"Token {token_idx} should have gradient (bidirectional within dataframe)"
    
    def test_single_modality(self, config):
        """Test with a single modality."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={'mouse': config["modality_configs"]['mouse']},
            mask_type="token_level",
        )
        model.eval()
        
        batch = {'mouse': torch.randn(2, 200, 40)}
        timestep = torch.rand(2)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        
        assert outputs['mouse'].shape == (2, 200, config["embed_dim"])
        assert not torch.isnan(outputs['mouse']).any()
    
    def test_multiple_modalities_different_lengths(self, config):
        """Test with modalities having different sequence lengths."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs=config["modality_configs"],
            mask_type="token_level",
        )
        model.eval()
        
        batch = {
            'audio_speak': torch.randn(2, 150, 128),
            'keyboard': torch.randn(2, 100, 160),
            'mouse': torch.randn(2, 200, 40),
        }
        timestep = torch.rand(2)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        
        assert outputs['audio_speak'].shape == (2, 150, 128)
        assert outputs['keyboard'].shape == (2, 100, config["embed_dim"])
        assert outputs['mouse'].shape == (2, 200, config["embed_dim"])
    
    def test_video_modality(self, config):
        """Test video modality."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={
                'video': {'input_dim': 256, 'hidden_dim': 512},
            },
            mask_type="token_level",
        )
        model.eval()
        
        B, N, D = 2, 320, 256
        batch = {'video': torch.randn(B, N, D)}
        timestep = torch.rand(B)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        
        assert outputs['video'].shape == (B, N, config["embed_dim"])
        assert not torch.isnan(outputs['video']).any()
    
    def test_video_gradient_flow(self, config):
        """Test gradient flow through video modality."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={
                'video': {'input_dim': 256, 'hidden_dim': 512},
            },
            mask_type="token_level",
        )
        
        batch = {
            'video': torch.randn(2, 192, 256, requires_grad=True),
        }
        timestep = torch.rand(2, requires_grad=True)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        loss = outputs['video'].sum()
        loss.backward()
        
        assert batch['video'].grad is not None
        assert not torch.isnan(batch['video'].grad).any()
    
    def test_video_with_other_modalities(self, config):
        """Test video modality combined with other modalities."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={
                'video': {'input_dim': 256, 'hidden_dim': 512},
                'audio_speak': {'input_dim': 128, 'hidden_dim': 256},
                'mouse': {'input_dim': 40, 'hidden_dim': 128},
            },
            mask_type="token_level",
        )
        model.eval()
        
        batch = {
            'video': torch.randn(2, 128, 256),
            'audio_speak': torch.randn(2, 60, 128),
            'mouse': torch.randn(2, 80, 40),
        }
        timestep = torch.rand(2)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        
        assert set(outputs.keys()) == {'video', 'audio_speak', 'mouse'}
        assert outputs['video'].shape == (2, 128, config["embed_dim"])
        assert outputs['audio_speak'].shape == (2, 60, config["embed_dim"])
        assert outputs['mouse'].shape == (2, 80, config["embed_dim"])
        
        for name, tensor in outputs.items():
            assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    
    def test_timestep_embedding_shape(self, config):
        """Test timestep embeddings shape."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            mask_type="token_level",
        )
        
        batch_size = 4
        timestep = torch.rand(batch_size)
        
        batch = {'audio_speak': torch.randn(batch_size, 45, 128)}
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        
        assert emb.shape == (batch_size, 2, config["embed_dim"])
        assert not torch.isnan(emb).any()
    
    def test_different_timesteps_produce_different_outputs(self, config):
        """Test that different timesteps affect model output."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            mask_type="token_level",
        )
        model.eval()

        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[1].weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(block.adaLN_modulation[1].bias, mean=0.0, std=0.02)
        
        batch = {'audio_speak': torch.randn(2, 75, 128)}
        
        t0 = torch.zeros(2)
        t1 = torch.ones(2)
        
        x_noisy_0, pos_emb_0, emb_0 = _prepare_model_inputs(batch, t0, config["embed_dim"])
        x_noisy_1, pos_emb_1, emb_1 = _prepare_model_inputs(batch, t1, config["embed_dim"])
        
        with torch.no_grad():
            out_t0 = model(x_noisy_0, pos_emb_0, emb_0)
            out_t1 = model(x_noisy_1, pos_emb_1, emb_1)
        
        diff = (out_t0['audio_speak'] - out_t1['audio_speak']).abs().mean()
        assert diff > 0.001, f"Different timesteps should produce different outputs, got diff={diff}"
    
    def test_timestep_gradient_flow(self, config):
        """Test that timestep information flows through the model."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            mask_type="token_level",
        )
        
        batch = {
            'audio_speak': torch.randn(2, 75, 128, requires_grad=True),
        }
        timestep = torch.rand(2, requires_grad=True)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        outputs = model(x_noisy, pos_emb, emb)
        loss = outputs['audio_speak'].sum()
        loss.backward()
        
        assert batch['audio_speak'].grad is not None
        assert not torch.isnan(batch['audio_speak'].grad).any()
    
    def test_adaln_zero_initialization(self, config):
        """Test that AdaLN modulation is zero-initialized for stability."""
        model = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=1,
            modality_configs={'audio_speak': config["modality_configs"]['audio_speak']},
            mask_type="token_level",
        )
        
        for block in model.blocks:
            modulation_linear = block.adaLN_modulation[1]
            assert torch.allclose(modulation_linear.weight, torch.zeros_like(modulation_linear.weight)), \
                "AdaLN modulation weight should be zero-initialized"
            assert torch.allclose(modulation_linear.bias, torch.zeros_like(modulation_linear.bias)), \
                "AdaLN modulation bias should be zero-initialized"
        
        batch = {'audio_speak': torch.randn(1, 45, 128)}
        timestep = torch.rand(1)
        
        x_noisy, pos_emb, emb = _prepare_model_inputs(batch, timestep, config["embed_dim"])
        
        with torch.no_grad():
            output = model(x_noisy, pos_emb, emb)
        
        assert not torch.isnan(output['audio_speak']).any()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
