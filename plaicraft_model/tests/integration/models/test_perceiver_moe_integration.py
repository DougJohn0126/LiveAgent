"""Integration test for Perceiver + MoE Decoder."""

import pytest
import torch

from models.components.moe_decoder import MoEDecoder
from models.components.perceiver import PerceiverIO


class TestPerceiverMoEDecoderIntegration:
    """Test suite for Perceiver conditioning with MoE Decoder."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for integration tests."""
        return {
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
            "modality_configs": {
                'audio_speak': {'input_dim': 128, 'hidden_dim': 256},
                'keyboard': {'input_dim': 160, 'hidden_dim': 256},
                'mouse': {'input_dim': 40, 'hidden_dim': 128},
            },
            "batch_size": 2,
            "sampling_rates": {
                'audio_speak': 75.0,
                'audio_hear': 75.0,
                'keyboard': 50.0,
                'mouse': 100.0,
                'video': 10.0,
            },
            # Perceiver config
            "perceiver_depth": 4,
            "perceiver_num_latents": 64,
            "perceiver_latent_dim": 256,
        }
    
    def test_perceiver_as_conditioning(self, config):
        """Test using Perceiver output as conditioning for MoE decoder."""
        
        # Create Perceiver for processing encoder/context data
        perceiver = PerceiverIO(
            depth=config["perceiver_depth"],
            dim=512,  # Input dimension for perceiver (e.g., from video encoder)
            queries_dim=config["embed_dim"],
            num_latents=config["perceiver_num_latents"],
            latent_dim=config["perceiver_latent_dim"],
        )
        
        # Create MoE decoder with cross-attention enabled
        moe_decoder = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            modality_configs=config["modality_configs"],
            use_cross_attention=True,
            sampling_rates=config["sampling_rates"],
        )
        
        perceiver.eval()
        moe_decoder.eval()
        
        # Simulate encoder output (e.g., from video encoder)
        encoder_output = torch.randn(config["batch_size"], 50, 512)
        
        # Process through Perceiver to get conditioning
        with torch.no_grad():
            # Perceiver without queries returns latent representation
            conditioning = perceiver(encoder_output)
        
        # Create decoder input batch with strict [B, N, D] format
        batch = {
            'audio_speak': torch.randn(config["batch_size"], 150, 128),  # 10*15 tokens
            'keyboard': torch.randn(config["batch_size"], 100, 160),    # 10*2*5 tokens
            'mouse': torch.randn(config["batch_size"], 200, 40),        # 10*2*10 tokens
        }
        timestep = torch.rand(config["batch_size"])  # Continuous timesteps in [0, 1]
        
        # Forward pass through MoE decoder with Perceiver conditioning
        with torch.no_grad():
            outputs = moe_decoder(batch, timestep=timestep, conditioning=conditioning)
        
        # Verify outputs
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {'audio_speak', 'keyboard', 'mouse'}
        
        # Check shapes - outputs should match input [B, N, D]
        assert outputs['audio_speak'].shape == (config["batch_size"], 150, 128)
        assert outputs['keyboard'].shape == (config["batch_size"], 100, 160)
        assert outputs['mouse'].shape == (config["batch_size"], 200, 40)
        
        # Verify conditioning shape matches what decoder expects
        assert conditioning.shape == (config["batch_size"], config["perceiver_num_latents"], config["perceiver_latent_dim"])
        
        # Check no NaN values
        for name, tensor in outputs.items():
            assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    
    def test_gradient_flow_perceiver_to_decoder(self, config):
        """Test gradient flow from decoder through Perceiver conditioning."""
        
        # Create models
        perceiver = PerceiverIO(
            depth=2,  # Smaller for faster test
            dim=512,
            queries_dim=config["embed_dim"],
            num_latents=32,
            latent_dim=config["embed_dim"],
        )
        
        moe_decoder = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={
                'audio_speak': {'input_dim': 128, 'hidden_dim': 256},
                'mouse': {'input_dim': 40, 'hidden_dim': 128},
            },
            use_cross_attention=True,
            sampling_rates=config["sampling_rates"],
        )
        
        # Create inputs with gradients in [B, N, D] format
        encoder_output = torch.randn(config["batch_size"], 30, 512, requires_grad=True)
        
        batch = {
            'audio_speak': torch.randn(config["batch_size"], 150, 128, requires_grad=True),
            'mouse': torch.randn(config["batch_size"], 200, 40, requires_grad=True),
        }
        timestep = torch.rand(config["batch_size"], requires_grad=True)
        
        # Forward pass
        conditioning = perceiver(encoder_output)
        outputs = moe_decoder(batch, timestep=timestep, conditioning=conditioning)
        
        # Compute loss
        loss = sum(tensor.sum() for tensor in outputs.values())
        loss.backward()
        
        # Check gradients exist
        assert encoder_output.grad is not None
        assert batch['audio_speak'].grad is not None
        assert batch['mouse'].grad is not None
        
        # Check for NaN gradients
        assert not torch.isnan(encoder_output.grad).any()
        for name, tensor in batch.items():
            assert tensor.grad is not None
            assert not torch.isnan(tensor.grad).any()
    
    def test_conditioning_affects_decoder_output(self, config):
        """Test that different Perceiver conditioning produces different outputs."""
        
        perceiver = PerceiverIO(
            depth=2,
            dim=512,
            queries_dim=config["embed_dim"],
            num_latents=32,
            latent_dim=config["embed_dim"],
        )
        
        moe_decoder = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs={'audio_speak': {'input_dim': 128, 'hidden_dim': 256}},
            use_cross_attention=True,
            sampling_rates=config["sampling_rates"],
        )
        
        perceiver.eval()
        moe_decoder.eval()
        
        # Same decoder input [B, N, D]
        batch = {'audio_speak': torch.randn(2, 150, 128)}
        timestep = torch.zeros(2)
        
        # Different encoder outputs
        encoder1 = torch.randn(2, 30, 512)
        encoder2 = torch.randn(2, 30, 512)
        
        with torch.no_grad():
            # Get different conditioning
            cond1 = perceiver(encoder1)
            cond2 = perceiver(encoder2)
            
            # Get outputs
            out1 = moe_decoder(batch, timestep=timestep, conditioning=cond1)
            out2 = moe_decoder(batch, timestep=timestep, conditioning=cond2)
        
        # Different conditioning should produce different outputs
        diff = (out1['audio_speak'] - out2['audio_speak']).abs().mean()
        assert diff > 0.005, "Different conditioning should affect decoder output"
    
    def test_full_pipeline_with_all_modalities(self, config):
        """Test complete pipeline with all modalities: audio_speak, keyboard, mouse."""
        
        # Create models
        perceiver = PerceiverIO(
            depth=config["perceiver_depth"],
            dim=768,  # Typical video encoder dimension
            queries_dim=config["embed_dim"],
            num_latents=config["perceiver_num_latents"],
            latent_dim=config["perceiver_latent_dim"],
        )
        
        moe_decoder = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            modality_configs=config["modality_configs"],
            use_cross_attention=True,
            sampling_rates=config["sampling_rates"],
        )
        
        perceiver.eval()
        moe_decoder.eval()
        
        # Simulate realistic scenario
        batch_size = config["batch_size"]
        
        # Video encoder output (e.g., from a vision transformer)
        video_features = torch.randn(batch_size, 100, 768)
        
        # Process through Perceiver
        with torch.no_grad():
            conditioning = perceiver(video_features)
        
        # Decoder inputs in strict [B, N, D] format
        action_batch = {
            'audio_speak': torch.randn(batch_size, 300, 128),  # 20*15 tokens
            'keyboard': torch.randn(batch_size, 200, 160),     # 20*2*5 tokens
            'mouse': torch.randn(batch_size, 400, 40),         # 20*2*10 tokens
        }
        timestep = torch.rand(batch_size)
        
        # Full forward pass
        with torch.no_grad():
            action_outputs = moe_decoder(action_batch, timestep=timestep, conditioning=conditioning)
        
        # Verify all modalities processed
        assert len(action_outputs) == 3
        
        # Verify shapes match input [B, N, D]
        expected_shapes = {
            'audio_speak': (batch_size, 300, 128),
            'keyboard': (batch_size, 200, 160),
            'mouse': (batch_size, 400, 40),
        }
        
        for modality_name in action_batch.keys():
            output_tensor = action_outputs[modality_name]
            expected = expected_shapes[modality_name]
            assert output_tensor.shape == expected, f"{modality_name}: expected {expected}, got {output_tensor.shape}"
        
        # Verify no NaN
        for tensor in action_outputs.values():
            assert not torch.isnan(tensor).any()
    
    def test_long_sequence_lengths(self, config):
        """Test with longer sequence lengths across batch and modalities."""
        
        perceiver = PerceiverIO(
            depth=2,
            dim=512,
            queries_dim=config["embed_dim"],
            num_latents=32,
            latent_dim=config["embed_dim"],
        )
        
        moe_decoder = MoEDecoder(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=2,
            modality_configs=config["modality_configs"],
            use_cross_attention=True,
            sampling_rates=config["sampling_rates"],
        )
        
        perceiver.eval()
        moe_decoder.eval()
        
        # Encoder output
        encoder_output = torch.randn(3, 75, 512)
        
        # Longer sequences in [B, N, D] format
        batch = {
            'audio_speak': torch.randn(3, 450, 128),   # 30*15 tokens
            'keyboard': torch.randn(3, 300, 160),      # 30*2*5 tokens
            'mouse': torch.randn(3, 600, 40),          # 30*2*10 tokens
        }
        timestep = torch.rand(3)
        
        with torch.no_grad():
            conditioning = perceiver(encoder_output)
            outputs = moe_decoder(batch, timestep=timestep, conditioning=conditioning)
        
        # Check each modality output matches input [B, N, D]
        assert outputs['audio_speak'].shape == (3, 450, 128)
        assert outputs['keyboard'].shape == (3, 300, 160)
        assert outputs['mouse'].shape == (3, 600, 40)


if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v", "-p", "no:warnings"]))
