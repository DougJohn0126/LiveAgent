#!/usr/bin/env python3
"""Test script to verify tokenization and decoding pipeline."""

import torch
from models.components.multimodal_io import MultimodalIO
from data.data_classes import FullData
from utils.constants import MODALITY_SHAPES, VIDEO_FPS, AUDIO_TOKEN_FPS

def test_video_patchify_unpatchify():
    """Test that video patchification can be reversed."""
    # Simulate patch encoding/decoding (matching multimodal I/O logic)
    patch_h, patch_w = 8, 8
    
    # Original video shape: [B, T, F, C, H, W]
    B, T, F, C, H, W = 2, 1, 2, 4, 96, 160
    video = torch.randn(B, T, F, C, H, W)
    
    # Patchify (exactly as in MultimodalIO._tokenize_video)
    gh, gw = H // patch_h, W // patch_w
    patches = video.reshape(B, T, F, C, gh, patch_h, gw, patch_w)
    patches = patches.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, T, F*gh*gw, C*patch_h*patch_w)
    
    print(f"Original video shape: {video.shape}")
    print(f"Patchified shape: {patches.shape}")
    
    # Unpatchify (what our decode_base_dim does)
    # Input: [B, T, F*gh*gw, C*patch_h*patch_w]
    # We need to reverse the permutation (0, 1, 2, 4, 6, 3, 5, 7)
    # The original was [B, T, F, C, gh, patch_h, gw, patch_w]
    # After permute (0, 1, 2, 4, 6, 3, 5, 7) we get [B, T, F, gh, gw, C, patch_h, patch_w]
    # So to go back, we need the inverse permutation
    patches_reshaped = patches.reshape(B, T, F, gh, gw, C, patch_h, patch_w)
    # Inverse of (0, 1, 2, 4, 6, 3, 5, 7) is (0, 1, 2, 5, 3, 6, 4, 7)
    video_recovered = patches_reshaped.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(B, T, F, C, H, W)
    
    print(f"Recovered video shape: {video_recovered.shape}")
    
    # Check if they match (should be very close due to floating point)
    error = (video - video_recovered).abs().max()
    print(f"Max reconstruction error: {error:.2e}")
    
    assert video_recovered.shape == video.shape, f"Shape mismatch: {video_recovered.shape} != {video.shape}"
    assert error < 1e-6, f"Patchification error too large: {error}"
    print("✓ Video patchify/unpatchify test passed!\n")


def test_project_outputs_and_decode():
    """Test project_outputs and decode_base_dim functions."""
    # Initialize multimodal I/O adapter
    patch_h, patch_w = 8, 8
    model_dim = 512
    
    multimodal_io = MultimodalIO(
        patch_h=patch_h,
        patch_w=patch_w,
        model_dim=model_dim,
        player_embed_dim=128,
    )
    
    B, T = 2, 4
    
    # Test 1: moe_decoder_output_to_fulldata (MoEDecoder output → FullData)
    print("Test 1: moe_decoder_output_to_fulldata (flattened [B,T*L,model_dim] → FullData)")
    print("=" * 60)
    
    # Simulated MoEDecoder outputs: [B, T*L, model_dim]
    predictions = {
        'video': torch.randn(B, T*480, model_dim),  # 480 = 2*12*20 patches
        'audio_speak': torch.randn(B, T*15, model_dim),
        'audio_hear': torch.randn(B, T*15, model_dim),
        'key_press': torch.randn(B, T*10, model_dim),
        'mouse_movement': torch.randn(B, T*20, model_dim),
    }
    
    print("Input shapes (from MoEDecoder):")
    for name, tensor in predictions.items():
        print(f"  {name}: {tensor.shape}")
    
    active_modality_names = list(predictions.keys())
    modality_lengths = [predictions[name].shape[1] for name in active_modality_names]
    modality_shapes = {
        'video': (T, 480),
        'audio_speak': (T, 15),
        'audio_hear': (T, 15),
        'key_press': (T, 10),
        'mouse_movement': (T, 20),
    }
    x_flat = torch.cat([predictions[name] for name in active_modality_names], dim=1)

    reference_full_data = FullData(batch={
        'video': torch.zeros(B, T, 2, 4, 96, 160),
        'audio_speak': torch.zeros(B, T, 15, 128),
        'audio_hear': torch.zeros(B, T, 15, 128),
        'key_press': torch.zeros(B, T, 10, 16),
        'mouse_movement': torch.zeros(B, T, 20, 2),
    })

    decoded_predictions = multimodal_io.moe_decoder_output_to_fulldata(
        x_flat=x_flat,
        active_modality_names=active_modality_names,
        modality_shapes=modality_shapes,
        reference_full_data=reference_full_data,
    )
    
    print("\nOutput shapes (after project_outputs):")
    expected_shapes = {
        'video': (B, T, 2, 4, 96, 160),
        'audio_speak': (B, T, 15, 128),
        'audio_hear': (B, T, 15, 128),
        'key_press': (B, T, 10, 16),
        'mouse_movement': (B, T, 20, 2),
    }
    
    decoded_dict = decoded_predictions.to_dict()
    for name in expected_shapes:
        tensor = decoded_dict[name]
        expected = expected_shapes[name]
        actual = tensor.shape
        match = "✓" if actual == expected else "✗"
        print(f"  {name}: {actual} (expected {expected}) {match}")
        assert actual == expected, f"Shape mismatch for {name}"
    
    print("✓ moe_decoder_output_to_fulldata test passed!\n")
    
    # Test 2: decode_base_dim ([B,T,L,base_dim] → original shapes)
    print("Test 2: decode_base_dim ([B,T,L,base_dim] → original shapes)")
    print("=" * 60)
    
    base_dim = multimodal_io.base_dim
    base_dim_tokens = {
        'video': torch.randn(B, T, 480, base_dim),  # 480 patches
        'audio_speak': torch.randn(B, T, 15, base_dim),
        'audio_hear': torch.randn(B, T, 15, base_dim),
        'key_press': torch.randn(B, T, 10, base_dim),
        'mouse_movement': torch.randn(B, T, 20, base_dim),
    }
    
    print("Input shapes (base_dim tokens):")
    for name, tensor in base_dim_tokens.items():
        print(f"  {name}: {tensor.shape}")
    
    decoded = multimodal_io.decode_base_dim(base_dim_tokens)
    
    print("\nOutput shapes (after decode_base_dim):")
    for name, tensor in decoded.items():
        expected = expected_shapes[name]
        actual = tensor.shape
        match = "✓" if actual == expected else "✗"
        print(f"  {name}: {actual} (expected {expected}) {match}")
        assert actual == expected, f"Shape mismatch for {name}"
    
    print("✓ decode_base_dim test passed!\n")


if __name__ == "__main__":
    print("Testing video patchification...")
    test_video_patchify_unpatchify()
    
    print("\nTesting multimodal I/O decoding functions...")
    test_project_outputs_and_decode()
    
    print("\n✅ All tests passed!")
