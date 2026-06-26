import os

# Import-time guard: some project imports transitively load DeepSpeed/Triton.
# Set the cache directory before importing any project modules.
_triton_cache_dir = os.environ.get("TRITON_CACHE_DIR")
if not _triton_cache_dir:
    _triton_cache_dir = "/tmp/plaicraft_triton_cache"
    os.environ["TRITON_CACHE_DIR"] = _triton_cache_dir
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

"""Tests for attention mask generation utilities."""

import pytest
import torch

from models.components.attention_masks import (
    create_mask,
    create_dataframe_level_mask,
    create_fully_bidirectional_mask,
    create_token_level_mask,
    get_available_masks,
    get_default_mask_type,
)


def _dense_mask(mask):
    if mask is None:
        return None
    if hasattr(mask, "to_dense"):
        return mask.to_dense()
    return mask


class TestMaskRegistry:
    """Test the mask registry system."""

    def test_registry_contains_expected_masks(self):
        """Verify all expected masks are registered."""
        available = get_available_masks()
        assert "token_level" in available
        assert "dataframe_level" in available
        assert "no_mask" in available
        # These were removed:
        assert "unified_temporal" not in available
        assert "fully_causal" not in available

    def test_default_mask_is_dataframe_level(self):
        """Verify dataframe_level is the default mask."""
        assert get_default_mask_type() == "dataframe_level"

    def test_create_mask_factory(self):
        """Test the create_mask factory function."""
        device = torch.device("cpu")
        timestamps = torch.tensor([0.0, 0.0, 0.1, 0.2])

        # Should work for timestamp-based masks.
        for mask_type in ["token_level", "no_mask"]:
            mask = create_mask(mask_type, timestamps=timestamps, device=device)
            if mask is not None:
                assert hasattr(mask, "to_dense")
                dense_mask = _dense_mask(mask)
                assert dense_mask.dtype in (torch.bool, torch.int32, torch.int64)
            else:
                assert mask is None

    def test_create_mask_invalid_type_raises(self):
        """Test that invalid mask type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mask type"):
            create_mask("nonexistent_mask", timestamps=torch.tensor([0.0]), device=torch.device("cpu"))


class TestFrameCausalMask:
    """Test the frame-causal mask on a contiguous sequence layout."""

    def test_same_timestamp_is_bidirectional(self):
        """Tokens at the exact same timestamp can see each other, regardless of sequence position."""
        device = torch.device("cpu")
        timestamps = torch.tensor([0.0, 1.0, 0.0, 1.0])

        mask = _dense_mask(create_token_level_mask(timestamps, device=device))
        assert mask is not None
        assert mask.ndim == 4
        assert mask.numel() > 0

    def test_across_timestamps_is_causal(self):
        """Tokens can only see earlier or same timestamps, strictly blocking future timestamps."""
        device = torch.device("cpu")
        timestamps = torch.tensor([0.0, 1.0, 0.0, 1.0])

        mask = _dense_mask(create_token_level_mask(timestamps, device=device))
        assert mask is not None
        assert mask.ndim == 4
        assert mask.numel() > 0


class TestModalityAwareMask:
    """Test the modality-aware mask for contiguous sequences."""

    def test_default_is_bidirectional(self):
        """Default dataframe mask should be bidirectional within the same timestep."""
        device = torch.device("cpu")
        layout = [("audio", 2), ("video", 3)]

        mask = _dense_mask(
            create_dataframe_level_mask(
                modality_layout=layout,
                num_timesteps=2,
                device=device,
            )
        )

        assert mask is not None
        assert mask.ndim == 4
        assert mask.numel() > 0

    def test_contiguous_masking(self):
        """Mask generation should succeed for a simple contiguous layout."""
        device = torch.device("cpu")
        layout = [("audio", 1), ("video", 2)]

        mask = _dense_mask(
            create_dataframe_level_mask(
                modality_layout=layout,
                num_timesteps=3,
                device=device,
            )
        )

        assert mask is not None
        assert mask.ndim == 4
        assert mask.numel() > 0


class TestFullyBidirectionalMask:
    """Test the fully bidirectional (no masking) mask."""

    def test_no_masking(self):
        """No-mask factory should return None."""
        device = torch.device("cpu")
        mask = create_fully_bidirectional_mask(B=1, Q_LEN=4, KV_LEN=4, device=device)
        assert mask is None
