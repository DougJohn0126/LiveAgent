"""Unit tests for generative noise_schedulers (diffusion, EDM)."""

import pytest
import torch
import copy
from training.noise_schedulers import EDMScheduler, FlowMatchingScheduler, BaseScheduler


def _make_batch(batch_size: int, units: int, tokens_per_unit: int, dim: int):
    """Simulates tokenized data from MultimodalIO."""
    mod_data = {
        "tokens": torch.randn(batch_size, units, tokens_per_unit, dim),
        "time": torch.randn(batch_size, units, tokens_per_unit),
    }
    return {
        "video": copy.deepcopy(mod_data),
        "audio_speak": copy.deepcopy(mod_data),
        "key_press": copy.deepcopy(mod_data),
        "mouse_movement": copy.deepcopy(mod_data),
    }


def _assert_scheduler_output(x_tau, tau, eps, x_0, target_modalities):
    assert isinstance(x_tau, dict)
    assert isinstance(eps, dict)
    
    # 1. Verify x_tau contains only target modalities (flat keys)
    assert set(x_tau.keys()) == set(target_modalities), f"x_tau keys {set(x_tau.keys())} != expected {target_modalities}"

    # 2. Verify eps contains exactly the target modalities
    assert set(eps.keys()) == set(target_modalities), f"eps keys {set(eps.keys())} != expected {target_modalities}"
    
    B = tau.shape[0]
    for mod_name in target_modalities:
        # Check eps
        assert torch.is_tensor(eps[mod_name])
        assert eps[mod_name].shape[0] == B
        
        # Check x_tau tokens
        assert "tokens" in x_tau[mod_name]
        val = x_tau[mod_name]["tokens"]
        ref = x_0[mod_name]["tokens"]
        
        assert val.shape == ref.shape
        assert not torch.isnan(val).any()
        
        # Check that metadata was preserved
        assert "time" in x_tau[mod_name]
        assert torch.equal(x_tau[mod_name]["time"], x_0[mod_name]["time"])


class TestFlowMatchingScheduler:
    """Test suite for Flow Matching scheduler."""

    def test_forward_pass(self):
        target_modalities = ["audio_speak", "key_press", "mouse_movement"]
        scheduler = FlowMatchingScheduler(target_modalities=target_modalities, sampling_type="uniform")
        x_0 = _make_batch(2, 4, 8, 256)

        x_tau, tau, eps = scheduler(x_0)

        _assert_scheduler_output(x_tau, tau, eps, x_0, target_modalities)
        assert (tau >= 0).all() and (tau <= 1).all()

    def test_logitnormal_sampling(self):
        target_modalities = ["audio_speak", "key_press", "mouse_movement"]
        scheduler = FlowMatchingScheduler(target_modalities=target_modalities, sampling_type="logitnormal", p_mean=-1.2, p_std=1.2)
        x_0 = _make_batch(2, 4, 8, 256)

        x_tau, tau, eps = scheduler(x_0)

        _assert_scheduler_output(x_tau, tau, eps, x_0, target_modalities)
        assert (tau >= 0).all() and (tau <= 1).all()

    def test_time_shift(self):
        target_modalities = ["audio_speak", "key_press", "mouse_movement"]
        # Use a large difference in effective dimensions to see effect
        scheduler_shifted = FlowMatchingScheduler(target_modalities=target_modalities, use_time_shift=True, base_dim=1)
        scheduler_norm = FlowMatchingScheduler(target_modalities=target_modalities, use_time_shift=False)
        
        x_0 = _make_batch(1, 4, 8, 256)
        
        torch.manual_seed(123)
        x_tau_shifted, tau_shifted, _ = scheduler_shifted(x_0)
        
        torch.manual_seed(123)
        x_tau_norm, tau_norm, _ = scheduler_norm(x_0)
        
        assert torch.allclose(tau_shifted, tau_norm)
        for mod in target_modalities:
            assert "tokens" in x_tau_shifted[mod]
            assert not torch.allclose(x_tau_shifted[mod]["tokens"], x_tau_norm[mod]["tokens"], atol=1e-6)

    def test_loss_target(self):
        target_modalities = ["audio_speak", "key_press", "mouse_movement"]
        scheduler = FlowMatchingScheduler(target_modalities=target_modalities)
        x_0 = _make_batch(2, 4, 8, 256)
        
        x_tau, tau, eps = scheduler(x_0)
        velocity = scheduler.get_loss_target(x_tau, tau, eps)
        
        for name in target_modalities:
            true_velocity = eps[name] - x_0[name]["tokens"]
            # Flow matching v_t = eps - x_0. 
            # Calculated as (eps - x_tau) / (1 - tau)
            assert torch.allclose(velocity[name], true_velocity, atol=1e-4)


class TestSchedulerDeterminism:
    """Test deterministic behavior when using a fixed seed."""

    def test_deterministic_with_seed(self):
        target_modalities = ["audio_speak", "key_press", "mouse_movement"]
        scheduler = FlowMatchingScheduler(target_modalities=target_modalities, sampling_type="uniform")
        x_0 = _make_batch(2, 4, 8, 256)

        torch.manual_seed(123)
        x_tau_1, tau_1, eps_1 = scheduler(x_0)
        torch.manual_seed(123)
        x_tau_2, tau_2, eps_2 = scheduler(x_0)

        assert torch.allclose(tau_1, tau_2)
        for mod in target_modalities:
            assert torch.allclose(x_tau_1[mod]["tokens"], x_tau_2[mod]["tokens"])
            assert torch.allclose(eps_1[mod], eps_2[mod])
            assert torch.allclose(x_tau_1[mod]["time"], x_tau_2[mod]["time"])
