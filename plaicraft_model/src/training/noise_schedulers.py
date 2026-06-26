"""
Generative process schedulers (diffusion, consistency, EDM).

Each scheduler defines how to add noise and interpolate based on a time parameter.
All schedulers train the model to predict x_1 (the original clean data).
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
from data.data_classes import FullData
from utils.noise_utils import get_noise_sampler, time_shift, calculate_effective_dimension
from utils.constants import VALID_MODALITIES


class BaseScheduler(ABC, nn.Module):
    """Base class for generative process schedulers."""
    
    def __init__(self, target_modalities: list):
        super().__init__()
        for mod in target_modalities:
            if mod not in VALID_MODALITIES:
                raise ValueError(f"Unrecognized target modality: {mod}. Allowed modalities are: {VALID_MODALITIES}")
        self.target_modalities = target_modalities

    @abstractmethod
    def forward(
        self,
        x_1: Dict[str, torch.Tensor],
    ) -> tuple:
        """
        Sample tau and interpolate to x_t.
        
        Args:
            x_1: Target/data distribution dict [B, T, L, D]
        
        Returns:
            (x_t, tau, eps): Interpolated latents [B, T, L, D], 
                             time parameter [B, 1], and noise eps.
        """
        pass
    
    def get_loss_target(
        self, 
        eps: Dict[str, torch.Tensor],
        x_0: FullData,
        **kwargs, # <-- Allow taking extra arguments for diffusion scheme in the future.
    ) -> FullData:
        """
        Returns the target for the model to predict (e.g., x_0 or velocity).
        Default returns clean data x_0.
        """
        return x_0


class EDMScheduler(BaseScheduler):
    """
    Elucidating the Design Space of Diffusion-Based Generative Models (EDM).
    Uses σ(t) schedule with power-law noise levels.
    """
    def __init__(self, target_modalities: list, p: float = -1.2, sigma_min: float = 0.002, sigma_max: float = 80.0, **kwargs):
        super().__init__(target_modalities)
        pass
    
    def forward(
        self,
        x_1: Dict[str, torch.Tensor],
    ) -> tuple:
        pass


class FlowMatchingScheduler(BaseScheduler):
    """
    Flow Matching (FM) with independent couplings using the noise-level convention.
    
    Convention (aligned with DM, deviates from original FM paper):
      - t = 0: Clean data (x_0)
      - t = 1: Pure noise (eps)
      
    Interpolation (Straight trajectory):
      x_t = (1 - t) * x_0 + t * eps
      
    Target velocity (Constant):
      v_t = dx_t / dt = eps - x_0
    """

    def __init__(self, 
                 target_modalities: list,
                 sampling_type: str = "logitnormal", 
                 use_time_shift: bool = False, 
                 base_dim: int = 4096, 
                 **kwargs
                 ):
        super().__init__(target_modalities)
        self.noise_sampler = get_noise_sampler(sampling_type, **kwargs)
        self.use_time_shift = use_time_shift
        self.base_dim = base_dim

    def _forward_full_data(self, x_0: FullData) -> tuple:
        # Get reference modality for device/dtype/batch_size (use first target modality)
        ref_modality = self.target_modalities[0]
        ref_tensor = x_0.get_modality(ref_modality)
        assert ref_tensor is not None, f"'{ref_modality}' not found in x_0."

        dtype = ref_tensor.dtype
        B = ref_tensor.shape[0]
        device = ref_tensor.device

        tau = self.noise_sampler(B, device, dtype)

        eps = {}
        x_tau_batch = x_0.to_dict()
        for modality in VALID_MODALITIES:
            if modality not in self.target_modalities:
                x_tau_batch[modality] = None

        for modality in self.target_modalities:
            x_0_tensor = x_0.get_modality(modality)
            if x_0_tensor is None:
                raise ValueError(f"Modality {modality} not in x_0.")

            if self.use_time_shift:
                m = calculate_effective_dimension({modality: x_0_tensor})
                alpha = m / self.base_dim
                modality_tau = time_shift(tau, alpha)
            else:
                modality_tau = tau

            tau_reshaped = modality_tau.view(B, *([1] * (x_0_tensor.ndim - 1)))
            eps_mod = torch.randn_like(x_0_tensor, device=device)
            eps[modality] = eps_mod
            x_tau_tensor = (1.0 - tau_reshaped) * x_0_tensor + tau_reshaped * eps_mod
            x_tau_batch[modality] = x_tau_tensor

        return FullData(batch=x_tau_batch), tau, eps



    def forward(
        self,
        x_0: FullData,
    ) -> tuple:
        """
        Sample tau (noise level) and compute straight-line interpolation.

        Args:
            x_0: FullData, containing clean data.
            
        Returns:
            x_tau: FullData, containing only the target modalities + metadata + padding_mask.
            tau: torch.Tensor, of shape [B, 1].
            eps: Dict[str, torch.Tensor], containing only the target modalities. 
        """        
        return self._forward_full_data(x_0)


    def get_loss_target(
        self, 
        eps: Dict[str, torch.Tensor],
        x_0: FullData,
        **kwargs,
    ) -> FullData:
        """
        The target is the constant velocity vector: v = eps - x_0.
        By using x_0 directly, we avoid the unstable division by (1 - tau)
        as tau approaches 1.

        Returns:
            velocity: FullData with target modalities converted to velocity.
        """
        # Start with a copy of x_0 to preserve metadata and masks.
        velocity_dict = x_0.to_dict()
        for modality in VALID_MODALITIES:
            if modality not in self.target_modalities:
                velocity_dict[modality] = None

        # Compute velocity for each target modality.
        for name in self.target_modalities:
            x_0_tensor = x_0.get_modality(name)
            if name in eps:
                velocity_dict[name] = eps[name] - x_0_tensor

        return FullData(batch=velocity_dict)


# Registry for easy scheduler selection
SCHEDULERS = {
    "edm": EDMScheduler,
    "flow_matching": FlowMatchingScheduler,
}


def get_scheduler(scheduler_type: str, **kwargs) -> BaseScheduler:
    """Get a scheduler by name."""
    if scheduler_type not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {scheduler_type}. Options: {list(SCHEDULERS.keys())}")
    return SCHEDULERS[scheduler_type](**kwargs)
