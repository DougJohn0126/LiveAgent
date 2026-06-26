import torch
from functools import partial

def logit_normal_sampler(B: int, 
                         device: torch.device, 
                         dtype: torch.dtype, 
                         p_mean: float = -1.2, 
                         p_std: float = 1.2
                         ) -> torch.Tensor:
    """
    Logit-Normal sampling for timesteps.
    Reference for CIFAR-10 p_mean: -1.2, p_std: 1.2: https://github.com/facebookresearch/flow_matching/blob/main/examples/image/training/train_loop.py
    """
    z = torch.randn(B, 1, device=device, dtype=dtype)
    return torch.sigmoid(z * p_std + p_mean)


def uniform_sampler(B: int, 
                    device: torch.device, 
                    dtype: torch.dtype
                    ) -> torch.Tensor:
    """Uniform sampling for timesteps."""
    return torch.rand(B, 1, device=device, dtype=dtype)


def time_shift(mu: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Dimension-dependent noise schedule shift.
    Reference: Esser et al. (2024), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    Equation: t_m = (alpha * t_n) / (1 + (alpha - 1) * t_n)
    Args:
        mu: Original timesteps t_n in [0, 1]
        alpha: Dimension-dependent scaling factor m/n
    """
    return (alpha * mu) / (1 + (alpha - 1) * mu).clamp(min=1e-5)


def calculate_effective_dimension(x_0: dict) -> int:
    """
    Calculates the effective data dimension m: sum(tokens * dimensionality).
    x_0: Dict[str, torch.Tensor] where tensors are [B, T, L, D] or similar.
    """
    m = 0
    for tensor in x_0.values():
        if torch.is_tensor(tensor):
            # Exclude batch dimension
            # Assuming shape is [B, ...]
            m += tensor[0].numel()
    return m


def get_noise_sampler(sampling_type: str, 
                      **kwargs
                      ):
    """
    Factory to get a noise level sampler.
    """
    if sampling_type == "uniform":
        return uniform_sampler
    elif sampling_type == "logitnormal":
        return partial(logit_normal_sampler, 
                       p_mean=kwargs.get("p_mean", -1.2), 
                       p_std=kwargs.get("p_std", 1.2)
                       )
    else:
        raise ValueError(f"Unknown sampling type: {sampling_type}")
