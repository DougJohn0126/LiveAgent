import torch
from tqdm import tqdm
from data.data_classes import FullData
from utils.constants import MODALITY_SHAPES, VALID_MODALITIES
from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

###############################################################################
## CORE UTILITIES
###############################################################################


def apply_euler_update(
    x_tau: FullData,
    velocity: FullData,
    dtau: float,
    target_modalities: list
) -> FullData:
    """
    Apply Euler update x += velocity * dt for each target modality.
    """
    updated = x_tau.to_dict()
    for mod in target_modalities:
        updated[mod] = x_tau.get_modality(mod) + velocity.get_modality(mod) * dtau
    return FullData(batch=updated)


def apply_heun_update(
    x_tau: FullData,
    velocity_1: FullData,
    velocity_2: FullData,
    dtau: float,
    target_modalities: list,
) -> FullData:
    """
    Apply Heun (trapezoidal) update: x += 0.5 * (v1 + v2) * dt for each target modality.
    """
    updated = x_tau.to_dict()
    for mod in target_modalities:
        v_avg = 0.5 * (velocity_1.get_modality(mod) + velocity_2.get_modality(mod))
        updated[mod] = x_tau.get_modality(mod) + v_avg * dtau
    return FullData(batch=updated)


@torch.no_grad()
def flow_matching_sampler(
    model: torch.nn.Module,
    memory: dict[str, torch.Tensor],
    x_tau_init: FullData,
    target_modalities: list = VALID_MODALITIES,
    num_steps: int = 50,
    method: str = "euler",
):
    """
    Sampler for Flow Matching in TOKEN space.
    Solves dx/dt = v(x, t) from t=1.0 down to t=0.0.

    Args:
        method: "euler" (first-order) or "heun" (second-order predictor-corrector).
                Heun uses 2x model calls per step but has O(dt²) truncation error.

    Returns:
        x_tau (FullData): Final denoised data in original modality space.
    """
    if method not in ("euler", "heun"):
        raise ValueError(f"Unknown flow matching method: {method!r}. Choose 'euler' or 'heun'.")
    stm = memory.get("stm", None)
    if stm is None:
        raise ValueError("memory must include 'stm'")
    device = stm.device
    B = x_tau_init.get_modality(target_modalities[0]).shape[0]

    # Linear schedule: 1.0 (pure noise) -> 0.0 (clean data).
    tau_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    x_tau = x_tau_init

    pbar = tqdm(range(num_steps), desc=f"Flow Matching ({method.capitalize()})", leave=False)

    for i in pbar:
        tau = tau_steps[i]
        tau_next = tau_steps[i + 1]
        dtau = (tau_next - tau).item()
        tau_tensor = torch.full((B, 1), tau, device=device, dtype=torch.float32)

        # Model forward pass (predictor)
        vel_1 = model(x_tau=x_tau, tau=tau_tensor, memory=memory)

        if method == "euler":
            x_tau = apply_euler_update(x_tau, vel_1, dtau, target_modalities)
        elif method == "heun": 
            # Average current and next velocities (trapezoidal rule)
            x_pred = apply_euler_update(x_tau, vel_1, dtau, target_modalities)
            tau_next_tensor = torch.full((B, 1), tau_next, device=device, dtype=torch.float32)
            vel_2 = model(x_tau=x_pred, tau=tau_next_tensor, memory=memory)
            x_tau = apply_heun_update(x_tau, vel_1, vel_2, dtau, target_modalities)
    return x_tau


###############################################################################
## FACTORY
###############################################################################

def get_denoising_fn(config):
    """
    Returns a unified sampler function. 
    """
    @torch.no_grad()
    def sampler_fn(model, memory: dict[str, torch.Tensor], x_tau_init: FullData, target_modalities=VALID_MODALITIES, num_steps=2):
        denoising_type = getattr(config, "denoising_type", "flow")
        if denoising_type == "flow":
            return flow_matching_sampler(
                model=model,
                memory=memory,
                x_tau_init=x_tau_init,
                target_modalities=target_modalities,
                num_steps=num_steps,
                method=getattr(config, "flow_matching_sampler", "euler"),
            )
        raise ValueError(f"Unknown denoising type: {denoising_type}")
    return sampler_fn


def init_noise_target(
    batch_size: int,
    target_pred_len: int,
    target_modalities: list,
    metadata=None,
    device=None,
) -> FullData:
    """Initializes a FullData object with Gaussian noise for target modalities."""
    batch = {"metadata": metadata}
    for mod in target_modalities:
        batch[mod] = torch.randn((batch_size, target_pred_len, *MODALITY_SHAPES[mod]), device=device)

    batch["dataframe_indices"] = (
        torch.arange(0, target_pred_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    return FullData(batch=batch)


@torch.no_grad()
def generate_chunk(
    model: torch.nn.Module,
    config,
    memory: dict,
    batch_size: int,
    target_pred_len: int,
    target_modalities: list,
    metadata=None,
) -> FullData:
    """Executes a single generative pass."""
    device = next(model.parameters()).device
    x_tau_init = init_noise_target(
        batch_size=batch_size,
        target_pred_len=target_pred_len,
        target_modalities=target_modalities,
        metadata=metadata,
        device=device,
    )
    sampler_fn = get_denoising_fn(config)

    return sampler_fn(
        model=model,
        memory=memory,
        x_tau_init=x_tau_init,
        target_modalities=target_modalities,
        num_steps=getattr(config, "num_denoising_steps", 50),
    )