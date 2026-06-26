import logging
import math
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from lightning.pytorch.strategies import DeepSpeedStrategy

from typing import Dict

from data.data_classes import FullData
from utils.constants import VALID_MODALITIES

logger = logging.getLogger(__name__)


class PlaiV1LightningModule(pl.LightningModule):
    """Lightning trainer module for the plai_v1 model."""

    def __init__(
        self,
        model: torch.nn.Module,
        noise_scheduler: torch.nn.Module,
        modality_loss_weights: Dict[str, float] = None,
        datamodule = None,
        **kwargs
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=["datamodule", "model", "noise_scheduler"])

        assert model is not None, "PlaiV1LightningModule requires a 'model' instance."
        assert noise_scheduler is not None, "PlaiV1LightningModule requires a 'noise_scheduler' instance."

        self.model = model
        self.noise_scheduler = noise_scheduler

        # Store modality loss weights from Hydra config (or use 1.0 as fallback)
        if modality_loss_weights is None:
            modality_loss_weights = {}
        self.modality_loss_weights = {
            mod_name: float(modality_loss_weights.get(mod_name, 1.0)) for mod_name in VALID_MODALITIES
        }

        # DataModule (optional, can be passed separately to Trainer)
        self.datamodule = datamodule
        self._validate_modality_configuration()

    def _validate_modality_configuration(self) -> None:
        generated_modalities = set(self.noise_scheduler.target_modalities)
        if not generated_modalities:
            raise ValueError("noise_scheduler.target_modalities must contain at least one modality")

        decoder_modalities = set(getattr(self.model.moe_decoder, "modality_names", []))
        missing_decoder = generated_modalities - decoder_modalities
        if missing_decoder:
            raise ValueError(
                "Generated modalities must be covered by MoE experts. "
                f"Missing experts for: {sorted(missing_decoder)}"
            )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer custom tensorclass batch to device.
        
        This hook is called by Lightning to move the batch to the correct device.
        We need it because our batch is a pair of FullData tensorclasses,
        not a simple dict or tensor.
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            target, context = batch
            return target.to(device=device), context.to(device=device)
        return batch.to(device=device)

    def get_train_losses(self, batch):
        """Compute training losses for a batch.
        
        Loss is computed in original data space:
            - video: [B, T, 2, 4, 96, 160]
            - audio_*: [B, T, 15, 128]
            - key_press: [B, T, 10, 16]
            - mouse_movement: [B, T, 20, 2]
        """
        if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
            raise RuntimeError("Expected batch to be (target, context) tuple")

        # 1. Unpack batch. (target is our clean x_0 data)
        target, context = batch

        # 2. Add noise directly on raw target FullData.
        x_tau, tau, noise = self.noise_scheduler(target)
        
        # 4. Get the stable loss target by passing the clean target (x_0)
        trgt_v_full = self.noise_scheduler.get_loss_target(eps=noise, x_0=target) 

        # 4. Forward pass with raw x_tau/context (sequence prep happens inside model).
        pred_v_full = self.model(x_tau=x_tau, tau=tau, context=context)

        # 5. Loss loop over target modalities in raw FullData space.
        total_loss = torch.tensor(0.0, device=target.device, dtype=torch.float32)
        per_mod_losses = {}
        for mod_name in self.noise_scheduler.target_modalities:
            pred_v = pred_v_full.get_modality(mod_name)
            trgt_v = trgt_v_full.get_modality(mod_name)

            if pred_v is None or trgt_v is None:
                raise ValueError(
                    "Missing generated modality tensor during loss computation. "
                    f"modality={mod_name}, pred_is_none={pred_v is None}, target_is_none={trgt_v is None}"
                )

            # Keep loss math in fp32 for numerical stability under mixed precision.
            valid_pred_fp32 = pred_v.float()
            valid_trgt_fp32 = trgt_v.float()
            
            mod_loss = F.mse_loss(valid_pred_fp32, valid_trgt_fp32)
            per_mod_losses[mod_name] = mod_loss
            total_loss += self.modality_loss_weights.get(mod_name, 1.0) * mod_loss

        return {"total_loss": total_loss, **per_mod_losses}


    def training_step(self, batch, batch_idx):
        losses = self.get_train_losses(batch)
        log_data = {f"train/{k}": v for k, v in losses.items()}
        
        # sync_dist=False prevents blocking all_reduce operations on every forward/backward pass.
        self.log_dict(log_data, on_step=True, on_epoch=False, sync_dist=False)
        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.get_train_losses(batch)
        log_data = {f"val/{k}": v for k, v in losses.items()}
        
        # sync_dist=True is retained here as it is mathematically required for accurate global validation.
        self.log_dict(log_data, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)
        return losses["total_loss"]

    def _build_lr_scheduler(self, optimizer):
        """Build learning rate scheduler based on config."""
        sched = self.hparams.lr_scheduler.lower()
        if sched in ('none', 'constant'):
            return None

        total_steps = int(self.hparams.lr_total_steps)
        warmup_steps = int(self.hparams.lr_warmup_steps) if self.hparams.lr_warmup_steps > 0 \
                    else int(self.hparams.lr_warmup_pct * total_steps)
        warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))
        min_ratio = float(self.hparams.lr_min_ratio)

        def with_warmup(body, w):
            def f(step):
                if w > 0 and step < w:
                    return float(step + 1) / float(w)
                p = max(0., (step - w) / max(1., total_steps - w))
                return body(p)
            return f

        if sched == 'cosine':
            def body(p):
                return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * p))
            lr_lambda = with_warmup(body, w=0)

        elif sched == 'cosine_warmup':
            def body(p):
                return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * p))
            lr_lambda = with_warmup(body, w=warmup_steps)

        elif sched == 'linear_warmup':
            def body(p):
                return max(min_ratio, 1.0 - p * (1.0 - min_ratio))
            lr_lambda = with_warmup(body, w=warmup_steps)

        elif sched == 'poly':
            power = float(self.hparams.lr_poly_power)
            def body(p):
                return min_ratio + (1 - min_ratio) * (1 - p) ** power
            lr_lambda = with_warmup(body, w=warmup_steps)

        elif sched == 'exponential':
            gamma = float(self.hparams.lr_exp_gamma)
            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                val = gamma ** (step - warmup_steps)
                return max(min_ratio, val)
        else:
            return None

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": f"lr/{sched}",
        }

    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler."""
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay
        
        # Check the actual strategy class rather than a string
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            # Inspect the DeepSpeed config dictionary attached to the strategy
            ds_config = self.trainer.strategy.config
            zero_opt = ds_config.get("zero_optimization", {})
            
            if "offload_optimizer" in zero_opt:
                opt = DeepSpeedCPUAdam(self.model.parameters(), lr=lr, weight_decay=wd)
            else:
                opt = FusedAdam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
            
        sch_cfg = self._build_lr_scheduler(opt)
        if sch_cfg is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": sch_cfg}

    def on_train_start(self):
        """Log number of parameters at training start."""
        self.log("num_parameters", float(sum(p.numel() for p in self.model.parameters())))

    def on_train_epoch_start(self):
        """Update sampler epoch for proper shuffling in distributed training."""
        if self.trainer.datamodule is not None and hasattr(self.trainer.datamodule, "train_sampler"):
            if self.trainer.datamodule.train_sampler is not None:
                self.trainer.datamodule.train_sampler.set_epoch(self.current_epoch)

    

