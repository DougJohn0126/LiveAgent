#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import re

import hydra
import rootutils
import lightning as L
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils.metrics_computation import validate as validate_metrics
from utils.metrics_computation_db import validate_db_driven
from utils import RankedLogger, extras
from utils.utils import resolve_wandb_artifact_path
from models.components.ema import LitEma

log = RankedLogger(__name__, rank_zero_only=True)


def _resolve_model_dtype(cfg: DictConfig) -> torch.dtype:
    precision = str(cfg.get("model_precision", "bf16")).lower()
    precision_to_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if precision not in precision_to_dtype:
        raise ValueError(f"Unsupported model_precision '{precision}'. Use bf16, fp16, or fp32.")
    return precision_to_dtype[precision]


def _resolve_checkpoint_path(cfg: DictConfig) -> str:
    if cfg.get("wandb_artifact_path"):
        log.info("Loading model from W&B checkpoint artifact")
        return resolve_wandb_artifact_path(
            artifact_path=cfg.wandb_artifact_path,
            redownload_checkpoints=bool(cfg.get("redownload_checkpoints", False)),
        )
    if cfg.get("ckpt_path"):
        log.info("Loading model from local checkpoint")
        return str(cfg.ckpt_path)
    raise ValueError("One of 'wandb_artifact_path' or 'ckpt_path' must be provided.")


def _save_semantic_evaluation_metadata(out_path: Path, datamodule: LightningDataModule) -> None:
    samples_dir = out_path / "samples"
    if not samples_dir.is_dir():
        log.warning("Samples directory not found: %s", samples_dir)
        return

    dataset = getattr(datamodule, "test_dataset", None)
    if dataset is None:
        log.warning("Datamodule has no test_dataset; cannot save semantic evaluation metadata.")
        return

    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
            continue

        try:
            sample_idx = int(sample_dir.name.replace("sample_", ""))
        except ValueError:
            continue

        if sample_idx >= len(dataset):
            continue

        item = dataset[sample_idx]
        target_fd = item["target"] if isinstance(item, dict) else item
        metadata = getattr(target_fd, "metadata", None)
        metadata = getattr(metadata, "tolist", lambda: metadata)()

        if not metadata:
            continue

        if isinstance(metadata, list) and metadata and isinstance(metadata[0], dict):
            metadata = metadata[0]

        metadata_path = sample_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def _find_ema_state_in_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any] | None:
    callbacks_state = checkpoint.get("callbacks", {})
    if not isinstance(callbacks_state, dict):
        return None

    for callback_state in callbacks_state.values():
        if isinstance(callback_state, dict) and isinstance(callback_state.get("ema_state"), dict):
            return callback_state["ema_state"]
    return None


def _load_model_weights(model: torch.nn.Module, raw_state_dict: Dict[str, Any]) -> None:
    """Load model weights with a standard, minimal key normalization path."""
    state_dict = raw_state_dict
    prefixes = ("module.model.", "model.", "module.", "_orig_mod.")

    for prefix in prefixes:
        stripped = {
            k[len(prefix):]: v
            for k, v in raw_state_dict.items()
            if isinstance(k, str) and k.startswith(prefix)
        }
        if stripped:
            state_dict = stripped
            break

    model_state = model.state_dict()
    filtered_state_dict: Dict[str, Any] = {}
    unexpected: list[str] = []
    shape_mismatch: list[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if hasattr(value, "shape") and model_state[key].shape != value.shape:
            shape_mismatch.append(key)
            continue
        filtered_state_dict[key] = value

    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    remaining_unexpected = list(getattr(incompatible, "unexpected_keys", []))

    def _extract_expert_modalities(keys: list[str]) -> set[str]:
        pattern = re.compile(r"(?:^|\.)experts\.([^\.]+)\.")
        modalities: set[str] = set()
        for key in keys:
            match = pattern.search(key)
            if match:
                modalities.add(match.group(1))
        return modalities

    ckpt_experts = _extract_expert_modalities(list(state_dict.keys()))
    model_experts = _extract_expert_modalities(list(model_state.keys()))
    missing_experts = sorted(model_experts - ckpt_experts)
    ignored_experts = sorted(ckpt_experts - model_experts)

    if missing:
        log.warning("Missing keys while loading checkpoint (%d).", len(missing))
    if unexpected:
        log.warning("Ignored checkpoint keys not present in model (%d).", len(unexpected))
    if shape_mismatch:
        log.warning("Ignored checkpoint keys with shape mismatch (%d).", len(shape_mismatch))
    if remaining_unexpected:
        log.warning("Unexpected keys while loading checkpoint (%d).", len(remaining_unexpected))
    if missing_experts:
        log.warning("Checkpoint missing MoE experts for modalities: %s", missing_experts)
    if ignored_experts:
        log.warning("Checkpoint has extra MoE experts not used by model: %s", ignored_experts)


def _load_checkpoint_state_dict(
    ckpt_path: str,
    *,
    is_deepspeed_zero_checkpoint: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    p = Path(ckpt_path)

    if is_deepspeed_zero_checkpoint:
        if not p.is_dir():
            raise RuntimeError(
                "is_deepspeed_zero_checkpoint=True requires ckpt_path to be a DeepSpeed checkpoint directory. "
                f"Got: {p}"
            )
        log.info("Detected DeepSpeed sharded checkpoint directory; converting to fp32 state dict...")
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(p))
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(f"No state_dict recovered from DeepSpeed checkpoint directory: {p}")
        return state_dict, None

    if p.is_dir():
        raise RuntimeError(
            "ckpt_path points to a directory but is_deepspeed_zero_checkpoint is False. "
            "Set is_deepspeed_zero_checkpoint=true for DeepSpeed sharded checkpoints."
        )

    checkpoint = torch.load(str(p), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"No loadable state_dict found in checkpoint file: {p}")
    return state_dict, checkpoint


def run_offline_sample(cfg: DictConfig) -> Tuple[Path, LightningDataModule]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if not cfg.get("data"):
        raise ValueError("Eval config must include 'data' datamodule configuration.")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    ckpt_path = _resolve_checkpoint_path(cfg)
    is_zero_ckpt = bool(cfg.get("is_deepspeed_zero_checkpoint", False))
    log.info(f"Loading weights from {ckpt_path}")
    raw_state_dict, checkpoint_obj = _load_checkpoint_state_dict(
        ckpt_path,
        is_deepspeed_zero_checkpoint=is_zero_ckpt,
    )
    _load_model_weights(model, raw_state_dict)

    if bool(cfg.get("use_ema_weights", False)):
        if is_zero_ckpt:
            log.warning(
                "use_ema_weights=True but checkpoint is DeepSpeed ZeRO directory; "
                "EMA callback state is unavailable in this format. Using base weights."
            )
            ema_state = None
        else:
            ema_state = _find_ema_state_in_checkpoint(checkpoint_obj or {})
        if ema_state is None:
            log.warning("use_ema_weights=True but no EMA callback state was found in checkpoint; using base model weights.")
        else:
            ema = LitEma(model=model, decay=0.9999)
            ema.load_state_dict(ema_state, strict=False)
            ema.copy_to(model)
            log.info("Loaded EMA weights from callback state in checkpoint.")

    model.eval()

    device = torch.device(cfg.get("device", "cuda"))
    model_dtype = _resolve_model_dtype(cfg)
    log.info(f"Moving model to device={device} with precision={cfg.get('model_precision', 'bf16')}")
    model.to(device=device, dtype=model_dtype)

    sampler = hydra.utils.instantiate(
        cfg.inference,
        cfg=cfg,
        model=model,
        datamodule=datamodule,
        _recursive_=False,
    )
    out_path = Path(sampler.run())
    log.info(f"Sampling complete. Output directory: {out_path}")
    return out_path, datamodule


def run_eval(cfg: DictConfig) -> Tuple[Optional[Dict[str, Any]], Path]:
    """Unified evaluation entrypoint.

        Modes:
            - offline_sample: run sampling only
            - semantic_evaluation: run sampling + metrics (database-driven or full)
            - validation: run validation-window sampling + metrics

    Args:
        cfg: Evaluation configuration
    """
    mode = str(cfg.eval.mode)
    if mode not in ("offline_sample", "semantic_evaluation", "validation"):
        raise ValueError(
            f"Unsupported eval.mode '{mode}'. Use 'offline_sample', 'semantic_evaluation', or 'validation'."
        )

    if mode in ("semantic_evaluation", "validation"):
        # Semantic evaluation metrics require decoded generated/gt clip folders on disk.
        with open_dict(cfg):
            cfg.inference.decode = True
            cfg.inference.store_decoded_generated = True
            cfg.inference.store_decoded_gt = True

    out_path, datamodule = run_offline_sample(cfg)

    if mode == "offline_sample":
        return None, out_path

    log.info("Computing %s metrics...", mode)

    compute_full_metrics = bool(cfg.get("eval", {}).get("compute_full_metrics", False))

    if mode == "semantic_evaluation":
        _save_semantic_evaluation_metadata(out_path, datamodule)

        db_path = cfg.get("data", {}).get("semantic_evaluation_db_path", None)
        if not db_path:
            raise ValueError(
                "semantic_evaluation requires data.semantic_evaluation_db_path to be set."
            )

        report = validate_db_driven(
            sample_root=out_path,
            db_path=Path(db_path),
            compute_full_metrics=compute_full_metrics,
        )
    else:
        # Validation mode keeps legacy full-metrics behavior.
        report = validate_metrics(out_path)
    
    log.info("Metrics computation complete.")

    report_name = f"{mode}_report.json"
    out_json = out_path / report_name
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Evaluation report written to: {out_json}")
    
    return report, out_path


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    report, out_path = run_eval(cfg)
    mode = str(cfg.eval.mode)
    if mode == "offline_sample":
        log.info(f"Offline sampling complete. Output: {out_path}")
    elif mode == "validation":
        keys = sorted((report or {}).keys())
        log.info(f"Validation complete. Output: {out_path}. Report keys: {keys}")
    else:
        keys = sorted((report or {}).keys())
        log.info(f"Semantic evaluation complete. Output: {out_path}. Report keys: {keys}")


if __name__ == "__main__":
    main()
