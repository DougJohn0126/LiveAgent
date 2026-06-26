import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
import wandb
from eval import run_eval
from hydra import compose
from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils.constants import VALID_MODALITIES


@torch._dynamo.disable
def _run_eval_with_dynamo_disabled(cfg: DictConfig):
    return run_eval(cfg)


class SyncSemanticEvaluation(pl.Callback):
    """Run synchronous semantic evaluation every N train steps using a temporary checkpoint."""

    def __init__(
        self,
        enabled: bool = False,
        eval_config_name: str = "eval",
        eval_overrides: Optional[List[str]] = None,
        every_n_train_steps: int = 10000,
        direct_scalar_logging: bool = True,
        log_media: bool = True,
        fail_on_error: bool = False,
        start_index: Optional[int] = None,
        stop_index: Optional[int] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.eval_config_name = str(eval_config_name)
        self.eval_overrides = list(eval_overrides or [])
        self.every_n_train_steps = int(every_n_train_steps)
        self.direct_scalar_logging = bool(direct_scalar_logging)
        self.log_media = bool(log_media)
        self.fail_on_error = bool(fail_on_error)
        self.start_index = None if start_index is None else int(start_index)
        self.stop_index = None if stop_index is None else int(stop_index)
        self.max_samples = None if max_samples is None else int(max_samples)
        self._running = False
        self._last_validated_step = 0

    @staticmethod
    def _save_temp_checkpoint_with_callback(trainer: "pl.Trainer", step: int) -> Path:
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        checkpoint_dir = getattr(checkpoint_callback, "dirpath", None)
        if checkpoint_dir is None:
            checkpoint_dir = str(Path(trainer.default_root_dir) / "checkpoints")
        checkpoint_dir = Path(checkpoint_dir)

        if getattr(trainer, "global_rank", 0) == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(trainer.strategy, "barrier"):
            trainer.strategy.barrier("syncval_checkpoint_dir_ready")

        ckpt_path = checkpoint_dir / f"tmp_semantic_evaluation_ckpt_step{step:010d}.ckpt"

        if getattr(trainer, "global_rank", 0) == 0 and ckpt_path.exists():
            if ckpt_path.is_dir():
                shutil.rmtree(ckpt_path, ignore_errors=True)
            else:
                ckpt_path.unlink(missing_ok=True)

        if hasattr(trainer.strategy, "barrier"):
            trainer.strategy.barrier("syncval_prev_ckpt_removed")

        # Use the public Lightning API so strategy hooks synchronize checkpoint writing.
        trainer.save_checkpoint(str(ckpt_path))

        if hasattr(trainer.strategy, "barrier"):
            trainer.strategy.barrier("syncval_save_ckpt_complete")

        if getattr(trainer, "global_rank", 0) == 0 and not ckpt_path.exists():
            raise RuntimeError(f"Temporary sync semantic_evaluation checkpoint was not created: {ckpt_path}")

        return ckpt_path

    @staticmethod
    def _remove_temp_checkpoint(path: Optional[Path]) -> None:
        if path is None:
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            return
        path.unlink(missing_ok=True)

    @staticmethod
    def _flatten_report_for_scalars(report: Dict[str, Any]) -> Dict[str, float]:
        scalars: Dict[str, float] = {}

        video = report.get("video", {}) or {}
        for key in ("psnr", "ssim", "lpips", "fid"):
            if key in video:
                scalars[f"semantic_eval/video/{key}"] = float(video[key])

        audio = report.get("audio", {}) or {}
        speaking = audio.get("speaking", {}) or {}
        hearing = audio.get("hearing", {}) or {}
        if "frechet_w2v2" in speaking:
            scalars["semantic_eval/audio/speaking_fad_w2v2"] = float(speaking["frechet_w2v2"])
        if "frechet_w2v2" in hearing:
            scalars["semantic_eval/audio/hearing_fad_w2v2"] = float(hearing["frechet_w2v2"])

        key_click = report.get("keypress_mouseclick", {}) or {}
        for key in ("normalized_hamming", "accuracy", "hamming_distance", "total_positions"):
            if key in key_click:
                scalars[f"semantic_eval/keys_clicks/{key}"] = float(key_click[key])

        mouse = report.get("mouse_movement", {}) or {}
        mouse_key_map = {
            "ide": ("ide", "IDE", "IDE_mean"),
            "ade": ("ade", "ADE", "ADE_mean"),
            "fde": ("fde", "FDE", "FDE_mean"),
            "pld": ("pld", "PLD", "PLD_mean"),
        }
        for out_key, candidates in mouse_key_map.items():
            for candidate in candidates:
                if candidate in mouse:
                    scalars[f"semantic_eval/mouse/{out_key}"] = float(mouse[candidate])
                    break

        return scalars

    def _build_semantic_evaluation_cfg(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        train_step: int,
        output_dir: str,
    ) -> DictConfig:
        overrides = list(self.eval_overrides)
        overrides.extend(["data=semantic_evaluation", "eval.mode=semantic_evaluation"])
        cfg = compose(config_name=self.eval_config_name, overrides=overrides)

        runtime_train_cfg = getattr(pl_module, "_full_train_cfg", None)

        datamodule = getattr(trainer, "datamodule", None)
        dm_dataset_path = getattr(datamodule, "dataset_path", None)
        dm_eval_db = getattr(datamodule, "evaluation_metadata_db_path", None)

        with open_dict(cfg):
            cfg.train_step = int(train_step)
            cfg.paths.output_dir = output_dir

            if dm_dataset_path is not None:
                cfg.data.dataset_path = dm_dataset_path
            if dm_eval_db is not None:
                cfg.data.semantic_evaluation_db_path = dm_eval_db

            if isinstance(runtime_train_cfg, DictConfig):
                if "model" in runtime_train_cfg:
                    cfg.model = OmegaConf.create(
                        OmegaConf.to_container(runtime_train_cfg.model, resolve=False)
                    )
                if "noise_scheduler" in runtime_train_cfg:
                    cfg.noise_scheduler = OmegaConf.create(
                        OmegaConf.to_container(runtime_train_cfg.noise_scheduler, resolve=False)
                    )
                if "module" in runtime_train_cfg:
                    cfg.module = OmegaConf.create(
                        OmegaConf.to_container(runtime_train_cfg.module, resolve=False)
                    )
                if "data" in runtime_train_cfg and "target_length" in runtime_train_cfg.data:
                    cfg.data.target_length = int(runtime_train_cfg.data.target_length)

            # Backward-compatibility for older checkpoints/configs that predate
            # model.target_modalities. Prefer context_modalities from the loaded
            # training config, then fall back to all supported modalities.
            model_target_modalities = OmegaConf.select(cfg, "model.target_modalities", default=None)
            if model_target_modalities is None:
                fallback_modalities = OmegaConf.select(cfg, "model.context_modalities", default=None)
                if fallback_modalities is None:
                    fallback_modalities = sorted(VALID_MODALITIES)
                cfg.model.target_modalities = list(fallback_modalities)

            if "inference" not in cfg:
                raise RuntimeError("Eval config must include an 'inference' section.")

            if self.max_samples is not None:
                cfg.inference.num_samples = int(self.max_samples)
            if self.start_index is not None:
                cfg.inference.start_index = int(self.start_index)
            if self.stop_index is not None:
                cfg.inference.stop_index = int(self.stop_index)

        OmegaConf.resolve(cfg)
        return cfg

    @staticmethod
    def _get_wandb_run_from_loggers(trainer: "pl.Trainer"):
        for lg in list(getattr(trainer, "loggers", []) or []):
            exp = getattr(lg, "experiment", None)
            if exp is not None and hasattr(exp, "log"):
                return exp
        return None

    def _build_media_payload(self, out_path: str, step: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"semantic_eval/step": int(step)}
        if not self.log_media:
            return payload

        root = Path(out_path)
        if not root.exists():
            return payload

        clips = []
        for p in root.rglob("*"):
            if p.is_dir() and (p / "generated").is_dir() and (p / "gt").is_dir():
                clips.append(p)

        # Always upload all produced media; restriction should be done via sampling indices.
        clips = sorted(clips, key=lambda x: str(x))
        if not clips:
            return payload

        overlays = []
        audio_speak_gen = []
        audio_speak_gt = []
        audio_hear_gen = []
        audio_hear_gt = []

        for base in clips:
            overlay = base / "full_modality_overlay.mp4"
            ai_gen = base / "generated" / "audio_speak.wav"
            ai_gt = base / "gt" / "audio_speak.wav"
            ao_gen = base / "generated" / "audio_hear.wav"
            ao_gt = base / "gt" / "audio_hear.wav"

            if overlay.exists():
                overlays.append(wandb.Video(str(overlay), caption=f"step{step}:{base.name}", format="mp4"))
            if ai_gen.exists():
                audio_speak_gen.append(
                    wandb.Audio(str(ai_gen), sample_rate=16000, caption=f"step{step}:{base.name}")
                )
            if ai_gt.exists():
                audio_speak_gt.append(
                    wandb.Audio(str(ai_gt), sample_rate=16000, caption=f"step{step}:{base.name}")
                )
            if ao_gen.exists():
                audio_hear_gen.append(
                    wandb.Audio(str(ao_gen), sample_rate=16000, caption=f"step{step}:{base.name}")
                )
            if ao_gt.exists():
                audio_hear_gt.append(
                    wandb.Audio(str(ao_gt), sample_rate=16000, caption=f"step{step}:{base.name}")
                )

        if overlays:
            payload["semantic_eval/overlay_videos"] = overlays
        if audio_speak_gen:
            payload["semantic_eval/audio_speak/gen"] = audio_speak_gen
        if audio_speak_gt:
            payload["semantic_eval/audio_speak/gt"] = audio_speak_gt
        if audio_hear_gen:
            payload["semantic_eval/audio_hear/gen"] = audio_hear_gen
        if audio_hear_gt:
            payload["semantic_eval/audio_hear/gt"] = audio_hear_gt

        return payload

    @staticmethod
    def _log_scalars_to_all_loggers(trainer: "pl.Trainer", scalars: Dict[str, float], step: int) -> None:
        if not scalars:
            return
        for lg in list(getattr(trainer, "loggers", []) or []):
            if hasattr(lg, "log_metrics"):
                lg.log_metrics(scalars, step=int(step))

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not self.enabled:
            return

        if self.every_n_train_steps <= 0:
            if getattr(trainer, "global_rank", 0) == 0:
                print("[SyncVal] every_n_train_steps must be > 0; disabling sync semantic_evaluation callback.")
            self.enabled = False
            return

        datamodule = getattr(trainer, "datamodule", None)
        evaluation_metadata_db_path = getattr(datamodule, "evaluation_metadata_db_path", None)
        if not evaluation_metadata_db_path:
            if getattr(trainer, "global_rank", 0) == 0:
                print("[SyncVal] evaluation_metadata_db_path is not set; disabling sync semantic_evaluation callback.")
            self.enabled = False
            return

        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_callback is None:
            if getattr(trainer, "global_rank", 0) == 0:
                print("[SyncVal] ModelCheckpoint callback not found; disabling sync semantic_evaluation callback.")
            self.enabled = False
            return

        if not bool(getattr(checkpoint_callback, "save_last", False)):
            if getattr(trainer, "global_rank", 0) == 0:
                print("[SyncVal] ModelCheckpoint.save_last is False; enabling it is recommended for sync semantic_evaluation.")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        if not self.enabled:
            return
        if not getattr(trainer, "training", False):
            return
        if self._running:
            return

        step = int(getattr(trainer, "global_step", 0))
        if step <= 0:
            return
        if step % self.every_n_train_steps != 0:
            return
        if self._last_validated_step == step:
            return

        temp_ckpt_path: Optional[Path] = None
        temp_output_dir: Optional[Path] = None

        self._running = True
        semantic_evaluation_exc: Optional[Exception] = None
        try:
            temp_ckpt_path = self._save_temp_checkpoint_with_callback(
                trainer=trainer,
                step=step,
            )

            tmp_base = os.getenv("TMPDIR") or tempfile.gettempdir()
            syncval_root = Path(tmp_base) / "syncval_cache"
            syncval_root.mkdir(parents=True, exist_ok=True)
            temp_output_dir = Path(
                tempfile.mkdtemp(prefix=f"step_{step:010d}_", dir=str(syncval_root))
            )
            step_output_dir = temp_output_dir / "outputs"
            step_output_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(trainer.strategy, "barrier"):
                trainer.strategy.barrier("syncval_before_semantic_evaluation")

            if getattr(trainer, "global_rank", 0) == 0:
                try:
                    cfg = self._build_semantic_evaluation_cfg(
                        trainer=trainer,
                        pl_module=pl_module,
                        train_step=step,
                        output_dir=str(step_output_dir),
                    )

                    print(f"[SyncVal] running semantic_evaluation for step={step} from ckpt={temp_ckpt_path}")

                    with open_dict(cfg):
                        cfg.ckpt_path = str(temp_ckpt_path)
                        cfg.wandb_artifact_path = None
                        cfg.is_deepspeed_zero_checkpoint = bool(temp_ckpt_path.is_dir())

                    report, out_path = _run_eval_with_dynamo_disabled(cfg)

                    if report is None:
                        raise RuntimeError("Expected semantic_evaluation report in eval.mode=semantic_evaluation, got None.")

                    if self.direct_scalar_logging:
                        scalars = self._flatten_report_for_scalars(report)
                        self._log_scalars_to_all_loggers(trainer, scalars, step=step)

                    run = self._get_wandb_run_from_loggers(trainer)
                    if run is not None:
                        media_payload = self._build_media_payload(str(out_path), step=step)
                        if len(media_payload) > 1:
                            run.log(media_payload, step=int(step), commit=True)

                    # Clear Dynamo compile caches after eager semantic_evaluation to prevent
                    # stale shape-specialized state from affecting resumed training.
                    torch._dynamo.reset()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    self._last_validated_step = step
                    print(f"[SyncVal] completed semantic_evaluation for step={step}")
                except Exception as e:
                    semantic_evaluation_exc = e

            if hasattr(trainer.strategy, "barrier"):
                trainer.strategy.barrier("syncval_after_rank0_semantic_evaluation")

            if semantic_evaluation_exc is not None:
                if getattr(trainer, "global_rank", 0) == 0:
                    print(f"[SyncVal] semantic_evaluation failed at step={step}: {semantic_evaluation_exc}")
                if self.fail_on_error and getattr(trainer, "global_rank", 0) == 0:
                    raise semantic_evaluation_exc
        except Exception as e:
            if getattr(trainer, "global_rank", 0) == 0:
                print(f"[SyncVal] semantic_evaluation failed at step={step}: {e}")
            if self.fail_on_error:
                raise
        finally:
            self._remove_temp_checkpoint(temp_ckpt_path)
            if temp_output_dir is not None and temp_output_dir.exists():
                shutil.rmtree(temp_output_dir, ignore_errors=True)
            self._running = False
