import lightning.pytorch as pl
from hydra import compose
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.constants import VALID_MODALITIES
from .sync_semantic_evaluation import SyncSemanticEvaluation


class SyncValidation(SyncSemanticEvaluation):
    """Run synchronous validation every N train steps using a temporary checkpoint."""

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not self.enabled:
            return

        if self.every_n_train_steps <= 0:
            if getattr(trainer, "global_rank", 0) == 0:
                print("[SyncVal] every_n_train_steps must be > 0; disabling sync validation callback.")
            self.enabled = False
            return

        datamodule = getattr(trainer, "datamodule", None)
        validation_metadata_db_path = getattr(datamodule, "validation_metadata_db_path", None)
        if not validation_metadata_db_path:
            if getattr(trainer, "global_rank", 0) == 0:
                print("[SyncVal] validation_metadata_db_path is not set; disabling sync validation callback.")
            self.enabled = False
            return

        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_callback is None:
            if getattr(trainer, "global_rank", 0) == 0:
                print("[SyncVal] ModelCheckpoint callback not found; disabling sync validation callback.")
            self.enabled = False
            return

        if not bool(getattr(checkpoint_callback, "save_last", False)):
            if getattr(trainer, "global_rank", 0) == 0:
                print("[SyncVal] ModelCheckpoint.save_last is False; enabling it is recommended for sync validation.")

    def _build_validation_cfg(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        train_step: int,
        output_dir: str,
    ) -> DictConfig:
        overrides = list(self.eval_overrides)
        overrides.extend(["data=validation", "eval.mode=validation"])
        cfg = compose(config_name=self.eval_config_name, overrides=overrides)

        runtime_train_cfg = getattr(pl_module, "_full_train_cfg", None)

        datamodule = getattr(trainer, "datamodule", None)
        dm_dataset_path = getattr(datamodule, "dataset_path", None)

        with open_dict(cfg):
            cfg.train_step = int(train_step)
            cfg.paths.output_dir = output_dir

            if dm_dataset_path is not None:
                cfg.data.dataset_path = dm_dataset_path

            if isinstance(runtime_train_cfg, DictConfig):
                if "data" in runtime_train_cfg:
                    for key in (
                        "training_metadata_db_path",
                        "validation_metadata_db_path",
                        "modalities",
                        "window_length_frames",
                        "hop_length_frames",
                        "player_names",
                        "batch_size",
                        "num_workers",
                        "shuffle",
                        "target_length",
                        "k_subsample_size",
                        "stm_context_length",
                    ):
                        value = OmegaConf.select(runtime_train_cfg, f"data.{key}", default=None)
                        if value is not None:
                            cfg.data[key] = value

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
