from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg.get("model")
    hparams["module"] = cfg.get("module")
    hparams["noise_scheduler"] = cfg.get("noise_scheduler")

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # log individual model component sizes if applicable
    if hasattr(model, "model"):
        backbone = model.model
        if hasattr(backbone, "context_embedder") and backbone.context_embedder is not None:
            context_embedder = backbone.context_embedder
            hparams["model/params/context_embedder"] = sum(p.numel() for p in context_embedder.parameters())

            # Support both legacy single-perceiver and new split LTM/STM perceivers.
            if hasattr(context_embedder, "ltm_perceiver") and context_embedder.ltm_perceiver is not None:
                hparams["model/params/context_embedder/ltm_perceiver"] = sum(
                    p.numel() for p in context_embedder.ltm_perceiver.parameters()
                )
            if hasattr(context_embedder, "stm_perceiver") and context_embedder.stm_perceiver is not None:
                hparams["model/params/context_embedder/stm_perceiver"] = sum(
                    p.numel() for p in context_embedder.stm_perceiver.parameters()
                )
            if hasattr(context_embedder, "perceiver") and context_embedder.perceiver is not None:
                hparams["model/params/context_embedder/perceiver"] = sum(
                    p.numel() for p in context_embedder.perceiver.parameters()
                )

            if hasattr(context_embedder, "recurrent_encoder") and context_embedder.recurrent_encoder is not None:
                hparams["model/params/context_embedder/recurrent_encoder"] = sum(
                    p.numel() for p in context_embedder.recurrent_encoder.parameters()
                )
        if hasattr(backbone, "moe_decoder") and backbone.moe_decoder is not None:
            hparams["model/params/moe_decoder"] = sum(p.numel() for p in backbone.moe_decoder.parameters())
        if hasattr(backbone, "multimodal_io") and backbone.multimodal_io is not None:
            hparams["model/params/multimodal_io"] = sum(p.numel() for p in backbone.multimodal_io.parameters())
        if hasattr(backbone, "timestep_embedder") and backbone.timestep_embedder is not None:
            hparams["model/params/timestep_embedder"] = sum(p.numel() for p in backbone.timestep_embedder.parameters())

    hparams["data"] = cfg.get("data")
    hparams["trainer"] = cfg.get("trainer")

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
