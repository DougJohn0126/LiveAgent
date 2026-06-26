import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path
import os
import logging
import tempfile

from omegaconf import DictConfig
import torch
import wandb

from utils import pylogger, rich_utils
from utils.checkpoint_zip import unzip_checkpoint_archive
from utils.deepspeed_zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


# ============================================================================
# Checkpoint utilities (moved from common.py)
# ============================================================================

def summarize(data, indent: int = 0):
    pad = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"{pad}{k}:")
            summarize(v, indent + 1)
    elif isinstance(data, (tuple, list)):
        for i, v in enumerate(data):
            print(f"{pad}[{i}]:")
            summarize(v, indent + 1)
    elif torch.is_tensor(data):
        print(f"{pad}{tuple(data.shape)}  {data.dtype}")
    elif data is None:
        print(f"{pad}None")
    else:
        print(f"{pad}{type(data)}")


def _is_deepspeed_checkpoint_dir(p: Path) -> bool:
    """
    A DeepSpeed ZeRO checkpoint folder typically contains:
      - 'checkpoint/' subfolder with many *_optim_states.pt shards
      - 'latest' file with the tag of the latest step
    """
    if not p.is_dir():
        return False

    if (p / "checkpoint").exists() and (p / "latest").exists():
        return True

    has_model_states = any(p.glob("**/*_model_states.pt"))
    has_optim_states = any(p.glob("**/*_optim_states.pt"))
    return has_model_states and has_optim_states


def _find_deepspeed_checkpoint_dir(base_dir: Path) -> Optional[Path]:
    candidates = [base_dir, base_dir.parent]

    for candidate in candidates:
        if _is_deepspeed_checkpoint_dir(candidate):
            return candidate

    for subdir in sorted([p for p in base_dir.rglob("*") if p.is_dir()]):
        if _is_deepspeed_checkpoint_dir(subdir):
            return subdir

    return None


def _normalize_state_dict_keys(sd: dict) -> dict:
    """
    Strip common wrappers added by Lightning/DDP/DeepSpeed so the keys match
    the bare model used at sampling time. Order matters.
    """
    def strip_prefix(k: str) -> str:
        for pre in ("module.model.", "model.", "module."):
            if k.startswith(pre):
                return k[len(pre):]
        return k
    return {strip_prefix(k): v for k, v in sd.items()}


def _cache_root(redownload: bool) -> Optional[str]:
    """
    Stable cache root unless redownload=True.
    Priority:
      1) $WANDB_ARTIFACT_DIR if set
      2) ./wandb_cache/downloads
      3) None (let caller decide) when redownload=True
    """
    if redownload:
        return None
    env_root = os.environ.get("WANDB_ARTIFACT_DIR", "").strip()
    if env_root:
        Path(env_root).mkdir(parents=True, exist_ok=True)
        return env_root
    local = Path.cwd() / "wandb_cache" / "downloads"
    local.mkdir(parents=True, exist_ok=True)
    return str(local)


def _find_ckpt_in_dir(base_dir: Path, checkpoint_filename: Optional[str]) -> str:
    """
    Return absolute path to a .ckpt inside base_dir.
    - If checkpoint_filename is given, first try 'checkpoints/<filename>', then anywhere.
    - Otherwise pick the newest *.ckpt, preferring 'checkpoints/' subtree if present.
    """
    if checkpoint_filename:
        cand = base_dir / "checkpoints" / checkpoint_filename
        if cand.exists():
            return str(cand.resolve())
        matches = list(base_dir.glob(f"**/{checkpoint_filename}"))
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(matches[0].resolve())
        raise FileNotFoundError(f"[ckpt] '{checkpoint_filename}' not found under '{base_dir}'")

    search_roots = []
    ck_dir = base_dir / "checkpoints"
    if ck_dir.exists():
        search_roots.append(ck_dir)
    search_roots.append(base_dir)

    for root in search_roots:
        cks = sorted(root.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cks:
            return str(cks[0].resolve())

    raise FileNotFoundError(f"[ckpt] No .ckpt found under '{base_dir}'")


def _unzip_downloaded_artifact(base_dir: Path) -> Path:
    zip_files = sorted(base_dir.glob("**/*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(
            f"[ckpt] Expected a zipped checkpoint artifact under '{base_dir}', but found no .zip files"
        )

    unzip_errors = []
    for zip_file in zip_files:
        try:
            extracted_path = Path(unzip_checkpoint_archive(str(zip_file)))
            return extracted_path
        except Exception as ex:
            unzip_errors.append(f"{zip_file}: {ex}")

    details = " | ".join(unzip_errors)
    raise RuntimeError(f"[ckpt] Found zip files but failed to extract any of them. Details: {details}")


def resolve_wandb_artifact_ckpt(
    artifact_ref: str,
    checkpoint_filename: Optional[str] = "last.ckpt",
    redownload_checkpoints: bool = False,
    allow_universal_dir: bool = False,
) -> str:
    """
    Download a W&B artifact like 'entity/project/model-XXXX-best-1:v0' into
    '<cache_root>/<artifact.name>:<artifact.version>/' and return a usable path:

      - If the artifact root contains a DeepSpeed checkpoint (has 'checkpoint/' and 'latest'),
        return that DIRECTORY path.
      - Otherwise, return a concrete .ckpt FILE path inside that directory
        (prefer 'checkpoints/<checkpoint_filename>' when provided).

    The cache_root resolves to:
      • $WANDB_ARTIFACT_DIR  (if set), or
      • ./wandb_cache/downloads
    A unique temp root is used only when redownload_checkpoints=True.
    """
    if not artifact_ref or ":" not in artifact_ref or artifact_ref.count("/") < 2:
        raise ValueError(f"[ckpt] Expected artifact ref 'entity/project/name:alias', got '{artifact_ref}'")

    api = wandb.Api()
    art = api.artifact(artifact_ref)

    # choose base root
    if redownload_checkpoints:
        base_root = Path(tempfile.mkdtemp(prefix="wandb_cache_"))
    else:
        base_root_str = _cache_root(False)
        if not base_root_str:
            raise RuntimeError("[ckpt] Could not determine cache root")
        base_root = Path(base_root_str)

    # deterministic subfolder: "<name>:<version>"
    dest = (base_root / f"{art.name}:{art.version}").resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # download into that subfolder; W&B returns the directory it used
    local_dir = Path(art.download(root=str(dest)))
    if not local_dir.exists():
        raise FileNotFoundError(f"[ckpt] Artifact downloaded path does not exist: {local_dir}")

    # Normalize: some SDK versions return parent; prefer our 'dest' if populated
    try:
        if local_dir.resolve() != dest and any(dest.iterdir()):
            local_dir = dest
    except Exception:
        pass

    resolved_root = _unzip_downloaded_artifact(local_dir)

    # DeepSpeed directory in extracted artifact tree
    deepspeed_dir = _find_deepspeed_checkpoint_dir(resolved_root)
    if deepspeed_dir is not None:
        return str(deepspeed_dir.resolve())

    # Otherwise, find a .ckpt file (Lightning)
    try:
        return _find_ckpt_in_dir(resolved_root, checkpoint_filename)
    except FileNotFoundError:
        if allow_universal_dir and resolved_root.is_dir():
            return str(resolved_root.resolve())
        raise


# Thin wrapper to support any existing callers that use this name.
def resolve_wandb_artifact_path(
    artifact_path: str,
    redownload_checkpoints: bool = False,
    allow_universal_dir: bool = False,
) -> str:
    """
    Treat 'artifact_path' as a W&B ARTIFACT REF and return the concrete path.
    """
    return resolve_wandb_artifact_ckpt(
        artifact_ref=artifact_path,
        checkpoint_filename=None,
        redownload_checkpoints=redownload_checkpoints,
        allow_universal_dir=allow_universal_dir,
    )


def resolve_resume_checkpoint(
    wandb_artifact_path: Optional[str],
    ckpt_path: Optional[str],
    allow_universal_dir: bool = False,
) -> Optional[str]:
    """
    Decide which checkpoint to resume from, downloading if needed.
    Preference: local ckpt_path > wandb_artifact_path. Returns a concrete path or None.
    """
    ckpt_log = logging.getLogger(__name__)
    ckpt_log.info(
        "[ckpt] Resume request: ckpt_path=%s wandb_artifact_path=%s",
        ckpt_path,
        wandb_artifact_path,
    )

    if ckpt_path:
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"[ckpt] --ckpt_path does not exist: {ckpt_path}")
        if p.is_dir():
            if not _is_deepspeed_checkpoint_dir(p):
                if allow_universal_dir:
                    ckpt_log.info("[ckpt] Using local DeepSpeed universal checkpoint dir: %s", ckpt_path)
                    return str(p.resolve())
                raise RuntimeError(f"[ckpt] --ckpt_path is a directory but not a DeepSpeed checkpoint: {ckpt_path}")
            ckpt_log.info("[ckpt] Using local DeepSpeed checkpoint dir: %s", ckpt_path)
            return str(p.resolve())
        if p.suffix != ".ckpt":
            raise RuntimeError(f"[ckpt] --ckpt_path must be a .ckpt file or DeepSpeed dir: {ckpt_path}")
        ckpt_log.info("[ckpt] Using local Lightning checkpoint file: %s", ckpt_path)
        return str(p.resolve())

    if wandb_artifact_path:
        path = resolve_wandb_artifact_ckpt(
            wandb_artifact_path,
            checkpoint_filename=None,
            redownload_checkpoints=False,
            allow_universal_dir=allow_universal_dir,
        )
        ckpt_log.info("[ckpt] Resolved W&B checkpoint path: %s", path)
        return path

    ckpt_log.info("[ckpt] No checkpoint provided → fresh start.")
    return None


def load_fp32_state_dict_from_any(ckpt_path: str) -> dict:
    """
    Load a model state_dict from either:
      - A DeepSpeed ZeRO checkpoint directory (converted to fp32), or
      - A Lightning .ckpt file, or
      - An artifact directory that contains a .ckpt file.
    """
    ckpt_log = logging.getLogger(__name__)
    p = Path(ckpt_path)

    try:
        if p.is_dir():
            if _is_deepspeed_checkpoint_dir(p):
                ckpt_log.info("[ckpt] Loading DeepSpeed fp32 state_dict from dir: %s", p)
                sd = get_fp32_state_dict_from_zero_checkpoint(str(p))
            else:
                target = _find_ckpt_in_dir(p, None)
                ckpt_log.info("[ckpt] Loading Lightning checkpoint file: %s", target)
                loaded = torch.load(target, map_location="cpu")
                sd = loaded.get("state_dict", loaded)
        else:
            if p.suffix != ".ckpt":
                raise RuntimeError(f"[ckpt] Expected a .ckpt file, got: {p}")
            ckpt_log.info("[ckpt] Loading Lightning checkpoint file: %s", p)
            loaded = torch.load(str(p), map_location="cpu")
            sd = loaded.get("state_dict", loaded)

        sd = _normalize_state_dict_keys(sd)

        num_tensors = len(sd)
        num_params = 0
        for v in sd.values():
            try:
                num_params += int(v.numel())
            except Exception:
                pass
        if num_tensors == 0:
            raise RuntimeError("[ckpt] Loaded an empty state_dict!")
        ckpt_log.info("[ckpt] Loaded state_dict: %d tensors, ~%d parameters", num_tensors, num_params)
        return sd

    except Exception as e:
        ckpt_log.error("[ckpt] Failed to load state_dict from %s: %s", ckpt_path, e)
        raise
