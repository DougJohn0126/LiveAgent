import json
import logging
import os
import pathlib
import re
import tempfile
from typing import Callable, List, Optional

import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from utils.checkpoint_zip import zip_checkpoint_path

log = logging.getLogger(__name__)
_STEP_RE = re.compile(r"[^\d]step[=_-]?(\d+)[^\d]?")


class CustomWandbLogger(WandbLogger):
    """W&B logger with project-specific semantic evaluation artifact ingestion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._artifact_hooks: List[Callable[..., None]] = []
        self._val_metric_defined: bool = False
        self._missing_model_collection_logged: bool = False
        self._missing_val_collection_logged: bool = False

    def add_on_checkpoint_artifact(self, fn: Callable[..., None]) -> None:
        self._artifact_hooks.append(fn)

    def _infer_step_from_path(self, path: str) -> Optional[int]:
        m = _STEP_RE.search(str(path))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint, *, trainer=None) -> None:
        if not self._log_model:
            return

        should_log_now = (
            self._log_model == "all"
            or (self._log_model is True and checkpoint_callback.save_top_k == -1)
        )

        if not should_log_now:
            if self._log_model is True:
                self._checkpoint_callback = checkpoint_callback
            self._ingest_semantic_evaluation_artifacts(self.experiment)
            return

        run = self.experiment
        self._log_zipped_checkpoint_artifact(run, checkpoint_callback)
        self._prune_model_artifacts(run, checkpoint_callback)

        train_step: Optional[int] = None
        train_epoch: Optional[int] = None
        if trainer is not None:
            try:
                train_step = int(getattr(trainer, "global_step"))
            except Exception:
                train_step = None
            try:
                train_epoch = int(getattr(trainer, "current_epoch"))
            except Exception:
                train_epoch = None

        if train_step is None:
            last_path = getattr(checkpoint_callback, "last_model_path", None)
            if last_path:
                train_step = self._infer_step_from_path(last_path)

        family = f"model-{run.id}-last"
        artifact_ref = f"{run.entity}/{run.project}/model-{run.id}:latest"
        aliases = ("latest",)

        for fn in self._artifact_hooks:
            try:
                fn(
                    run=run,
                    family=family,
                    aliases=aliases,
                    artifact_ref=artifact_ref,
                    train_step=train_step,
                    train_epoch=train_epoch,
                )
            except Exception as e:
                log.warning("artifact hook failed: %s", e)

        self._ingest_semantic_evaluation_artifacts(run)

    def _log_zipped_checkpoint_artifact(self, run, checkpoint_callback: ModelCheckpoint) -> None:
        ckpt_path = getattr(checkpoint_callback, "last_model_path", None) or getattr(checkpoint_callback, "best_model_path", None)
        if not isinstance(ckpt_path, str) or not ckpt_path:
            return

        if not os.path.exists(ckpt_path):
            return

        try:
            zipped = zip_checkpoint_path(ckpt_path)
            artifact = wandb.Artifact(
                name=f"model-{run.id}",
                type="model",
                metadata={
                    "checkpoint_path": ckpt_path,
                    "original_filename": pathlib.Path(ckpt_path).name,
                    "is_zipped": True,
                },
            )
            artifact.add_file(zipped, name=pathlib.Path(zipped).name)

            aliases = ["latest", "last"]
            step = self._infer_step_from_path(ckpt_path)
            if step is not None:
                aliases.append(f"step-{step}")

            best_model_path = getattr(checkpoint_callback, "best_model_path", None)
            if isinstance(best_model_path, str) and best_model_path == ckpt_path:
                aliases.append("best")

            run.log_artifact(artifact, aliases=aliases)
        except Exception as e:
            log.warning("W&B: failed to zip+upload checkpoint artifact from %s: %s", ckpt_path, e)

    def _artifact_aliases(self, artifact) -> set[str]:
        aliases = set()
        for alias in list(getattr(artifact, "aliases", []) or []):
            if isinstance(alias, str):
                aliases.add(alias)
            else:
                name = getattr(alias, "name", None)
                if isinstance(name, str):
                    aliases.add(name)
        return aliases

    def _artifact_version_num(self, artifact) -> int:
        name = str(getattr(artifact, "name", ""))
        m = re.search(r":v(\d+)$", name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return -1
        version = getattr(artifact, "version", None)
        if isinstance(version, str):
            m2 = re.search(r"v(\d+)$", version)
            if m2:
                try:
                    return int(m2.group(1))
                except Exception:
                    return -1
        return -1

    def _artifact_score(self, artifact) -> Optional[float]:
        md = getattr(artifact, "metadata", {}) or {}
        for key in ("score", "best_model_score", "monitor_value", "monitor"):
            if key in md:
                try:
                    return float(md[key])
                except Exception:
                    continue
        return None

    def _artifact_checkpoint_basename(self, artifact) -> Optional[str]:
        md = getattr(artifact, "metadata", {}) or {}
        for key in ("original_filename", "checkpoint_path", "path", "filename"):
            value = md.get(key)
            if isinstance(value, str) and value:
                return pathlib.Path(value).name
        return None

    def _prune_model_artifacts(self, run, checkpoint_callback: ModelCheckpoint) -> None:
        try:
            save_top_k = int(getattr(checkpoint_callback, "save_top_k", 0) or 0)
        except Exception:
            save_top_k = 0
        save_last = bool(getattr(checkpoint_callback, "save_last", False))

        if save_top_k == -1:
            return
        if save_top_k <= 0 and not save_last:
            return

        try:
            api = wandb.Api()
            coll_name = f"{run.entity}/{run.project}/model-{run.id}"
            try:
                coll = api.artifact_collection(type_name="model", name=coll_name)
            except Exception:
                if not self._missing_model_collection_logged:
                    log.info("W&B: model artifact collection not found yet: %s", coll_name)
                    self._missing_model_collection_logged = True
                return

            artifacts = list(coll.artifacts())
            if not artifacts:
                return

            keep_versions: set[str] = set()

            if save_last:
                latest_candidates = [a for a in artifacts if "latest" in self._artifact_aliases(a)]
                if latest_candidates:
                    latest_candidates.sort(key=self._artifact_version_num, reverse=True)
                    latest_name = str(getattr(latest_candidates[0], "name", ""))
                    if latest_name:
                        keep_versions.add(latest_name)

            if save_top_k > 0:
                mode = str(getattr(checkpoint_callback, "mode", "min") or "min").lower()
                topk_to_keep = []
                best_k_models = getattr(checkpoint_callback, "best_k_models", {}) or {}
                desired_ckpt_names = {
                    pathlib.Path(str(path)).name
                    for path in best_k_models.keys()
                    if isinstance(path, str) and path
                }

                if desired_ckpt_names:
                    topk_to_keep = [
                        art for art in artifacts if self._artifact_checkpoint_basename(art) in desired_ckpt_names
                    ]

                if not topk_to_keep:
                    scored = []
                    for art in artifacts:
                        score = self._artifact_score(art)
                        if score is None:
                            continue
                        scored.append((score, self._artifact_version_num(art), art))

                    if scored:
                        if mode == "max":
                            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        else:
                            scored.sort(key=lambda x: (x[0], -x[1]))
                        topk_to_keep = [art for _, _, art in scored[:save_top_k]]
                    else:
                        best_candidates = [a for a in artifacts if "best" in self._artifact_aliases(a)]
                        best_candidates.sort(key=self._artifact_version_num, reverse=True)
                        topk_to_keep = best_candidates[:save_top_k]

                if len(topk_to_keep) < save_top_k:
                    missing = save_top_k - len(topk_to_keep)
                    existing = {str(getattr(a, "name", "")) for a in topk_to_keep}
                    if save_last:
                        existing |= keep_versions
                    newest = sorted(artifacts, key=self._artifact_version_num, reverse=True)
                    for art in newest:
                        n = str(getattr(art, "name", ""))
                        if not n or n in existing:
                            continue
                        topk_to_keep.append(art)
                        existing.add(n)
                        missing -= 1
                        if missing <= 0:
                            break

                for art in topk_to_keep:
                    n = str(getattr(art, "name", ""))
                    if n:
                        keep_versions.add(n)

            deleted = 0
            for art in artifacts:
                n = str(getattr(art, "name", ""))
                if not n or n in keep_versions:
                    continue
                try:
                    art.delete(delete_aliases=True)
                    deleted += 1
                except TypeError:
                    art.delete()
                    deleted += 1
                except Exception as e:
                    log.warning("W&B: failed to delete model artifact %s: %s", n, e)

            if deleted:
                log.info(
                    "W&B: model artifact retention pruned %d old version(s); kept %d version(s).",
                    deleted,
                    len(keep_versions),
                )
        except Exception as e:
            log.warning("W&B: model artifact retention encountered an error: %s", e)

    def _ensure_val_metric_defined(self, run) -> None:
        if self._val_metric_defined:
            return
        if not getattr(run, "define_metric", None):
            raise RuntimeError("W&B run does not support define_metric; cannot bind custom axis.")
        run.define_metric("val/step", summary="max", hidden=True)
        self._val_metric_defined = True

    def _ensure_val_series_bound(self, run, keys) -> None:
        if not hasattr(self, "_val_bound_keys"):
            self._val_bound_keys = set()

        for k in list(keys):
            if k in self._val_bound_keys:
                continue
            run.define_metric(name=k, step_metric="val/step")
            self._val_bound_keys.add(k)

    def _ingest_semantic_evaluation_artifacts(self, run) -> None:
        try:
            api = wandb.Api()
            coll_name = f"{run.entity}/{run.project}/val-samples-{run.id}"
            try:
                coll = api.artifact_collection(type_name="val", name=coll_name)
            except Exception:
                if not self._missing_val_collection_logged:
                    log.info("W&B: semantic evaluation artifact collection not found: %s", coll_name)
                    self._missing_val_collection_logged = True
                return

            prev_max = -1
            try:
                v = run.summary.get("val_ingested_step_max")
                prev_max = int(v) if v is not None else -1
            except Exception:
                prev_max = -1
            if str(os.getenv("FORCE_REINGEST_ALL", "0")).strip().lower() not in ("", "0", "false"):
                prev_max = -1

            arts = list(coll.artifacts())
            if not arts:
                return

            def _alias_step(a):
                for al in getattr(a, "aliases", []):
                    nm = al if isinstance(al, str) else getattr(al, "name", None)
                    if isinstance(nm, str) and nm.startswith("step-"):
                        try:
                            return int(nm.split("-", 1)[1])
                        except Exception:
                            pass
                return None

            work = []
            for a in arts:
                s = _alias_step(a)
                if s is not None:
                    work.append((s, a))
            work.sort(key=lambda t: t[0])

            new_items = [(s, a) for (s, a) in work if s > prev_max]
            if not new_items:
                return

            self._ensure_val_metric_defined(run)

            cache_root = pathlib.Path(tempfile.gettempdir()) / f"wandb_val_ingest_{os.getpid()}"
            cache_root.mkdir(parents=True, exist_ok=True)

            log.info("W&B: ingesting %d semantic evaluation artifact(s) (prev_max=%d).", len(new_items), prev_max)
            ingested_max = prev_max
            for step, art in new_items:
                try:
                    base = art.name.split(":", 1)[0]
                    run.use_artifact(f"{art.entity}/{art.project}/{base}:step-{int(step)}")
                except Exception as e:
                    log.warning("W&B: use_artifact failed for %s step-%d: %s", art.name, step, e)

                try:
                    local = pathlib.Path(art.download(root=str(cache_root)))
                except Exception as e:
                    log.warning("Download failed for %s step-%d: %s", art.name, step, e)
                    continue

                reports = list(local.rglob("semantic_evaluation_report.json"))
                if len(reports) != 1:
                    log.warning(
                        "Semantic evaluation artifact %s step-%d expected exactly one semantic_evaluation_report.json, found %d.",
                        art.name,
                        step,
                        len(reports),
                    )
                    continue

                try:
                    with reports[0].open("r", encoding="utf-8") as fp:
                        rep = json.load(fp)
                except Exception as e:
                    log.warning("Failed to read semantic_evaluation_report.json for %s step-%d: %s", art.name, step, e)
                    continue

                scalars = {}
                v = rep.get("video", {})
                a = rep.get("audio", {})
                km = rep.get("keypress_mouseclick", {})

                for key in ("psnr", "ssim", "lpips", "fid"):
                    if key in v:
                        scalars[f"val/video/{key}"] = float(v[key])

                s1 = a.get("speaking", {})
                s2 = a.get("hearing", {})
                if "frechet_w2v2" in s1:
                    scalars["val/audio/speaking_fad_w2v2"] = float(s1["frechet_w2v2"])
                if "frechet_w2v2" in s2:
                    scalars["val/audio/hearing_fad_w2v2"] = float(s2["frechet_w2v2"])

                if "normalized_hamming" in km:
                    scalars["val/keys_clicks/normalized_hamming"] = float(km["normalized_hamming"])
                if "accuracy" in km:
                    scalars["val/keys_clicks/accuracy"] = float(km["accuracy"])
                if "hamming_distance" in km:
                    scalars["val/keys_clicks/hamming_distance"] = float(km["hamming_distance"])
                if "total_positions" in km:
                    scalars["val/keys_clicks/total_positions"] = float(km["total_positions"])

                clips = []
                for p in local.rglob("*"):
                    if p.is_dir() and (p / "generated").is_dir() and (p / "gt").is_dir():
                        clips.append(p)

                overlays = []
                audio_speak_gen_list, audio_speak_gt_list = [], []
                audio_hear_gen_list, audio_hear_gt_list = [], []

                for base in sorted(clips, key=lambda x: str(x)):
                    overlay = base / "full_modality_overlay.mp4"
                    ai_gen = base / "generated" / "audio_speak.wav"
                    ao_gen = base / "generated" / "audio_hear.wav"
                    ai_gt = base / "gt" / "audio_speak.wav"
                    ao_gt = base / "gt" / "audio_hear.wav"

                    missing = [str(req) for req in (overlay, ai_gen, ao_gen, ai_gt, ao_gt) if not req.exists()]
                    if missing:
                        log.warning("Media skipped for %s (missing files: %s)", str(base), "; ".join(missing))
                        continue

                    try:
                        session_id = base.parent.name
                        prompt_resp = base.name
                        label = f"{session_id}_{prompt_resp}_{int(step)}"

                        overlays.append(wandb.Video(str(overlay), caption=label, format="mp4"))
                        audio_speak_gen_list.append(wandb.Audio(str(ai_gen), sample_rate=16000, caption=label))
                        audio_speak_gt_list.append(wandb.Audio(str(ai_gt), sample_rate=16000, caption=label))
                        audio_hear_gen_list.append(wandb.Audio(str(ao_gen), sample_rate=16000, caption=label))
                        audio_hear_gt_list.append(wandb.Audio(str(ao_gt), sample_rate=16000, caption=label))
                    except Exception as e:
                        log.warning("W&B media construction failed for %s: %s", str(base), e)
                        continue

                payload = {"val/step": int(step)}
                payload.update(scalars)

                media_items = {}
                if overlays:
                    media_items["val/overlay_videos"] = overlays
                if audio_speak_gen_list:
                    media_items["val/audio_speak/gen"] = audio_speak_gen_list
                if audio_speak_gt_list:
                    media_items["val/audio_speak/gt"] = audio_speak_gt_list
                if audio_hear_gen_list:
                    media_items["val/audio_hear/gen"] = audio_hear_gen_list
                if audio_hear_gt_list:
                    media_items["val/audio_hear/gt"] = audio_hear_gt_list
                payload.update(media_items)

                if len(payload) > 1:
                    keys_to_bind = [k for k in payload.keys() if k != "val/step"]
                    self._ensure_val_series_bound(run, keys_to_bind)
                    run.log(payload, step=int(step), commit=True)

                if step > ingested_max:
                    ingested_max = int(step)

            if ingested_max > prev_max:
                run.summary["val_ingested_step_max"] = int(ingested_max)

        except Exception as e:
            log.warning("Semantic evaluation ingestion encountered an error: %s", e)
