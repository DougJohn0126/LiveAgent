import subprocess
import time
from pathlib import Path

import lightning.pytorch as pl


class AsyncSemanticEvaluation(pl.Callback):
    """
    Submits a semantic_evaluation job via sbatch when a *last* checkpoint uploads.
    Passes:
      1) RUN_PATH     = entity/project/run_id
      2) ARTIFACT_REF = entity/project/model-<runid>-last:latest
      3) TRIGGER_TS   = epoch seconds at submission
      4) --train_step=<INT>  forwarded to sample.py (if known)
    """

    def __init__(
        self,
        sbatch_script: str | None = None,
        extra_args: list[str] | None = None,
        submit_host: str | None = None,
        enabled: bool = False,
    ):
        super().__init__()
        self.sbatch_script = sbatch_script
        self.extra_args = extra_args or []
        self.submit_host = submit_host
        self._enabled = enabled

    def _ssh_opts(self) -> list[str]:
        cfg = str(Path.home() / ".ssh" / "config")
        opts = [
            "-F",
            cfg,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
        ]
        print("[AsyncSemEval] ssh opts:", " ".join(opts))
        return opts

    def _stage_remote_script(
        self,
        host: str,
        local_script: str,
        run_id: str,
        trigger_ts: str,
    ) -> str | None:
        p = Path(local_script)
        if not p.is_file():
            print(f"[AsyncSemEval] local sbatch_script not found: {local_script}")
            return None

        remote_dir = f"/tmp/plaival-{run_id}-{trigger_ts}"
        mk = ["ssh"] + self._ssh_opts() + [host, "mkdir", "-p", remote_dir]
        r = subprocess.run(mk, capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stdout or r.stderr or "").strip()
            print(f"[AsyncSemEval] remote mkdir failed rc={r.returncode}: {msg}")
            return None

        remote_path = f"{remote_dir}/{p.name}"
        cp = ["scp"] + self._ssh_opts() + [str(p), f"{host}:{remote_path}"]
        r = subprocess.run(cp, capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stdout or r.stderr or "").strip()
            print(f"[AsyncSemEval] scp failed rc={r.returncode}: {msg}")
            return None

        ch = ["ssh"] + self._ssh_opts() + [host, "chmod", "+x", remote_path]
        r = subprocess.run(ch, capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stdout or r.stderr or "").strip()
            print(f"[AsyncSemEval] remote chmod failed rc={r.returncode}: {msg}")
            return None

        return remote_path

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not self._enabled:
            return
        if not self.sbatch_script:
            if getattr(trainer, "global_rank", 0) == 0:
                print(
                    "[AsyncSemEval] enabled=True but no sbatch_script provided; async semantic_evaluation disabled."
                )
            self._enabled = False
            return

        hook_logger = None
        for lg in list(getattr(trainer, "loggers", []) or []):
            if hasattr(lg, "add_on_checkpoint_artifact"):
                hook_logger = lg
                break

        if hook_logger is None:
            if getattr(trainer, "global_rank", 0) == 0:
                print("[AsyncSemEval] W&B hook not available; async semantic_evaluation disabled.")
            self._enabled = False
            return
        self._enabled = True

        def _on_upload(
            *,
            run,
            family: str,
            aliases: tuple[str, ...],
            artifact_ref: str | None = None,
            train_step: int | None = None,
            **_,
        ):
            if not family.endswith("-last"):
                return
            if getattr(trainer, "global_rank", 0) != 0:
                return

            run_path = f"{run.entity}/{run.project}/{run.id}"
            ref = artifact_ref if artifact_ref else f"{run.entity}/{run.project}/{family}:latest"
            trigger_ts = str(int(time.time()))

            # forward known training step so validator can alias correctly even if artifact lacks metadata
            forwarded = list(self.extra_args)
            if isinstance(train_step, int):
                forwarded += [f"--train_step={int(train_step)}"]

            if self.submit_host:
                staged = self._stage_remote_script(
                    self.submit_host,
                    self.sbatch_script,
                    run.id,
                    trigger_ts,
                )
                if not staged:
                    print("[AsyncSemEval] staging failed; not submitting.")
                    return
                cmd = ["ssh"] + self._ssh_opts() + [
                    self.submit_host,
                    "sbatch",
                    staged,
                    run_path,
                    ref,
                    trigger_ts,
                ] + forwarded
                where = f"ssh:{self.submit_host}"
            else:
                cmd = ["sbatch", self.sbatch_script, run_path, ref, trigger_ts] + forwarded
                where = "local"

            out = subprocess.run(cmd, check=False, capture_output=True, text=True)
            msg = (out.stdout or out.stderr or "").strip()
            if out.returncode != 0:
                print(f"[AsyncSemEval] sbatch submission failed ({where}) rc={out.returncode}: {msg}")
                return
            print(f"[AsyncSemEval] sbatch submitted ({where}) for {run_path} -> {msg}")

        hook_logger.add_on_checkpoint_artifact(_on_upload)  # type: ignore[arg-type]
