from __future__ import annotations

import time
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.strategies import DeepSpeedStrategy


class DeepSpeedFlopsEstimator(pl.Callback):
    """Estimate model FLOPs for training steps when using DeepSpeed.

    This callback profiles selected train batches with DeepSpeed's FLOPs profiler,
    logs measured forward-pass FLOPs/MACs, and reports a common training FLOPs
    estimate of `3 * forward_flops` (forward + backward + optimizer/update).
    """

    def __init__(
        self,
        enabled: bool = False,
        start_step: int = 10,
        every_n_steps: int = 0,
        max_profiled_steps: int = 1,
        log_to_progress_bar: bool = True,
        sync_dist: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self.start_step = int(start_step)
        self.every_n_steps = int(every_n_steps)
        self.max_profiled_steps = int(max_profiled_steps)
        self.log_to_progress_bar = log_to_progress_bar
        self.sync_dist = sync_dist

        self._profiler = None
        self._profiler_available = False
        self._profiling_active = False
        self._step_t0: Optional[float] = None
        self._num_profiled = 0
        self._warned_not_deepspeed = False

    def _is_deepspeed(self, trainer) -> bool:
        return isinstance(trainer.strategy, DeepSpeedStrategy)

    def _should_profile_step(self, trainer) -> bool:
        if not self.enabled or not trainer.training:
            return False
        if not self._profiler_available:
            return False
        if self.max_profiled_steps > 0 and self._num_profiled >= self.max_profiled_steps:
            return False

        step = int(trainer.global_step)
        if step < self.start_step:
            return False
        if self.every_n_steps <= 0:
            return step == self.start_step
        return (step - self.start_step) % self.every_n_steps == 0

    def _get_profiled_module(self, pl_module):
        return getattr(pl_module, "model", pl_module)

    def _call_numeric(self, fn) -> Optional[float]:
        try:
            val = fn(as_string=False)
        except TypeError:
            val = fn()
        except Exception:
            return None

        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def on_fit_start(self, trainer, pl_module):
        if not self.enabled:
            return

        if not self._is_deepspeed(trainer):
            if not self._warned_not_deepspeed and trainer.global_rank == 0:
                pl_module.print(
                    "[DeepSpeedFlopsEstimator] Disabled profiling because trainer strategy is not DeepSpeed."
                )
                self._warned_not_deepspeed = True
            return

        try:
            from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
        except Exception:
            self._profiler_available = False
            if trainer.global_rank == 0:
                pl_module.print(
                    "[DeepSpeedFlopsEstimator] Could not import DeepSpeed FlopsProfiler; skipping FLOPs estimation."
                )
            return

        profiled_module = self._get_profiled_module(pl_module)
        self._profiler = FlopsProfiler(profiled_module)
        self._profiler_available = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._should_profile_step(trainer):
            return

        try:
            self._profiler.start_profile(ignore_list=None)
            self._profiling_active = True
            self._step_t0 = time.perf_counter()
        except Exception:
            self._profiling_active = False
            self._step_t0 = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._profiling_active:
            return

        try:
            self._profiler.stop_profile()

            fwd_flops = self._call_numeric(self._profiler.get_total_flops)
            fwd_macs = self._call_numeric(self._profiler.get_total_macs)
            params = self._call_numeric(self._profiler.get_total_params)

            step_seconds = None
            if self._step_t0 is not None:
                step_seconds = max(1e-9, time.perf_counter() - self._step_t0)

            if fwd_flops is not None:
                pl_module.log(
                    "perf/flops_fwd",
                    fwd_flops,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=self.log_to_progress_bar,
                    logger=True,
                    sync_dist=self.sync_dist,
                )

                train_flops_est = 3.0 * fwd_flops
                pl_module.log(
                    "perf/flops_train_est",
                    train_flops_est,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=self.log_to_progress_bar,
                    logger=True,
                    sync_dist=self.sync_dist,
                )

                if step_seconds is not None:
                    pl_module.log(
                        "perf/tflops_train_est_per_s",
                        train_flops_est / step_seconds / 1e12,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=self.log_to_progress_bar,
                        logger=True,
                        sync_dist=self.sync_dist,
                    )

            if fwd_macs is not None:
                pl_module.log(
                    "perf/macs_fwd",
                    fwd_macs,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=self.sync_dist,
                )

            if params is not None:
                pl_module.log(
                    "perf/params_profiled",
                    params,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=self.sync_dist,
                )

        finally:
            try:
                self._profiler.end_profile()
            except Exception:
                pass

            self._profiling_active = False
            self._step_t0 = None
            self._num_profiled += 1

            if trainer.global_rank == 0:
                pl_module.print(
                    f"[DeepSpeedFlopsEstimator] Profiled step {trainer.global_step}: "
                    "logged perf/flops_fwd and perf/flops_train_est (3x forward FLOPs)."
                )
