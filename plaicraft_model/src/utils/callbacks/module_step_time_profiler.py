from collections import defaultdict
from time import perf_counter

import lightning.pytorch as pl
import torch


class ModuleStepTimeProfiler(pl.Callback):
    """Profiles per-component times by aggregating leaf submodule forward/backward timings."""

    def __init__(
        self,
        enabled: bool = True,
        log_to_progress_bar: bool = True,
        synchronize_cuda_timing: bool = True,
        print_summary_every_n_steps: int = 0,
    ):
        super().__init__()
        self.enabled = enabled
        self.log_to_progress_bar = log_to_progress_bar
        self.synchronize_cuda_timing = synchronize_cuda_timing
        self.print_summary_every_n_steps = int(print_summary_every_n_steps)
        self._handles = []
        self._component_names = []
        self._component_short_names = {}
        self._fwd_stacks = defaultdict(list)  # keyed by leaf key
        self._bwd_stacks = defaultdict(list)  # keyed by leaf key
        self._fwd_accum_ms = defaultdict(float)
        self._bwd_accum_ms = defaultdict(float)
        self._fwd_cuda_pairs = defaultdict(list)
        self._bwd_cuda_pairs = defaultdict(list)
        self._backward_step_t0 = None

    def _should_profile(self, trainer) -> bool:
        return self.enabled and trainer.training

    def _reset_step_state(self):
        self._fwd_stacks.clear()
        self._bwd_stacks.clear()
        self._fwd_accum_ms.clear()
        self._bwd_accum_ms.clear()
        self._fwd_cuda_pairs.clear()
        self._bwd_cuda_pairs.clear()
        self._backward_step_t0 = None

    def _iter_leaf_modules(self, root_module):
        for _, m in root_module.named_modules():
            if len(list(m.children())) == 0:
                yield m

    def _start_stamp(self, module):
        if torch.cuda.is_available() and isinstance(module, torch.nn.Module):
            p = next(module.parameters(), None)
            if p is not None and p.is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream(p.device))
                return ("cuda", p.device, start_event)
        return ("cpu", perf_counter())

    def _stop_stamp(self, root_name, start_stamp):
        kind = start_stamp[0]
        if kind == "cuda":
            _, device, start_event = start_stamp
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(device))
            return ("cuda", root_name, start_event, end_event)
        _, t0 = start_stamp
        return ("cpu", root_name, (perf_counter() - t0) * 1000.0)

    def _shorten_component_name(self, name: str) -> str:
        aliases = {
            "context_embedder": "ctx",
            "moe_decoder": "moe",
            "multimodal_io": "mmio",
            "timestep_embedder": "tstep",
            "model": "model",
        }
        return aliases.get(name, name[:8])

    def _forward_pre_hook(self, key, root_name):
        def _hook(module, _inputs):
            self._fwd_stacks[key].append(self._start_stamp(module))

        return _hook

    def _forward_hook(self, key, root_name):
        def _hook(module, _inputs, _output):
            if not self._fwd_stacks[key]:
                return
            stamp = self._fwd_stacks[key].pop()
            stop = self._stop_stamp(root_name, stamp)
            if stop[0] == "cpu":
                _, root, ms = stop
                self._fwd_accum_ms[root] += ms
            else:
                _, root, start_event, end_event = stop
                self._fwd_cuda_pairs[root].append((start_event, end_event))

        return _hook

    def _backward_pre_hook(self, key, root_name):
        def _hook(module, _grad_output):
            self._bwd_stacks[key].append(self._start_stamp(module))

        return _hook

    def _backward_hook(self, key, root_name):
        def _hook(module, _grad_input, _grad_output):
            if not self._bwd_stacks[key]:
                return
            stamp = self._bwd_stacks[key].pop()
            stop = self._stop_stamp(root_name, stamp)
            if stop[0] == "cpu":
                _, root, ms = stop
                self._bwd_accum_ms[root] += ms
            else:
                _, root, start_event, end_event = stop
                self._bwd_cuda_pairs[root].append((start_event, end_event))

        return _hook

    def on_fit_start(self, trainer, pl_module):
        if not self.enabled:
            return

        model = getattr(pl_module, "model", None)
        if model is None:
            return

        children = list(model.named_children())
        if not children:
            children = [("model", model)]

        self._component_names = [name for name, _ in children]
        self._component_short_names = {
            name: self._shorten_component_name(name) for name in self._component_names
        }
        seen_ids = set()
        for root_name, child in children:
            for idx, leaf in enumerate(self._iter_leaf_modules(child)):
                leaf_id = id(leaf)
                if leaf_id in seen_ids:
                    continue
                seen_ids.add(leaf_id)
                key = f"{root_name}:{idx}:{leaf_id}"
                self._handles.append(
                    leaf.register_forward_pre_hook(self._forward_pre_hook(key, root_name))
                )
                self._handles.append(leaf.register_forward_hook(self._forward_hook(key, root_name)))
                self._handles.append(
                    leaf.register_full_backward_pre_hook(self._backward_pre_hook(key, root_name))
                )
                self._handles.append(
                    leaf.register_full_backward_hook(self._backward_hook(key, root_name))
                )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._should_profile(trainer):
            return
        self._reset_step_state()

    def on_before_backward(self, trainer, pl_module, loss):
        if not self._should_profile(trainer):
            return
        if self.synchronize_cuda_timing and torch.cuda.is_available() and pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        self._backward_step_t0 = perf_counter()

    def on_after_backward(self, trainer, pl_module):
        if not self._should_profile(trainer):
            return
        if self._backward_step_t0 is None:
            return
        if self.synchronize_cuda_timing and torch.cuda.is_available() and pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        bwd_step_ms = (perf_counter() - self._backward_step_t0) * 1000.0
        pl_module.log(
            "perf/bwd_ms_step",
            bwd_step_ms,
            on_step=True,
            on_epoch=False,
            prog_bar=self.log_to_progress_bar,
            logger=True,
            sync_dist=False,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._should_profile(trainer):
            return

        if self.synchronize_cuda_timing and torch.cuda.is_available() and pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)

        for name, pairs in self._fwd_cuda_pairs.items():
            for start_event, end_event in pairs:
                self._fwd_accum_ms[name] += start_event.elapsed_time(end_event)
        for name, pairs in self._bwd_cuda_pairs.items():
            for start_event, end_event in pairs:
                self._bwd_accum_ms[name] += start_event.elapsed_time(end_event)

        total_fwd = 0.0
        total_bwd = 0.0
        summary_parts = []
        for name in self._component_names:
            fwd_ms = float(self._fwd_accum_ms.get(name, 0.0))
            bwd_ms = float(self._bwd_accum_ms.get(name, 0.0))
            total_fwd += fwd_ms
            total_bwd += bwd_ms
            short = self._component_short_names.get(name, name)
            summary_parts.append(f"{short}:f={fwd_ms:.1f}/b={bwd_ms:.1f}")

            pl_module.log(
                f"perf/fwd_ms/{name}",
                fwd_ms,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
            pl_module.log(
                f"perf/bwd_ms/{name}",
                bwd_ms,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

            pl_module.log(
                f"perf/fwd/{short}",
                fwd_ms,
                on_step=True,
                on_epoch=False,
                prog_bar=self.log_to_progress_bar,
                logger=False,
                sync_dist=False,
            )
            pl_module.log(
                f"perf/bwd/{short}",
                bwd_ms,
                on_step=True,
                on_epoch=False,
                prog_bar=self.log_to_progress_bar,
                logger=False,
                sync_dist=False,
            )

        pl_module.log(
            "perf/fwd_ms_total",
            total_fwd,
            on_step=True,
            on_epoch=False,
            prog_bar=self.log_to_progress_bar,
            logger=True,
            sync_dist=False,
        )
        pl_module.log(
            "perf/bwd_ms_total",
            total_bwd,
            on_step=True,
            on_epoch=False,
            prog_bar=self.log_to_progress_bar,
            logger=True,
            sync_dist=False,
        )

        if (
            self.print_summary_every_n_steps > 0
            and trainer.global_rank == 0
            and (trainer.global_step + 1) % self.print_summary_every_n_steps == 0
        ):
            pl_module.print(
                "[PERF] "
                + " | ".join(summary_parts)
                + f" | total_f={total_fwd:.1f} total_b={total_bwd:.1f}"
            )

    def on_fit_end(self, trainer, pl_module):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._component_names = []
        self._component_short_names = {}
        self._reset_step_state()
