from pathlib import Path
from typing import Tuple

import torch
from omegaconf import DictConfig

from data.data_classes import FullData
from data.datamodule import DataModule
from inference.decode import decode_and_save
from inference.denoising import generate_chunk
from inference.memory import MemoryContext
from models.plai_v1 import PlaiV1Model
from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PlaiV1Sampler:
    """Offline sampler orchestrator for autoregressive/teacher-forcing inference."""

    def __init__(self, cfg: DictConfig, model: PlaiV1Model, datamodule: DataModule, **kwargs):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.datamodule = datamodule

        self.model = model.to(self.device).eval()
        eval_mode = str(self.cfg.eval.mode)
        if eval_mode == "validation" and hasattr(self.datamodule, "validation_dataloader"):
            self.dataloader = self.datamodule.validation_dataloader()
        else:
            self.dataloader = self.datamodule.test_dataloader()

        self.dataset_modalities = list(getattr(self.datamodule, "modalities", []))
        if not self.dataset_modalities:
            raise ValueError("DataModule must define non-empty 'modalities' for sampling.")

        trained_target_modalities = list(
            getattr(getattr(self.model, "multimodal_io", None), "target_modalities", [])
        )
        if not trained_target_modalities:
            trained_target_modalities = list(self.cfg.model.get("target_modalities", self.dataset_modalities))

        requested_target_modalities = self.cfg.inference.get("target_modalities", None)
        if requested_target_modalities is None:
            self.target_modalities = trained_target_modalities
        else:
            trained_set = set(trained_target_modalities)
            filtered_requested = [m for m in list(requested_target_modalities) if m in trained_set]
            dropped = [m for m in list(requested_target_modalities) if m not in trained_set]
            if dropped:
                log.warning(
                    "Ignoring inference target modalities not present in trained target set: %s",
                    dropped,
                )
            if not filtered_requested:
                raise ValueError(
                    "inference.target_modalities has no overlap with trained target modalities. "
                    f"trained={trained_target_modalities}, requested={list(requested_target_modalities)}"
                )
            self.target_modalities = filtered_requested

        # Decoupled Memory State Tracking
        self.k = self.model.context_embedder.ltm_downsample_chunk_length
        self.stm_len = self.model.context_embedder.stm_context_length
        self.warmup_ltm_chunk_length = int(self.cfg.inference.get("warmup_ltm_chunk_length", self.k * 8))

        self._validate_memory_config()

    def _validate_memory_config(self):
        if self.warmup_ltm_chunk_length <= 0 or self.warmup_ltm_chunk_length % self.k != 0:
            raise ValueError(f"Invalid warmup_ltm_chunk_length: {self.warmup_ltm_chunk_length}")

    @staticmethod
    def _time_length(fd: FullData) -> int:
        return int(fd.dataframe_indices.shape[1])

    def _semantic_evaluation_target_length(self) -> int:
        if "data" not in self.cfg or "target_length" not in self.cfg.data:
            raise RuntimeError("Semantic evaluation sampling requires cfg.data.target_length to be set.")
        target_len = int(self.cfg.data.target_length)
        if target_len < 1:
            raise ValueError(f"cfg.data.target_length must be >= 1, got {target_len}")
        return target_len

    def _retained_warmup_length(self, warmup_len: int) -> int:
        retain_cfg = self.cfg.inference.get("include_warmup_length", None)
        if retain_cfg is None:
            return warmup_len

        retained = int(retain_cfg)
        if retained < 0:
            raise ValueError(f"include_warmup_length must be >= 0, got {retained}")
        return min(retained, warmup_len)

    @staticmethod
    def _align_time_length(pred_fd: FullData, gt_fd: FullData) -> Tuple[FullData, FullData]:
        pred_len = PlaiV1Sampler._time_length(pred_fd)
        gt_len = PlaiV1Sampler._time_length(gt_fd)
        common_len = min(pred_len, gt_len)
        return (
            FullData.slice_time(pred_fd, 0, common_len),
            FullData.slice_time(gt_fd, 0, common_len),
        )

    @staticmethod
    def _merge_generated(generated: FullData, reference: FullData, modalities: list) -> FullData:
        """Merges generated modalities into the reference FullData baseline."""
        merged = reference.to_dict()
        for mod in modalities:
            merged[mod] = generated.get_modality(mod)
        return FullData(batch=merged)

    def _should_sample(self, idx: int, selected_count: int) -> bool:
        start_index = int(self.cfg.inference.start_index)
        index_hop = int(self.cfg.inference.get("index_hop", 1))
        num_samples = self.cfg.inference.get("num_samples")
        
        if idx < start_index:
            return False
        if (idx - start_index) % index_hop != 0:
            return False
        if num_samples is not None and selected_count >= int(num_samples):
            return False
        return True

    def _pad_sequence(self, sequence: FullData, required_length: int) -> FullData:
        """Pads sequence to required_length using zero-copy tensor expansion."""
        available_len = self._time_length(sequence)
        if available_len >= required_length:
            return sequence

        pad_amount = required_length - available_len
        last_frame = FullData.slice_time(sequence, available_len - 1, available_len)
        pad_dict = last_frame.to_dict()

        for k, v in pad_dict.items():
            if torch.is_tensor(v) and v.ndim >= 2:
                expand_shape = list(v.shape)
                expand_shape[1] = pad_amount
                pad_dict[k] = v.expand(*expand_shape)

        if pad_dict["dataframe_indices"] is not None:
            last_idx = pad_dict["dataframe_indices"][:, 0:1]
            increments = torch.arange(1, pad_amount + 1, device=self.device, dtype=last_idx.dtype).unsqueeze(0)
            pad_dict["dataframe_indices"] = last_idx + increments

        return FullData.cat_time([sequence, FullData(batch=pad_dict)])

    def _generate_chunk(self, context: FullData, memory_ctx: MemoryContext, target_len: int) -> FullData:
        """Core execution step for generating a single sequence block."""
        context_len = self._time_length(context)
        context = DataModule.assign_indices(context, -context_len)
        memory = memory_ctx.update_and_get_memory(context)
        return generate_chunk(
            model=self.model,
            config=self.cfg.inference,
            memory=memory,
            batch_size=FullData.infer_batch_size(context),
            target_pred_len=target_len,
            target_modalities=self.target_modalities,
            metadata=context.metadata,
        )

    def _process_semantic_evaluation_batch(self, batch: Tuple[FullData, FullData], memory_ctx: MemoryContext) -> Tuple[FullData, FullData, int]:
        gt_target, context = [b.to(self.device) for b in batch]
        target_len = self._semantic_evaluation_target_length()

        pred_raw = self._generate_chunk(context, memory_ctx, target_len)
        gt_len = self._time_length(gt_target)
        if gt_len < target_len:
            pred_raw = FullData.slice_time(pred_raw, 0, gt_len)
            gt_aligned = gt_target
        elif gt_len > target_len:
            gt_aligned = FullData.slice_time(gt_target, 0, target_len)
        else:
            gt_aligned = gt_target

        pred_fd = self._merge_generated(pred_raw, gt_aligned, self.target_modalities)
        pred_fd, gt_aligned = self._align_time_length(pred_fd, gt_aligned)

        return pred_fd, gt_aligned, 0

    def _process_generation_batch(self, sequence: FullData, memory_ctx: MemoryContext) -> Tuple[FullData, FullData, int]:
        sequence = sequence.to(self.device)
        warmup_len = int(self.cfg.inference.get("warmup_length", 1))
        chunk_len = int(self.cfg.inference.get("chunk_length", 2))

        seq_len = self._time_length(sequence)
        req_gen = self.cfg.inference.get("generation_length")
        gen_len = int(req_gen) if req_gen is not None else max(seq_len - warmup_len, 0)

        if gen_len % chunk_len > 0:
            gen_len += chunk_len - (gen_len % chunk_len)

        sequence = self._pad_sequence(sequence, warmup_len + gen_len)

        current_ctx = FullData.slice_time(sequence, 0, warmup_len)
        pred_chunks, gt_chunks = [current_ctx], [current_ctx]

        for step in range(0, gen_len, chunk_len):
            pred_raw = self._generate_chunk(current_ctx, memory_ctx, chunk_len)

            abs_start = warmup_len + step
            gt_chunk = FullData.slice_time(sequence, abs_start, abs_start + chunk_len)
            pred_chunk = self._merge_generated(pred_raw, gt_chunk, self.target_modalities)

            pred_chunks.append(pred_chunk)
            gt_chunks.append(gt_chunk)

            if self.cfg.inference.sampling_mode == "autoregressive":
                current_ctx = FullData.cat_time([current_ctx, pred_chunk])
            else:
                current_ctx = FullData.cat_time([current_ctx, gt_chunk])

        pred_full = FullData.cat_time(pred_chunks)
        gt_full = FullData.cat_time(gt_chunks)

        retained_warmup = self._retained_warmup_length(warmup_len)
        trim_start = warmup_len - retained_warmup
        if trim_start > 0:
            total_len = self._time_length(pred_full)
            pred_full = FullData.slice_time(pred_full, trim_start, total_len)
            gt_full = FullData.slice_time(gt_full, trim_start, total_len)

        return pred_full, gt_full, retained_warmup

    def _save_outputs(self, pred_fd: FullData, gt_fd: FullData, out_dir: Path, warmup_length: int, metadata: object):
        if bool(self.cfg.inference.get("decode", True)):
            decode_and_save(
                pred_fd=pred_fd,
                gt_fd=gt_fd,
                out_dir=out_dir,
                window_length=warmup_length,
                make_audio_plots=bool(self.cfg.inference.get("make_audio_plots", True)),
                store_decoded_generated=bool(self.cfg.inference.get("store_decoded_generated", False)),
                store_decoded_gt=bool(self.cfg.inference.get("store_decoded_gt", False)),
                metadata=metadata,
            )
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(pred_fd.to_dict(), out_dir / "pred_fd.pt")
            torch.save(gt_fd.to_dict(), out_dir / "gt_fd.pt")

    def run(self) -> str:
        """Main orchestrator for offline sampling."""
        num_samples = self.cfg.inference.get("num_samples")
        max_samples = int(num_samples) if num_samples is not None else None

        if max_samples is not None and max_samples < 1:
            raise ValueError(f"inference.num_samples must be >= 1 when set, got {max_samples}")

        # Important:
        # - If num_samples is set, it is the number of selected samples to generate
        #   after start_index/index_hop filtering, and it overrides stop_index.
        # - If num_samples is unset, stop_index is the exclusive dataloader index bound.
        if max_samples is not None:
            stop_index = len(self.dataloader)
        else:
            stop_index = int(self.cfg.inference.stop_index)
            stop_index = min(stop_index, len(self.dataloader))

        selected_count = 0
        memory_ctx = MemoryContext(self.model, self.stm_len, self.k, self.warmup_ltm_chunk_length)

        for idx, batch in enumerate(self.dataloader):
            if idx >= stop_index:
                break
            if max_samples is not None and selected_count >= max_samples:
                break
            if not self._should_sample(idx, selected_count):
                continue

            memory_ctx.reset()

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                pred_fd, gt_fd, warmup_len = self._process_semantic_evaluation_batch(batch, memory_ctx)
                metadata = batch[0].metadata
            elif isinstance(batch, FullData):
                pred_fd, gt_fd, warmup_len = self._process_generation_batch(batch, memory_ctx)
                metadata = batch.metadata
            else:
                raise RuntimeError("Expected test_dataloader batch to be FullData or (target, context).")

            out_dir = Path(self.cfg.paths.output_dir) / "samples" / f"sample_{idx}"
            self._save_outputs(pred_fd, gt_fd, out_dir, warmup_len, metadata)
            selected_count += 1

        return str(Path(self.cfg.paths.output_dir))