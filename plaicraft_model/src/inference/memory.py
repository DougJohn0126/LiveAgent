from __future__ import annotations

import torch

from data.data_classes import FullData
from models.plai_v1 import PlaiV1Model


class MemoryContext:
    """Encapsulates STM/LTM tracking for autoregressive generation."""

    def __init__(
        self,
        model: PlaiV1Model,
        stm_len: int,
        k_chunk: int,
        warmup_ltm_chunk_length: int,
    ):
        self.model = model
        self.stm_len = int(stm_len)
        self.k = int(k_chunk)
        self.warmup_ltm_chunk_length = int(warmup_ltm_chunk_length)
        self.model_dim = int(self.model.h_dim)

        if self.k <= 0:
            raise ValueError(f"k_chunk must be >= 1, got {self.k}")
        if self.stm_len < 0:
            raise ValueError(f"stm_len must be >= 0, got {self.stm_len}")
        if self.warmup_ltm_chunk_length <= 0:
            raise ValueError(
                "warmup_ltm_chunk_length must be >= 1, "
                f"got {self.warmup_ltm_chunk_length}"
            )
        if self.warmup_ltm_chunk_length % self.k != 0:
            raise ValueError(
                "warmup_ltm_chunk_length must be divisible by k_chunk "
                f"({self.k}), got {self.warmup_ltm_chunk_length}"
            )

        self.cached_ltm: torch.Tensor | None = None
        self.processed_ltm_frames = 0

    def reset(self) -> None:
        self.cached_ltm = None
        self.processed_ltm_frames = 0
        self.model.context_embedder.enable_streaming_cache(reset_cache=True)

    @torch.no_grad()
    def update_and_get_memory(self, sequence: FullData) -> dict[str, torch.Tensor]:
        """Consume current sequence and return unified memory dictionary."""
        current_len = self._time_length(sequence)
        valid_ltm_len = (current_len // self.k) * self.k

        if valid_ltm_len > self.processed_ltm_frames:
            for chunk_start in range(
                self.processed_ltm_frames,
                valid_ltm_len,
                self.warmup_ltm_chunk_length,
            ):
                chunk_end = min(chunk_start + self.warmup_ltm_chunk_length, valid_ltm_len)
                self._chunk_process(sequence, chunk_start, chunk_end)
            self.processed_ltm_frames = valid_ltm_len

        stm_start = max(0, current_len - self.stm_len)
        stm_source = FullData.slice_time(sequence, stm_start, current_len)
        stm_dict = self.model.multimodal_io.fulldata_to_context_embedder_input(stm_source)
        stm_dict = self._make_stm_only_context(stm_dict)

        if self.cached_ltm is None:
            output = self.model.context_embedder(stm_dict)
            self.cached_ltm = output["ltm"]
        else:
            output = self.model.context_embedder(stm_dict, ltm_override=self.cached_ltm)

        return {
            "stm": output["stm"],
            "ltm": output["ltm"],
            "stm_time": output.get("stm_time", None),
            "stm_rope_pos": output.get("stm_rope_pos", None),
        }

    def _chunk_process(self, shifted_sequence: FullData, start: int, end: int) -> None:
        ltm_source = FullData.slice_time(shifted_sequence, start, end)
        context_dict = self.model.multimodal_io.fulldata_to_context_embedder_input(ltm_source)
        context_dict = self._make_ltm_only_context(context_dict)
        output = self.model.context_embedder(context_dict)
        self.cached_ltm = output["ltm"]

    @staticmethod
    def _time_length(fd: FullData) -> int:
        if fd.dataframe_indices is None:
            raise ValueError("Expected dataframe_indices in FullData.")
        return int(fd.dataframe_indices.shape[1])

    def _make_stm_only_context(self, context_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        stm_tokens = context_dict["stm_tokens"]
        batch_size = int(stm_tokens.shape[0])
        stm_dtype = stm_tokens.dtype
        device = stm_tokens.device

        context_dict["ltm_tokens"] = torch.zeros(
            (batch_size, 0, 0, self.model_dim),
            device=device,
            dtype=stm_dtype,
        )
        return context_dict

    def _make_ltm_only_context(self, context_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ltm_tokens = context_dict["ltm_tokens"]
        batch_size = int(ltm_tokens.shape[0])
        ltm_dtype = ltm_tokens.dtype
        device = ltm_tokens.device

        context_dict["stm_tokens"] = torch.zeros(
            (batch_size, 0, 0, self.model_dim),
            device=device,
            dtype=ltm_dtype,
        )
        context_dict["stm_flat_tokens"] = torch.zeros(
            (batch_size, 0, self.model_dim),
            device=device,
            dtype=ltm_dtype,
        )
        context_dict["stm_dataframe_indices"] = torch.zeros(
            (batch_size, 0),
            device=device,
            dtype=torch.float32,
        )
        return context_dict
