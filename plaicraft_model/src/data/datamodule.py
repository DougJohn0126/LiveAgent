"""Lightning DataModule for Pi0 plai_v1 training.

Training uses the "Mega-Read, K-Unroll" paradigm:
- DataLoader fetches a batch of B large contiguous windows from the dataset.
- A KFoldUnrollingWrapper iterates K times over that base window,
  sampling one random split index per iteration (shared across all B items).
- Each iteration yields a perfectly rectangular (target, context) pair —
  zero padding, fully valid, static shapes compatible with flex_attention.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import lightning.pytorch as pl
from typing import Optional

from data.datasets.mapstyle import MapStyleDataset
from data.datasets.semantic_evaluation import SemanticEvaluationDataset
from data.data_classes import FullData
from utils.normalization import scale_zscore, inverse_scale_zscore
from utils.constants import UNIT_DURATION_MS, VIDEO_FPS


class KFoldUnrollingWrapper:
    """Wraps a DataLoader and unrolls K (target, context) sub-batches per window (i.e., per single batch) from it.

    Each base window is a FullData of shape [B, L] (B items, L dataframes).

    A single split index is sampled uniformly from the valid range [stm_context_length, L - target_length]. This ensures the history length on the lhs of the split is at least >= stm_constext_length, while the total sequence length is <= L. Note that the full slice from 0:split_idx will be considered as 'history' at training. The resulting context target split is forwarded to the model to complete one training iteration. We will repeat the same operations (a new split) within the same base window [B, L] for the next training step, until we have completed K training steps on the same base window. 

    Since the sampled split index is reused within the batch (for all B items), the resulting batches are perfectly rectangular, therefore, no padding is required. This enables state shapes for flex_attention compilation.  

    'Relative' dataframe indices are assigned:
        context: [-split_idx, ..., -1]  
        target:  [0, 1, ..., target_length - 1]
    ,i.e., split_idx reflects the 'absolute' position of the start of the target.
    """

    def __init__(
        self,
        dataloader,
        k_subsample_size: int,
        stm_context_length: int,
        target_length: int,
        normalize_fn,
    ):
        self.dataloader = dataloader
        self.k_subsample_size = int(k_subsample_size)
        self.stm_context_length = int(stm_context_length)
        self.target_length = int(target_length)
        self.normalize_fn = normalize_fn
        # Tracks yielded train steps within the current epoch for resume.
        self._position_in_epoch = 0

    def __iter__(self):
        epoch_length = len(self)
        to_skip = int(self._position_in_epoch % epoch_length) if epoch_length > 0 else 0

        for base_batch in self.dataloader:
            base_batch = self.normalize_fn(base_batch)

            L = FullData.infer_time_length(base_batch)

            # Valid split range: context must have >= stm_context_length frames
            # and target must fit within the window.
            min_idx = self.stm_context_length
            max_idx = L - self.target_length
            if max_idx < min_idx:
                # Window too short for valid split; skip this base window.
                continue

            for _ in range(self.k_subsample_size):
                # torch.randint is seeded by Lightning's seed_everything for full
                # reproducibility across epochs and distributed workers.
                split_idx = int(torch.randint(min_idx, max_idx + 1, (1,)).item())

                if to_skip > 0:
                    to_skip -= 1
                    continue

                context_batch = FullData.slice_time(base_batch, 0, split_idx)
                target_batch = FullData.slice_time(base_batch, split_idx, split_idx + self.target_length)

                # Context is negative-relative, target is zero-based.
                context_batch = DataModule.assign_indices(context_batch, -split_idx)
                target_batch = DataModule.assign_indices(target_batch, 0)

                if epoch_length > 0:
                    self._position_in_epoch = (self._position_in_epoch + 1) % epoch_length

                yield target_batch, context_batch

    def __len__(self):
        return len(self.dataloader) * self.k_subsample_size

    def state_dict(self):
        """Serialize wrapper progress for checkpointing."""
        return {"position_in_epoch": int(self._position_in_epoch)}

    def load_state_dict(self, state_dict):
        """Restore wrapper progress from checkpoint state."""
        self._position_in_epoch = int(state_dict.get("position_in_epoch", 0))


class FixedSplitWindowWrapper:
    """Yield a deterministic (target, context) pair for each base window.

    The wrapper mirrors the training unroll path, but uses a fixed split point
    at `window_length - target_length` so validation is stable and reproducible.
    """

    def __init__(
        self,
        dataloader,
        target_length: int,
        stm_context_length: int,
        normalize_fn,
    ):
        self.dataloader = dataloader
        self.target_length = int(target_length)
        self.stm_context_length = int(stm_context_length)
        self.normalize_fn = normalize_fn
        self._position_in_epoch = 0

    def __iter__(self):
        epoch_length = len(self)
        to_skip = int(self._position_in_epoch % epoch_length) if epoch_length > 0 else 0

        for base_batch in self.dataloader:
            base_batch = self.normalize_fn(base_batch)
            L = FullData.infer_time_length(base_batch)
            split_idx = L - self.target_length
            if split_idx < self.stm_context_length:
                continue

            if to_skip > 0:
                to_skip -= 1
                continue

            context_batch = FullData.slice_time(base_batch, 0, split_idx)
            target_batch = FullData.slice_time(base_batch, split_idx, split_idx + self.target_length)

            context_batch = DataModule.assign_indices(context_batch, -split_idx)
            target_batch = DataModule.assign_indices(target_batch, 0)

            if epoch_length > 0:
                self._position_in_epoch = (self._position_in_epoch + 1) % epoch_length

            yield target_batch, context_batch

    def __len__(self):
        return len(self.dataloader)

    def state_dict(self):
        """Serialize wrapper progress for checkpointing."""
        return {"position_in_epoch": int(self._position_in_epoch)}

    def load_state_dict(self, state_dict):
        """Restore wrapper progress from checkpoint state."""
        self._position_in_epoch = int(state_dict.get("position_in_epoch", 0))


class DataModule(pl.LightningDataModule):
    """Lightning DataModule for Pi0 plai_v1 training and sampling.

    Training follows the "Mega-Read, K-Unroll" strategy:
    - DataLoader reads B contiguous windows in one pass (the Mega-Read),
    - A wrapper then produces K (target, context) sub-batches from that base window,
      sampling a single split index per sub-batch that is shared across all B items.
    - This produces perfectly rectangular sub-batches with zero padding.

    For test/sampling, it returns normalized full sequences.
    """

    def __init__(
        self,
        dataset_path: str,
        modalities: Optional[list] = None,
        window_length_frames: int = 100,
        hop_length_frames: Optional[int] = None,
        player_names: Optional[list] = None,
        training_metadata_db_path: Optional[str] = None,
        validation_metadata_db_path: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle: bool = False,
        target_length: int = 1,
        k_subsample_size: int = 10,
        stm_context_length: int = 2,
        evaluation_metadata_db_path: Optional[str] = None,
    ):
        """
        Args:
            dataset_path: Path to the dataset.
            modalities: List of modalities to load.
            window_length_frames: Number of dataframes in each base window.
            hop_length_frames: Number of dataframes to move the window at each step.
            player_names: List of player names whose data to load.
            training_metadata_db_path: Path to the training metadata database.
            batch_size: Number of base windows per DataLoader batch (B in [B, L]).
            num_workers: Number of dataloader workers.
            shuffle: Whether to shuffle the base windows.
            target_length: Length of target segment in dataframe units.
            k_subsample_size: Number of (target, context) sub-batches to unroll per base window.
            stm_context_length: Minimum context length; split index is always >= this value.
        """
        super().__init__()

        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if int(target_length) < 1:
            raise ValueError(f"target_length must be >= 1, got {target_length}.")
        if int(k_subsample_size) < 1:
            raise ValueError(f"k_subsample_size must be >= 1, got {k_subsample_size}.")
        if int(stm_context_length) < 1:
            raise ValueError(f"stm_context_length must be >= 1, got {stm_context_length}.")

        self.dataset_path = dataset_path
        self.modalities = sorted(set(modalities or ["video", "audio_hear", "audio_speak", "key_press", "mouse_movement"]))
        self.window_length_frames = window_length_frames
        self.hop_length_frames = hop_length_frames
        self.player_names = player_names
        self.training_metadata_db_path = training_metadata_db_path
        self.validation_metadata_db_path = validation_metadata_db_path
        self.batch_size = int(batch_size)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.target_length = int(target_length)
        self.k_subsample_size = int(k_subsample_size)
        self.stm_context_length = int(stm_context_length)
        self.evaluation_metadata_db_path = evaluation_metadata_db_path

        self.train_dataset = None
        self.test_dataset = None
        self.validation_dataset = None
        self.train_sampler = None
        self._train_loader_wrapper = None
        self._train_loader_state = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/testing."""
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                self.train_dataset = MapStyleDataset(
                    dataset_path=self.dataset_path,
                    modalities=self.modalities,
                    window_length_frames=self.window_length_frames,
                    hop_length_frames=self.hop_length_frames,
                    player_names=self.player_names,
                    global_database_path=self.training_metadata_db_path,
                )

        if stage == "validate" or stage is None:
            if self.validation_dataset is None:
                self.validation_dataset = MapStyleDataset(
                    dataset_path=self.dataset_path,
                    modalities=self.modalities,
                    window_length_frames=self.window_length_frames,
                    hop_length_frames=self.hop_length_frames,
                    player_names=self.player_names,
                    global_database_path=self.validation_metadata_db_path,
                )

        if stage == "test" or stage is None:
            if self.test_dataset is None:
                if self.evaluation_metadata_db_path:
                    self.test_dataset = SemanticEvaluationDataset(
                        dataset_path=self.dataset_path,
                        semantic_evaluation_db_path=self.evaluation_metadata_db_path,
                    )
                else:
                    self.test_dataset = MapStyleDataset(
                        dataset_path=self.dataset_path,
                        modalities=self.modalities,
                        window_length_frames=self.window_length_frames,
                        hop_length_frames=self.hop_length_frames,
                        player_names=self.player_names,
                        global_database_path=self.training_metadata_db_path,
                    )

    @staticmethod
    def assign_indices(fd: FullData, start_idx: int) -> FullData:
        """Centralized index injection. Assign [start_idx, start_idx + L) to FullData."""
        L = FullData.infer_time_length(fd)
        B = FullData.infer_batch_size(fd)
        device = fd.device

        fd_dict = fd.to_dict()
        fd_dict["dataframe_indices"] = (
            torch.arange(start_idx, start_idx + L, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        return FullData(batch=fd_dict)

    @staticmethod
    def _extract_window_start_ms(metadata_item):
        """Extract the window start timestamp in milliseconds from metadata."""
        frame_length_ms = 1000 // VIDEO_FPS

        if metadata_item is None:
            return 0

        if isinstance(metadata_item, list):
            if len(metadata_item) > 0 and isinstance(metadata_item[0], dict):
                start_frame = metadata_item[0].get("start_frame")
                if start_frame is None:
                    return 0
                return int(start_frame) * frame_length_ms
            return 0

        if isinstance(metadata_item, dict):
            start_frame = metadata_item.get("start_frame")
            if start_frame is None:
                return 0
            return int(start_frame) * frame_length_ms

        return 0

    @staticmethod
    def _filter_transcripts(transcripts, metadata_val, segment_start, segment_end):
        """Keep transcript entries that overlap the requested segment."""
        def _has_overlap(entry_start, entry_end, seg_start_ms, seg_end_ms):
            """Return True when transcript and segment time ranges intersect."""
            return entry_start < seg_end_ms and entry_end > seg_start_ms

        if transcripts is None:
            return None

        if not isinstance(transcripts, list):
            return transcripts

        if len(transcripts) == 0:
            return []

        looks_batched = bool(metadata_val) and len(transcripts) == len(metadata_val)
        if looks_batched:
            filtered_batch = []
            for sample_transcripts, sample_metadata in zip(transcripts, metadata_val):
                window_start_ms = DataModule._extract_window_start_ms(sample_metadata)
                seg_start_ms = window_start_ms + int(segment_start) * UNIT_DURATION_MS
                seg_end_ms = window_start_ms + int(segment_end) * UNIT_DURATION_MS

                if not isinstance(sample_transcripts, list):
                    filtered_batch.append(sample_transcripts)
                    continue

                filtered_sample = []
                for entry in sample_transcripts:
                    if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                        continue
                    entry_start = entry[1]
                    entry_end = entry[2]
                    if _has_overlap(entry_start, entry_end, seg_start_ms, seg_end_ms):
                        filtered_sample.append(entry)

                filtered_batch.append(filtered_sample)

            return filtered_batch

        seg_start_ms = int(segment_start) * UNIT_DURATION_MS
        seg_end_ms = int(segment_end) * UNIT_DURATION_MS
        filtered = []
        for entry in transcripts:
            if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                continue
            entry_start = entry[1]
            entry_end = entry[2]
            if _has_overlap(entry_start, entry_end, seg_start_ms, seg_end_ms):
                filtered.append(entry)
        return filtered

    def train_dataloader(self):
        """Return a KFoldUnrollingWrapper around the base MapStyle DataLoader.

        The base DataLoader reads `batch_size` windows per step (the Mega-Read).
        The wrapper iterates K times over each base window, sampling one random
        split index per iteration shared across all B items, and yields
        (target_batch, context_batch) pairs with zero padding.
        """
        if self.train_dataset is None:
            self.setup(stage="fit")

        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                self.train_dataset,
                shuffle=self.shuffle,
                seed=getattr(self, "seed", 42),
                drop_last=True,  # Discard incomplete batches across distributed nodes
            )
        self.train_sampler = sampler

        base_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=self.shuffle if sampler is None else None,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train_dataset.collate_fn,
            drop_last=True,  # Discard the final incomplete batch of the epoch
        )

        wrapper = KFoldUnrollingWrapper(
            dataloader=base_loader,
            k_subsample_size=self.k_subsample_size,
            stm_context_length=self.stm_context_length,
            target_length=self.target_length,
            normalize_fn=DataModule.normalize_full_data,
        )

        if self._train_loader_state:
            wrapper.load_state_dict(self._train_loader_state)

        self._train_loader_wrapper = wrapper
        return wrapper

    def test_dataloader(self):
        """Create test dataloader for offline sampling and semantic evaluation."""
        if self.test_dataset is None:
            self.setup(stage="test")

        if isinstance(self.test_dataset, SemanticEvaluationDataset):
            base_loader = DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                persistent_workers=self.num_workers > 0,
                collate_fn=self.test_dataset.collate_fn,
            )

            class SemanticEvaluationTupleWrapper:
                """Yield normalized (target, context) tuples from semantic evaluation CSV rows."""

                def __init__(self, loader):
                    self.loader = loader

                def __iter__(self):
                    for batch in self.loader:
                        if not isinstance(batch, dict) or "context" not in batch or "target" not in batch:
                            raise RuntimeError("SemanticEvaluationDataset batch must contain 'context' and 'target'.")

                        context = DataModule.normalize_full_data(batch["context"])
                        target = DataModule.normalize_full_data(batch["target"])

                        context_len = FullData.infer_time_length(context)
                        context = DataModule.assign_indices(context, -context_len)
                        target = DataModule.assign_indices(target, 0)

                        yield target, context

                def __len__(self):
                    """Return the wrapped dataloader length."""
                    return len(self.loader)

            return SemanticEvaluationTupleWrapper(base_loader)

        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                self.test_dataset,
                shuffle=self.shuffle,
                seed=getattr(self, "seed", 42),
                drop_last=False,
            )

        base_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=self.shuffle if sampler is None else None,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.test_dataset.collate_fn,
        )

        class SimpleWrapper:
            def __init__(self, loader):
                """Wrap a dataloader to support resumable iteration across epochs."""
                self.loader = loader
                self._position_in_epoch = 0

            def __iter__(self):
                """Iterate normalized batches and resume from the saved position."""
                epoch_length = len(self)
                to_skip = int(self._position_in_epoch % epoch_length) if epoch_length > 0 else 0

                for batch in self.loader:
                    if to_skip > 0:
                        to_skip -= 1
                        continue

                    batch = DataModule.normalize_full_data(batch)

                    # Continuous zero-based stream for offline/test sampling.
                    batch = DataModule.assign_indices(batch, 0)

                    if epoch_length > 0:
                        self._position_in_epoch = (self._position_in_epoch + 1) % epoch_length
                    yield batch

            def __len__(self):
                """Return the wrapped dataloader length."""
                return len(self.loader)

            def state_dict(self):
                """Serialize wrapper progress for checkpointing."""
                return {"position_in_epoch": int(self._position_in_epoch)}

            def load_state_dict(self, state_dict):
                """Restore wrapper progress from checkpoint state."""
                self._position_in_epoch = int(state_dict.get("position_in_epoch", 0))

        return SimpleWrapper(base_loader)

    def validation_dataloader(self):
        """Create a deterministic validation dataloader that yields (target, context) pairs."""
        if self.validation_dataset is None:
            self.setup(stage="validate")

        base_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            sampler=None,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.validation_dataset.collate_fn,
        )

        wrapper = FixedSplitWindowWrapper(
            dataloader=base_loader,
            target_length=self.target_length,
            stm_context_length=self.stm_context_length,
            normalize_fn=DataModule.normalize_full_data,
        )
        return wrapper

    def state_dict(self):
        """Persist training dataloader progress for resumable mid-epoch training."""
        state = {}
        if self._train_loader_wrapper is not None:
            state["train_loader"] = self._train_loader_wrapper.state_dict()
        elif self._train_loader_state is not None:
            state["train_loader"] = self._train_loader_state
        return state

    def load_state_dict(self, state_dict):
        """Restore training dataloader progress from checkpoint."""
        self._train_loader_state = state_dict.get("train_loader") if state_dict else None

    @staticmethod
    def normalize_full_data(full_data: FullData) -> FullData:
        """Apply z-score normalization to all latent modalities in FullData."""
        latent_dict = {}

        if full_data.video is not None:
            latent_dict["frame_latent"] = full_data.video
        if full_data.audio_speak is not None:
            latent_dict["audio_speak_latent"] = full_data.audio_speak
        if full_data.audio_hear is not None:
            latent_dict["audio_hear_latent"] = full_data.audio_hear
        if full_data.key_press is not None:
            latent_dict["keyboard_latent"] = full_data.key_press
        if full_data.mouse_movement is not None:
            latent_dict["mouse_latent"] = full_data.mouse_movement

        normalized_dict = scale_zscore(latent_dict)
        batch_dict = full_data.to_dict()

        if "frame_latent" in normalized_dict:
            batch_dict["video"] = normalized_dict["frame_latent"]
        if "audio_speak_latent" in normalized_dict:
            batch_dict["audio_speak"] = normalized_dict["audio_speak_latent"]
        if "audio_hear_latent" in normalized_dict:
            batch_dict["audio_hear"] = normalized_dict["audio_hear_latent"]
        if "keyboard_latent" in normalized_dict:
            batch_dict["key_press"] = normalized_dict["keyboard_latent"]
        if "mouse_latent" in normalized_dict:
            batch_dict["mouse_movement"] = normalized_dict["mouse_latent"]

        return FullData(batch=batch_dict)

    @staticmethod
    def denormalize_full_data(full_data: FullData) -> FullData:
        """Invert z-score normalization for all latent modalities in FullData."""
        latent_dict = {}

        if full_data.video is not None:
            latent_dict["frame_latent"] = full_data.video
        if full_data.audio_speak is not None:
            latent_dict["audio_speak_latent"] = full_data.audio_speak
        if full_data.audio_hear is not None:
            latent_dict["audio_hear_latent"] = full_data.audio_hear
        if full_data.key_press is not None:
            latent_dict["keyboard_latent"] = full_data.key_press
        if full_data.mouse_movement is not None:
            latent_dict["mouse_latent"] = full_data.mouse_movement

        denorm_dict = inverse_scale_zscore(latent_dict)
        batch_dict = full_data.to_dict()

        if "frame_latent" in denorm_dict:
            batch_dict["video"] = denorm_dict["frame_latent"]
        if "audio_speak_latent" in denorm_dict:
            batch_dict["audio_speak"] = denorm_dict["audio_speak_latent"]
        if "audio_hear_latent" in denorm_dict:
            batch_dict["audio_hear"] = denorm_dict["audio_hear_latent"]
        if "keyboard_latent" in denorm_dict:
            batch_dict["key_press"] = denorm_dict["keyboard_latent"]
        if "mouse_latent" in denorm_dict:
            batch_dict["mouse_movement"] = denorm_dict["mouse_latent"]

        return FullData(batch=batch_dict)

    @staticmethod
    def add_command_line_options(parser):
        """Add datamodule-specific command line options."""
        MapStyleDataset.add_command_line_options(parser)
        parser.add_argument("--batch_size", type=int, default=8, help="Number of slices sampled per base window.")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
        parser.add_argument("--shuffle", type=int, default=0, help="Whether to shuffle training data")
        parser.add_argument("--target_length", type=int, default=1, help="Target sequence length for diffusion target")
        return parser