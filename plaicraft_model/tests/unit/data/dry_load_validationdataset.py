"""Dry-load a single batch using ValidationDataset on real data.

Run:
    python tests/unit/data/dry_load_validationdataset.py

Paths can be set via environment variables (.env file):
- DATASET_PATH: Path to the processed dataset
- VALIDATION_METADATA_DB_PATH: Path to the validation metadata database

The script decodes the target window fully, but only decodes the last few
frames of the context window to keep very long prompts manageable.
"""

from datetime import datetime
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
import torch

# Ensure src/ and project root are on the path when running the script directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.data_classes import FullData
from data.datamodule import DataModule
from data.datasets.validation import ValidationDataset
from inference.decode import decode_and_save

# Load environment variables from .env file if it exists
load_dotenv()

DATASET_PATH = Path(os.getenv("DATASET_PATH", ""))
VALIDATION_METADATA_DB_PATH = Path(os.getenv("VALIDATION_METADATA_DB_PATH", ""))


def _print_fulldata(name, data):
    """Print FullData structure."""
    if data is None:
        print(f"{name}: None")
        return
    if not isinstance(data, FullData):
        print(f"{name}: {type(data)}")
        return

    print(f"{name} (FullData):")
    if data.video is not None:
        print(f"  video: shape={tuple(data.video.shape)}, dtype={data.video.dtype}")
    if data.audio_speak is not None:
        print(f"  audio_speak: shape={tuple(data.audio_speak.shape)}, dtype={data.audio_speak.dtype}")
    if data.audio_hear is not None:
        print(f"  audio_hear: shape={tuple(data.audio_hear.shape)}, dtype={data.audio_hear.dtype}")
    if data.key_press is not None:
        print(f"  key_press: shape={tuple(data.key_press.shape)}, dtype={data.key_press.dtype}")
    if data.mouse_movement is not None:
        print(f"  mouse_movement: shape={tuple(data.mouse_movement.shape)}, dtype={data.mouse_movement.dtype}")

    def _print_transcript_field(field_name, transcript_data):
        if transcript_data is None:
            print(f"  {field_name}: None")
            return

        if not isinstance(transcript_data, list):
            print(f"  {field_name}: {transcript_data}")
            return

        print(f"  {field_name}:")
        if len(transcript_data) > 0 and isinstance(transcript_data[0], list):
            for batch_idx, sample_transcript in enumerate(transcript_data):
                print(f"    sample[{batch_idx}] ({len(sample_transcript)} entries):")
                if len(sample_transcript) == 0:
                    print("      (empty)")
                    continue
                for entry in sample_transcript:
                    if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                        print(f"      - {entry[0]} [{entry[1]}ms -> {entry[2]}ms]")
                    else:
                        print(f"      - {entry}")
            return

        print(f"    ({len(transcript_data)} entries)")
        if len(transcript_data) == 0:
            print("    (empty)")
            return
        for entry in transcript_data:
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                print(f"    - {entry[0]} [{entry[1]}ms -> {entry[2]}ms]")
            else:
                print(f"    - {entry}")

    _print_transcript_field("transcript_speak", data.transcript_speak)
    _print_transcript_field("transcript_hear", data.transcript_hear)


def _print_window_summary(name, data):
    """Print a short summary for a batched FullData window."""
    if data is None:
        print(f"{name}: None")
        return

    if not isinstance(data, FullData):
        print(f"{name}: {type(data)}")
        return

    print(f"{name} time length: {FullData.infer_time_length(data)}")
    _print_fulldata(name, data)


def _normalize_for_decode(data: FullData) -> FullData:
    """Match the DataModule test-loader path before calling decode_and_save."""
    return DataModule.normalize_full_data(data)


def _decode_context_tail(context: FullData, tail_frames: int) -> tuple[FullData, int]:
    """Keep only the last tail_frames from a context window."""
    context_len = FullData.infer_time_length(context)
    if context_len <= 0:
        return context, 0

    tail_len = min(int(tail_frames), context_len)
    tail_start = context_len - tail_len
    return FullData.slice_time(context, tail_start, context_len), tail_len


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset path not found: {DATASET_PATH}")
        return
    if not VALIDATION_METADATA_DB_PATH.exists():
        print(f"Validation DB path not found: {VALIDATION_METADATA_DB_PATH}")
        return

    ds = ValidationDataset(
        dataset_path=str(DATASET_PATH),
        validation_db_path=str(VALIDATION_METADATA_DB_PATH),
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        collate_fn=ds.collate_fn,
        persistent_workers=True,
    )

    context_tail_frames = int(os.getenv("VALIDATION_CONTEXT_TAIL_FRAMES", "10"))

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("logs") / "decoded_examples" / f"validation_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading ValidationDataset batches to {base_output_dir} ===\n")
    print(f"Context tail decode length: {context_tail_frames} frames")

    for batch_idx, batch in enumerate(dl):

        if not isinstance(batch, dict) or "context" not in batch or "target" not in batch:
            print(f"Batch {batch_idx}: unexpected batch structure, skipping")
            continue

        target = batch["target"]
        context = batch["context"]
        context_tail, tail_len = _decode_context_tail(context, context_tail_frames)
        target_for_decode = _normalize_for_decode(target)
        context_tail_for_decode = _normalize_for_decode(context_tail)

        batch_dir = base_output_dir / f"batch_{batch_idx + 1}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing batch {batch_idx + 1}...")
        _print_window_summary("target", target)
        _print_window_summary("context_tail", context_tail)

        try:
            decode_and_save(
                pred_fd=target_for_decode,
                gt_fd=target_for_decode,
                out_dir=batch_dir / "target_decode",
                window_length=FullData.infer_time_length(target),
                video_filename="target.mp4",
                make_audio_plots=False,
                store_decoded_generated=False,
                store_decoded_gt=False,
                metadata=target_for_decode.metadata,
            )
        except Exception as e:
            print(f"  Error decoding target for batch {batch_idx + 1}: {e}")

        try:
            decode_and_save(
                pred_fd=context_tail_for_decode,
                gt_fd=context_tail_for_decode,
                out_dir=batch_dir / "context_tail_decode",
                window_length=tail_len,
                video_filename="context_tail.mp4",
                make_audio_plots=False,
                store_decoded_generated=False,
                store_decoded_gt=False,
                metadata=context_tail_for_decode.metadata,
            )
        except Exception as e:
            print(f"  Error decoding context tail for batch {batch_idx + 1}: {e}")

        print(f"  Batch {batch_idx + 1} complete\n")

    print(f"\n=== Decoding complete! Output saved to {base_output_dir} ===")


if __name__ == "__main__":
    main()