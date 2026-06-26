"""Dry-load a single batch using DataModule on real data.

Run:
    python tests/unit/data/dry_load_datamodule.py

Paths can be set via environment variables (.env file):
- DATASET_PATH: Path to the processed dataset
- TRAINING_METADATA_DB_PATH: Path to training metadata database
"""

from pathlib import Path
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import torch

# Ensure src/ and project root are on the path when running the script directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.datamodule import DataModule
from data.data_classes import FullData
from utils.constants import UNIT_DURATION_MS, VIDEO_FPS
from inference.decode import decode_and_save

# Load environment variables from .env file if it exists
load_dotenv()

DATASET_PATH = Path(os.getenv('DATASET_PATH'))
TRAINING_METADATA_DB_PATH = Path(os.getenv('TRAINING_METADATA_DB_PATH'))


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

        # Batched form: List[List[(word, start_ms, end_ms), ...]]
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

        # Flat form: List[(word, start_ms, end_ms), ...]
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


def _print_model_input(batch, batch_idx):
    """Print (target, context) batch structure."""
    if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
        print(f"Batch {batch_idx}: Not a (target, context) tuple, type={type(batch)}")
        return

    target, context = batch
    
    print(f"--- Sub-batch {batch_idx} ((target, context)) ---")

    target_indices = target.dataframe_indices if target is not None else None
    context_indices = context.dataframe_indices if context is not None else None

    if target_indices is not None or context_indices is not None:

        if target_indices is not None and target_indices.numel() > 0:
            target_len_u = int(target_indices.shape[1])
        else:
            target_len_u = 0

        if context_indices is not None and context_indices.numel() > 0:
            context_len_u = int(context_indices.shape[1])
        else:
            context_len_u = 0

        # Segment-local unit bounds inside current sampled window
        context_start_u = 0
        context_end_u = context_len_u
        target_start_u = context_len_u
        target_end_u = context_len_u + target_len_u

        # Convert to session-relative time bounds using metadata.start_frame (per sample)
        def _find_start_frame(meta):
            if isinstance(meta, dict):
                if "start_frame" in meta and meta["start_frame"] is not None:
                    return int(meta["start_frame"])
                return None

            if isinstance(meta, (list, tuple)):
                for item in meta:
                    found = _find_start_frame(item)
                    if found is not None:
                        return found

            return None

        metadata_val = target.metadata if target is not None else None
        if isinstance(metadata_val, list):
            sample_meta_list = metadata_val
        elif metadata_val is None:
            sample_meta_list = []
        else:
            sample_meta_list = [metadata_val]

        print("Relative timestep bounds (session-relative ms):")
        print(f"  context units=[{context_start_u}, {context_end_u})")
        print(f"  target  units=[{target_start_u}, {target_end_u})")

        if len(sample_meta_list) == 0:
            print("  sample metadata unavailable")
        else:
            max_samples_to_print = 4
            for sample_idx, sample_meta in enumerate(sample_meta_list[:max_samples_to_print]):
                start_frame = _find_start_frame(sample_meta)
                if start_frame is None:
                    print(f"  sample[{sample_idx}] start_frame unavailable in metadata")
                    continue

                # start_frame is in video frames (10 Hz), not in 200ms units.
                frame_length_ms = 1000 // VIDEO_FPS
                window_start_ms = start_frame * frame_length_ms
                context_start_ms = window_start_ms + context_start_u * UNIT_DURATION_MS
                context_end_ms = window_start_ms + context_end_u * UNIT_DURATION_MS
                target_start_ms = window_start_ms + target_start_u * UNIT_DURATION_MS
                target_end_ms = window_start_ms + target_end_u * UNIT_DURATION_MS

                print(
                    f"  sample[{sample_idx}] start_frame={start_frame}, "
                    f"window_start_ms={window_start_ms}"
                )
                print(
                    f"    context session_ms=[{context_start_ms}, {context_end_ms})"
                )
                print(
                    f"    target  session_ms=[{target_start_ms}, {target_end_ms})"
                )

            if len(sample_meta_list) > max_samples_to_print:
                print(
                    f"  ... ({len(sample_meta_list) - max_samples_to_print} more samples omitted)"
                )
    
    print("\nTarget segment:")
    _print_fulldata("  target", target)
    
    print("\nContext segment:")
    _print_fulldata("  context", context)
    
    print("\nRelative indices:")
    if target_indices is not None or context_indices is not None:
        if target_indices is not None:
            print(f"  target: shape={tuple(target_indices.shape)}, values={target_indices[0]}")
        if context_indices is not None:
            print(f"  context: shape={tuple(context_indices.shape)}, values={context_indices[0]}")
    else:
        print("  None")


def extract_fulldata_item(fd: FullData, item_idx: int) -> FullData:
    """Extract the item_idx-th item from a batched FullData object.
    
    Args:
        fd: Batched FullData object with batch dimension at dim=0
        item_idx: Index of the item to extract
        
    Returns:
        A new FullData object containing only the item_idx-th item (keeps batch dimension)
    """
    if fd is None:
        return None
    
    fd_dict = fd.to_dict()
    item_dict = {}
    
    for key, value in fd_dict.items():
        if torch.is_tensor(value):
            # Extract [item_idx:item_idx+1] to preserve batch dimension
            item_dict[key] = value[item_idx:item_idx+1]
        elif isinstance(value, list) and len(value) > item_idx:
            # For list modalities (metadata, transcripts), extract the specific item
            item_dict[key] = value[item_idx:item_idx+1] if len(value) > 0 else None
        else:
            item_dict[key] = value
    
    return FullData(batch=item_dict)


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset path not found: {DATASET_PATH}")
        return
    if not TRAINING_METADATA_DB_PATH.exists():
        print(f"Metadata DB path not found: {TRAINING_METADATA_DB_PATH}")
        return

    dm = DataModule(
        dataset_path=str(DATASET_PATH),
        modalities=["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"],
        window_length_frames=100,
        hop_length_frames=10,
        player_names=None,
        training_metadata_db_path=str(TRAINING_METADATA_DB_PATH),
        batch_size=8,
        num_workers=8,
        shuffle=True,
        target_length=10,
        k_subsample_size=5,
        stm_context_length=4,
    )

    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    
    # Setup output directory structure
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("logs") / "decoded_examples" / timestamp
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Loading and decoding batches to {base_output_dir} ===\n")
    
    batch_size = int(dm.batch_size)
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 10:
            break
        
        # Parse batch: (target, context)
        if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
            print(f"Batch {batch_idx}: Not a (target, context) tuple, skipping")
            continue
        
        target, context = batch
        
        # Create batch directory
        batch_dir = base_output_dir / f"batch_{batch_idx+1}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing batch {batch_idx+1}...")
        
        # Iterate through items in the batch
        for item_idx in range(batch_size):
            # Extract individual items
            target_item = extract_fulldata_item(target, item_idx)
            context_item = extract_fulldata_item(context, item_idx)
            
            # Create item directory
            item_dir = batch_dir / f"item_{item_idx+1}"
            item_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Decode and save target video
                target_output_dir = item_dir / "target_decode"
                target_output_dir.mkdir(parents=True, exist_ok=True)
                
                decode_and_save(
                    pred_fd=target_item,
                    gt_fd=target_item,
                    out_dir=target_output_dir,
                    window_length=0,
                    video_filename="target.mp4",
                    make_audio_plots=False,
                    store_decoded_generated=False,
                    store_decoded_gt=False,
                    metadata=target_item.metadata,
                )
                
                # Rename/move the video from target_decode to item directory
                src_video = target_output_dir / "target.mp4"
                if src_video.exists():
                    src_video.rename(item_dir / "target.mp4")
                    target_output_dir.rmdir()  # Remove empty directory
                
            except Exception as e:
                print(f"  Error decoding target for batch {batch_idx+1}, item {item_idx+1}: {e}")
            
            try:
                # Decode and save context video
                context_output_dir = item_dir / "context_decode"
                context_output_dir.mkdir(parents=True, exist_ok=True)
                
                decode_and_save(
                    pred_fd=context_item,
                    gt_fd=context_item,
                    out_dir=context_output_dir,
                    window_length=0,
                    video_filename="context.mp4",
                    make_audio_plots=False,
                    store_decoded_generated=False,
                    store_decoded_gt=False,
                    metadata=context_item.metadata,
                )
                
                # Rename/move the video from context_decode to item directory
                src_video = context_output_dir / "context.mp4"
                if src_video.exists():
                    src_video.rename(item_dir / "context.mp4")
                    context_output_dir.rmdir()  # Remove empty directory
                
            except Exception as e:
                print(f"  Error decoding context for batch {batch_idx+1}, item {item_idx+1}: {e}")
        
        print(f"  Batch {batch_idx+1} complete\n")
    
    print(f"\n=== Decoding complete! Output saved to {base_output_dir} ===")



if __name__ == "__main__":
    main()
