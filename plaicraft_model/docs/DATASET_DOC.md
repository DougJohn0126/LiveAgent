# Dataset download/transfer from S3

First you need to download the dataset or a portion of it using the download script:

1) Dry run (default; no download):
python -m plaicraft.data.download_data --format hdf5 --emails <HASHED_EMAIL>

2) Download (add --confirm-download):
python -m plaicraft.data.download_data --format hdf5 --confirm-download --emails <HASHED_EMAIL>

3) Download specific sessions (fastest; requires --emails):
python -m plaicraft.data.download_data --format hdf5 --confirm-download --emails <HASHED_EMAIL> --sessions <SESSION_ID_1> <SESSION_ID_2>

4) Download entire bucket (slow):
python -m plaicraft.data.download_data --format hdf5 --confirm-download


# PLAICraft Datasets and Dataloaders

This repository contains a PyTorch Dataset implementations for loading multimodal PLAICraft session data.

## Supported Modalities
* `video` (VAE latents, 10Hz)
* `audio_speak` (Player speech audio latents, 75Hz)
* `audio_hear` (In-game/Discord audio latents, 75Hz)
* `key_press` (Keyboard encodings, 50Hz)
* `mouse_movement` (Mouse movement bins, 100Hz)

---

## 1. Prerequisite: Folder Structure

Your data directory should look roughly like this. Note that specific datasets require specific auxiliary databases (like `word_boundaries.db`).

```text
<DATASET_PATH>/
├── global_database.db               # Main metadata DB
├── word_boundaries.db               # (Required for Audio Pretrain)
├── <player_email>/
│   └── <session_id>/
│       ├── encoded_video/
│       │   ├── batch_0000.pt
│       │   └── ...
│       ├── encoded_audio_continuous/
│       │   ├── <session_id>_encoded_audio_in.hdf5
│       │   └── <session_id>_encoded_audio_out.hdf5
│       └── <session_id>.db          # Per-session Action/Transcript DB
```

---

## 2. Dataset Types

### Standard Map-based Dataset
**File:** `src/plaicraft/data/datasets/mapstyle.py`
**Class:** `MapStyleDataset`

A standard pytorch map-style dataset. Loads fixed-length windows sliding across sessions. Good for standard autoregressive training.

```python
from data.datasets.mapstyle import MapStyleDataset
from torch.utils.data import DataLoader

dataset = MapStyleDataset(
    dataset_path="/path/to/data",
    modalities=["video", "audio_speak", "key_press", "mouse_movement"],
    window_length_frames=10,
    hop_length_frames=10,
    global_database_path="/path/to/global_database.db"
)

dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
```

### Iterable Dataset with SSD caching
**File:** `src/plaicraft/data/datasets/iterstyle.py`
**Class:** `IterStyleDataset`

A pytorch iterable-style dataset pairded with asynchronous SSD cache. It asynchronously copies random chunks of data from slow HDD storage to a local fast SSD (`/tmp` or `$SLURM_TMPDIR`) during training. Suitable if your training speed is I/O bounded.

* **Must paired with:** `DynamicLengthBatchSampler` (to group sequences of similar lengths).
* **Threading:** Handles background copying threads automatically.

```python
from data.datasets.iterstyle import IterStyleDataset
from data.samplers.dynamic_length import DynamicLengthBatchSampler

# 1. Initialize Dataset
dataset = IterStyleDataset(
    args=args, # Namespace object containing dataset_path, ssd_cache_dir, etc.
    seed=42
)

# 2. Initialize Sampler
batch_sampler = DynamicLengthBatchSampler(args=args, seed=42)

# 3. Loader (Note: persistent_workers=True is highly recommended)
dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=4,
    collate_fn=dataset.collate_fn,
    worker_init_fn=IterStyleDataset.worker_init_fn, # Crucial for seeding
    persistent_workers=True
)
```
**Key Arguments (passed via `args` namespace):**
* `--use_ssd_cache`: Enable the copying mechanism.
* `--ssd_cache_dir`: Target temp directory (default: `/tmp`).
* `--chunk_size_gb`: Size of data chunks to move (default: 2GB).
* `--iterations_per_cache`: How many batches to fetch before swapping the SSD chunk.

### Validation Dataset (Prompt/Response)
**File:** `src/plaicraft/data/datasets/validation.py`
**Class:** `ValidationDataset`

Driven by a CSV file. For every row in the CSV, it returns **two** aligned windows: a `prompt` (context) and a `response` (target).

* **CSV Columns Required:** `Session_ID`, `R_start (ms)`, `P_R_valid` (must be true), `Num`.
* **Structure:** Returns a nested dictionary.

```python
from data.datasets.validation import ValidationDataset

dataset = ValidationDataset(
    dataset_path="/path/to/data",
    csv_path="/path/to/validation_split.csv",
    modalities=["video", "audio_speak"],
    min_frames=2
)

item = dataset[0]
# item keys: ['prompt', 'response']
# item['prompt'] keys: ['video', 'audio_speak', 'metadata', ...]
```

---

## 3. Output Formats

### Standard Batch Output
(Used by `FixedWindow`, and `SSDCache`)

Each batch is a dictionary:

* `metadata`: List of length `B`. Contains `player_id`, `session_id`, timestamps.
    * *AudioPretrain* adds: `word`, `word_id`, `word_duration_ms`.
* `video`: `(B, T, 4, 96, 160)`
    * *Note:* `SSDCache` might require reshaping if the sampler returns variable T.
* `audio_speak`: `(B, 128, Audio_Tokens)` (Audio tokens approx 15 per 200ms).
* `audio_hear`: `(B, 128, Audio_Tokens)`
* `key_press`: `(B, T, 10, 16)` (keyboard encoding)
* `mouse_movement`: `(B, T, 20, 2)` (dx/dy binned)
* `transcript_speak`: List of length `B`. Each entry is a list of `(word, start_ms, end_ms)`.
* `padding_mask`: `(B, T, 1)` Used to identify if the portion is padded (not valid) or not. 

### Validation Batch Output
(Used by `ValidationDataset`)

Returns a nested dictionary separating the context from the future prediction:

```python
{
    "prompt": {
        "video": Tensor(...),
        "audio_speak": Tensor(...),
        "metadata": [...],
        # ... other modalities (0 to R_start)
    },
    "response": {
        "video": Tensor(...),
        "audio_speak": Tensor(...),
        "metadata": [...],
        # ... other modalities (R_start to R_start + Duration)
    }
}
```

---

## 4. Utility: Dynamic Length Batch Sampler

**File:** `src/plaicraft/data/samplers/dynamic_length.py`

Designed to work with the SSD Cache dataset. Instead of fixed batch sizes, it constructs batches based on total token count to minimize padding when training on variable-length sequences.

**Key Arguments:**
* `--max_tokens_per_batch`: Target size of a batch (sequence_length × batch_size).
* `--min_batch_utilization`: Threshold to determine if a batch is "full enough".
* `--length_similarity_ratio`: How similar lengths must be to be grouped together (e.g., 0.25 means lengths can vary by 25%).
