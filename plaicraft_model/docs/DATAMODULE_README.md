# Pi0 DataModule Usage Guide

## Canonical 200ms Unit (Source of Truth)

Use these exact shapes and rates everywhere. Canonical constants live in `src/utils/constants.py`. All shape references come from this single source of truth.

- Unit duration: **200ms** (`0.2s`)
- Video: **10Hz**, shape per unit `[B, T, 2, 4, 96, 160]`
- Audio in/out: **75Hz**, shape per unit `[B, T, 15, 128]`
- Keyboard: **50Hz**, shape per unit `[B, T, 10, 16]`
- Mouse: **100Hz**, shape per unit `[B, T, 20, 2]`

`T` above is the number of 200ms units. The inner `2` is the number of video frames per 200ms unit.

## Quick Start

### Command Line Usage

```bash
python train.py pi0_policy \
    --dataset_path /path/to/data \
    --modalities video audio_speak audio_hear key_press mouse_movement \
    --batch_size 8 \
    --use_datamodule 1 \
    --num_workers 4 \
    --shuffle 1
```

### Python API

```python
from data.datamodule import DataModule
from plaicraft.training.plai_v1_lightning_module import PlaiV1LightningModule
import lightning.pytorch as pl

# Create DataModule
datamodule = DataModule(
    dataset_path="/path/to/data",
    modalities=["video", "audio_speak", "audio_hear", "key_press", "mouse_movement"],
    window_length_frames=100,
    batch_size=8,
    num_workers=4,
    shuffle=True,
)

# Create trainer module
trainer_module = PlaiV1LightningModule(args, datamodule=datamodule)

# Create Lightning Trainer and train
trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1)
trainer.fit(trainer_module, datamodule=datamodule)
```

## Parameters

- `dataset_path`: Path to dataset directory
- `modalities`: List of modalities to load
- `window_length_frames`: Output sequence length (in units, not raw frames)
- `batch_size`: Number of slices sampled from each single base window read (effective training batch size)
- `num_workers`: Number of data loading workers
- `shuffle`: Whether to shuffle data

## What You Get

Each training batch contains:
- `video`, `audio_speak`, `audio_hear`, `key_press`, `mouse_movement`: Data tensors
- `padding_mask`: Padding mask for each timestep (`True` for valid, `False` for padded)
- `dataframe_indices`: Dataframe indices

The DataModule reads one base window per DataLoader fetch and emits one `(target, context)` batch where context is padded to a rectangular shape `[B, T_max, ...]`.
