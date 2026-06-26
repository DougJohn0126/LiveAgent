# PLAICraft Pi0 Plai V1 Model

## Overview

**plaicraft-model-pi0** is the plai_v1 model inspired by pi0-model for the PLAICraft environment. The model learns to generate actions while conditioning on previous states and actions. 

- **State**: 
  - `audio_hear` (in-game/Discord audio latents, 75Hz)
  - `video` (VAE latents from game video, 10Hz)
  
- **Actions** (predicted by the model):
  - `audio_speak` (player speech audio latents, 75Hz)
  - `keyboard` (keyboard key encodings, 50Hz)
  - `mouse` (mouse movement bins, 100Hz)

The conditioning consists of two parts:

- **Short-term Memory**: 
  - Through self-attention to the state and action latents of the previous frames.
  
- **Long-term Memory**:
  - Last hidden state of a MinGRU History Encoder that processes long-term context. 

---

## Installation

This repository is pip-installable for easy integration into other projects:

```bash
pip install git+https://github.com/plai-group/plaicraft-model-pi0.git
```

### First-time setup (recommended for contributors)

For local development, run the interactive setup wizard:

```bash
python setup.py
```

### Using Dataset Classes Elsewhere

After installation, you can import and use the classes in other projects:

```python
from data.datasets import MapStyleDataset, IterStyleDataset, SemanticEvaluationDataset
from data.samplers import DynamicLengthBatchSampler
from data.utils import normalization
```

This is particularly useful if you want to use the PLAICraft datasets in different training pipelines or research projects without duplicating the dataset code.

---

## Getting Started

### Download Dataset

Download the PLAICraft dataset from S3:

```bash
python3 -m plaicraft.data.download_data --format hdf5 --confirm-download --emails <HASHED_EMAIL>
```

**Available options:**
- `--sessions <SESSION_ID_1> <SESSION_ID_2> ...` - Download specific sessions (recommended for quick testing)
- `--format hdf5` - Use HDF5 format (recommended)
- Dry run without `--confirm-download` to preview what will be downloaded

See [docs/DATASET_DOC.md](docs/DATASET_DOC.md) for more details.

### 2. Download Pretrained Models (For Clusters Without Internet)

If you're using a compute cluster wOptional - for offline clusters)

On clusters without internet access on compute nodes, run on the login node:
```bash
python scripts/download_models.py  # See docs/OFFLINE_SETUP.md
```

Build the Singularity container (recommended for reproducibility):

```bash
singularity build plaicraft.sif containers/singularity/plaicraft.def
```

- CONTAINERS_PATH `/ubc/cs/research/ubc_ml/plaicraft/containers` (Updated by Charles 2/23/2026)

### Training

#### Using the training script locally: [TODO:complete]

```bash
python3 scripts/train.py
```

#### Using SLURM (on supported clusters):

Training scripts for different clusters are available in `scripts/plai_v1/`:

```bash
sbatch scripts/plai_v1/deepspeed.sh
```

### Sampling / Inference

Generate predictions from a trained model:

```bash
python3 scripts/sample.py \
  --model_type plai_v1 \
  --checkpoint_path <path_to_checkpoint> \
  --config <path_to_config.yaml> \
  --data_path <path_to_data> \
  --output_dir <output_dir>
```

Evaluation uses a unified entrypoint: `python src/eval.py`.

- Offline sampling only: `python src/eval.py eval.mode=offline_sample ...`
- Semantic evaluation (sampling + metrics): `python src/eval.py eval.mode=semantic_evaluation ...`

### Semantic Evaluation

Semantic evaluation is now **database-driven**, where each clip has a designated metric specified in the evaluation database. Only the requested metrics are computed for each clip.

**Standalone evaluation with checkpoint:**

```bash
python src/eval.py \
  experiment=evaluation \
  ckpt_path=/path/to/checkpoint.pt
```

**Enable in training:**

```bash
python src/train.py \
  callbacks.sync_semantic_evaluation.enabled=true
```

**Configuration:**
- Set `EVALUATION_METADATA_DB_PATH` environment variable
- Required: semantic evaluation sessions in `DATASET_PATH`
- See [DATABASE_DRIVEN_EVALUATION.md](docs/DATABASE_DRIVEN_EVALUATION.md) for full documentation

Common `sync_semantic_evaluation` callback settings (in `configs/callbacks/sync_semantic_evaluation.yaml`):
- `every_n_train_steps`: Validation interval in training steps
- `start_index` / `stop_index`: Validation sample range (max 48)
- `max_samples`: Optional cap on number of samples
- `direct_scalar_logging`: Log metrics to W&B (default: true)
- `log_media`: Log media artifacts to W&B (default: true)

---
## Notes

### Cluster Data & Utils status

| Cluster | Full data available | Size (Hours) | Latent Format | DATASET_PATH |
|---------|---------------------|-----------|----------------------|----------|
| ubcml | ✅ | 10K | hdf5 | `/ubc/cs/research/plai-scratch/plaicraft-dataset/processed` |
| vulcan | ❌ | 1.6K | hdf5 | `/project/aip-fwood/plaicraft_data/processed` |
| rorqual | ✅ | 10k | hdf5 | `/project/rrg-fwood/shared/plaicraft/data` |

| Cluster | Container |
|---------|-----------|
| ubcml | `/ubc/cs/research/ubc_ml/plaicraft/containers/plaicraft_ubuntu2404.sif` |
| vulcan | `/project/aip-fwood/plaicraft_data/containers/plaicraft_ubuntu2404_rdma.sif` |
| rorqual | `/project/rrg-fwood/shared/plaicraft/containers/plaicraft_ubuntu2404.sif` |

For global metadata databases, use the files under `data`.
`semantic_evaluation.db` is for semantic evaluation.
`latest_9.5k_hrs_12619_players_training` contains all available training sessions, with 9.5k total hours and 12619 players.
`high_quality_1.5k_hrs_1105_players_training` contains 1.5k hours of high-quality data (more interactive data) from 1105 players.


### Dataloader

We provide a various kinds of dataloader suitable for different usages, that are pip installable [TODO]

For more details on the dataset, see [docs/DATASET_DOC.md](docs/DATASET_DOC.md).

### WANDB & Deepspeed Checkpoint Helpers

We provide a list of helper utilities for wandb, checkpoint saving and conversion. 

1. [src/plaicraft/helpers/deepspeed_to_universal.py](src/plaicraft/helpers/deepspeed_to_universal.py): Transforms a deepspeed sharded checkpoint to a deepspeed universal checkpoint, use this first if you want to resume training from a checkpoint with different hardware configuration. [More info](https://www.deepspeed.ai/tutorials/universal-checkpointing/)
2. [src/plaicraft/helpers/wandb_logger.py](src/plaicraft/helpers/wandb_logger.py): Customized wandb logger
3. [src/plaicraft/helpers/deepspeed_zero_to_fp32.py](src/plaicraft/helpers/deepspeed_zero_to_fp32.py): This script extracts fp32 consolidated weights from a zero 1, 2 and 3 DeepSpeed checkpoints. It gets copied into the top level checkpoint dir, so the user can easily do the conversion at any point in the future. Use this for sampling. 

