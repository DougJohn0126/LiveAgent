# Offline Setup Guide

This guide explains how to set up the PLAICraft model for use on compute clusters without internet access.

## Overview

The PLAICraft model and validation metrics use several pretrained models that are normally downloaded on first use:

**HuggingFace Models (for generation and audio):**
- **madebyollin/sdxl-vae-fp16-fix** - VAE model for video decoding
- **facebook/encodec_24khz** - Encodec model for audio encoding  
- **facebook/wav2vec2-base** - Wav2Vec2 model and processor for audio validation

**PyTorch Hub / GitHub Models (for validation metrics):**
- **VGG16** - Feature extractor for LPIPS metric calculation during validation
- **Inception V3** - Feature extractor for FID metric calculation during validation

On clusters where compute nodes don't have internet access, you need to download all these models once from a login node, then point your compute jobs to the cached models.

## Quick Start

### 1. Download Models on Login Node

Run this script on a login node (which has internet access):

```bash
# Use default cache location (~/.cache/huggingface and ~/.cache/torch)
python scripts/download_models.py

# OR use custom shared cache locations
export HF_HOME=/project/shared/models/huggingface
export TORCH_HOME=/project/shared/models/torch
python scripts/download_models.py
```

**Preview what will be downloaded:**
```bash
python scripts/download_models.py --dry-run
```

### 2. Use Models in Compute Jobs

When submitting jobs to compute nodes, ensure the same cache locations are set:

#### In SLURM Scripts

Add to your SLURM script before running Python:

```bash
#!/bin/bash
#SBATCH --job-name=plaicraft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

# Point to the cached models
export HF_HOME=/project/shared/models/huggingface
export TORCH_HOME=/project/shared/models/torch

# Run training/inference
singularity exec --nv plaicraft.sif python src/train.py ...
```

#### In Python Code

The models will automatically use the cache when the environment variables are set. No code changes needed!

## Cache Configuration

### HuggingFace Models (VAE, Encodec, Wav2Vec2)

HuggingFace uses environment variables to control where models are cached:

**Recommended (Simple):**
- **`HF_HOME`** - Base directory for all HuggingFace data (recommended)
  - Default: `~/.cache/huggingface`
  - Models stored in: `$HF_HOME/hub/`

**Advanced (Fine-grained Control):**
- **`HF_HUB_CACHE`** - Directory for Hub models (default: `$HF_HOME/hub`)
- **`TRANSFORMERS_CACHE`** - Directory for transformers models (default: `$HF_HOME/transformers`)

### PyTorch Hub Models (VGG16)

PyTorch uses environment variables for hub model caching:

**Recommended:**
- **`TORCH_HOME`** - Base directory for PyTorch models (recommended)
  - Default: `~/.cache/torch`
  - Checkpoints stored in: `$TORCH_HOME/checkpoints/`

## Usage Examples

### Example 1: Personal Cache

```bash
# Download models to your home directory
python scripts/download_models.py

# In your SLURM job (uses default ~/.cache/huggingface and ~/.cache/torch)
sbatch slurm_scripts/train.sh
```

### Example 2: Shared Team Cache

```bash
# Download once to shared location
export HF_HOME=/project/plaicraft/shared_models/huggingface
export TORCH_HOME=/project/plaicraft/shared_models/torch
python scripts/download_models.py

# Team members use the same cache in their jobs
export HF_HOME=/project/plaicraft/shared_models/huggingface
export TORCH_HOME=/project/plaicraft/shared_models/torch
sbatch slurm_scripts/train.sh
```

### Example 3: Per-Job Cache

```bash
# Download to project scratch space
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
python scripts/download_models.py

# Copy to compute node local storage for fast access
#!/bin/bash
#SBATCH --job-name=plaicraft

# Copy to node-local storage
export NODE_HF_CACHE=/tmp/$USER/hf_cache
export NODE_TORCH_CACHE=/tmp/$USER/torch_cache
mkdir -p $NODE_HF_CACHE $NODE_TORCH_CACHE
cp -r /scratch/$USER/hf_cache/* $NODE_HF_CACHE/
cp -r /scratch/$USER/torch_cache/* $NODE_TORCH_CACHE/
export HF_HOME=$NODE_HF_CACHE
export TORCH_HOME=$NODE_TORCH_CACHE

# Run job
python src/train.py ...
```

## Verifying the Setup

After downloading, verify the models are cached:

```bash
# Check HuggingFace cache directory
ls -lh ~/.cache/huggingface/hub/

# Should see directories like:
# models--madebyollin--sdxl-vae-fp16-fix
# models--facebook--encodec_24khz  
# models--facebook--wav2vec2-base

# Check PyTorch cache directory
ls -lh ~/.cache/torch/checkpoints/

# Should see:
# vgg16-397923af.pth  (~528 MB)
# pt_inception-2015-12-05-6726825d.pth  (~104 MB)
```

## Troubleshooting

### "No such file or directory" errors

**Problem:** Models not found in cache

**Solution:** Ensure `HF_HOME` is set consistently:
```bash
# Check what's set
echo $HF_HOME

# Set it in both download and job scripts
export HF_HOME=/your/cache/path
```

### "Connection error" on compute nodes

**Problem:** Code is trying to download models

**Solution:** 
1. Verify HuggingFace models are downloaded: `ls $HF_HOME/hub/`
2. Verify PyTorch models are downloaded: `ls $TORCH_HOME/checkpoints/`
3. Ensure both `HF_HOME` and `TORCH_HOME` are exported in your job script
4. Check you're not accidentally using a different cache location

### Disk quota exceeded

**Problem:** Models are large and exceed home directory quota

**Solution:** Use a larger shared or scratch directory:
```bash
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
# or
export HF_HOME=/project/shared/models/huggingface
export TORCH_HOME=/project/shared/models/torch
```

### Models re-downloading every time

**Problem:** Cache location changes between runs

**Solution:**
1. Set both `HF_HOME` and `TORCH_HOME` consistently in all jobs
2. Consider adding to your `~/.bashrc`:
   ```bash
   export HF_HOME=/project/shared/models/huggingface
   export TORCH_HOME=/project/shared/models/torch
   ```

## Model Storage Requirements

Approximate disk space needed:

**HuggingFace Models:**
- **sdxl-vae-fp16-fix**: ~330 MB
- **encodec_24khz**: ~150 MB  
- **wav2vec2-base**: ~360 MB

**PyTorch Models:**
- **vgg16**: ~528 MB (for LPIPS validation metric)
- **inception-v3**: ~104 MB (for FID validation metric)

**Total: ~1.5 GB** (plus some overhead for model configurations and tokenizers)

## Additional Resources

- [HuggingFace Hub Cache Documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [Transformers Model Caching](https://huggingface.co/docs/transformers/installation#cache-setup)
- [Diffusers Installation](https://huggingface.co/docs/diffusers/installation)
- [PyTorch Hub Documentation](https://pytorch.org/hub/)

## Support

If you encounter issues not covered here, please:
1. Check that all required packages are installed: `pip install -r requirements.txt`
2. Verify internet connectivity on login node: `curl https://huggingface.co`
3. Check disk space: `df -h $HF_HOME && df -h $TORCH_HOME`
4. Verify cached files exist: `ls -lh $HF_HOME/hub/ && ls -lh $TORCH_HOME/checkpoints/`
5. Open an issue with error logs
