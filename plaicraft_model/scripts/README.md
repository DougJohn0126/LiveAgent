### SLURM SCRIPT USAGE INSTRUCTION
You need to modify the slurm parameter before use, for example, where the error and logs are being exported to. 

---

## Setup Scripts

### `download_models.py`

Downloads all required pretrained models for offline use on clusters without internet access.

**Usage:**
```bash
# Download to default locations (~/.cache/huggingface and ~/.cache/torch)
python scripts/download_models.py

# Download to custom locations  
export HF_HOME=/project/shared/models/huggingface
export TORCH_HOME=/project/shared/models/torch
python scripts/download_models.py

# Preview what will be downloaded
python scripts/download_models.py --dry-run
```

**Downloaded Models:**

**HuggingFace (VAE, Encodec, Wav2Vec2):**
- `madebyollin/sdxl-vae-fp16-fix` - VAE for video decoding
- `facebook/encodec_24khz` - Encodec for audio encoding
- `facebook/wav2vec2-base` - Wav2Vec2 for audio semantic evaluation

**PyTorch Hub (Semantic Evaluation Metrics):**
- `vgg16` - VGG16 for LPIPS metric
- `inception-v3` - Inception for FID metric

See [docs/OFFLINE_SETUP.md](../docs/OFFLINE_SETUP.md) for detailed offline setup instructions.

