#!/usr/bin/env python3
"""
Download all required pretrained models for offline use.

This script should be run on a login node with internet access before
submitting jobs to compute nodes that may not have internet connectivity.

Models are downloaded to their respective cache directories:
- HuggingFace models (VAE, Encodec, Wav2Vec2): HF_HOME (default: ~/.cache/huggingface)
- PyTorch hub models (VGG16): TORCH_HOME (default: ~/.cache/torch)

You can customize cache locations via environment variables:
- HF_HOME (recommended): sets the base cache directory for HuggingFace models
- TRANSFORMERS_CACHE: for transformers models
- HF_HUB_CACHE: for Hub models
- TORCH_HOME: for PyTorch hub models (VGG16)

Example usage:
    # Use default cache locations
    python scripts/download_models.py

    # Use custom cache locations
    export HF_HOME=/path/to/huggingface/cache
    export TORCH_HOME=/path/to/torch/cache
    python scripts/download_models.py

    # Check what models will be downloaded without downloading
    python scripts/download_models.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def download_vae_model(dry_run=False):
    """Download the VAE model for video decoding."""
    model_id = "madebyollin/sdxl-vae-fp16-fix"
    print(f"📦 VAE Model: {model_id}")
    
    if dry_run:
        print("   [DRY RUN] Would download this model\n")
        return
    
    try:
        from diffusers import AutoencoderKL
        import torch
        
        print("   Downloading...")
        _ = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        print("   ✅ Successfully downloaded\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return False
    
    return True


def download_encodec_model(dry_run=False):
    """Download the Encodec model for audio encoding."""
    model_id = "facebook/encodec_24khz"
    print(f"📦 Encodec Model: {model_id}")
    
    if dry_run:
        print("   [DRY RUN] Would download this model\n")
        return
    
    try:
        from transformers import EncodecModel
        
        print("   Downloading...")
        _ = EncodecModel.from_pretrained(model_id)
        print("   ✅ Successfully downloaded\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return False
    
    return True


def download_wav2vec2_models(dry_run=False):
    """Download Wav2Vec2 models for audio validation."""
    model_id = "facebook/wav2vec2-base"
    print(f"📦 Wav2Vec2 Models: {model_id}")
    
    if dry_run:
        print("   [DRY RUN] Would download processor and model\n")
        return
    
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        
        print("   Downloading processor...")
        _ = Wav2Vec2Processor.from_pretrained(model_id)
        
        print("   Downloading model...")
        _ = Wav2Vec2Model.from_pretrained(model_id)
        
        print("   ✅ Successfully downloaded\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return False
    
    return True


def download_vgg16_model(dry_run=False):
    """Download VGG16 model for LPIPS metric calculation during validation."""
    print(f"📦 VGG16 Model (for LPIPS validation metric)")
    
    if dry_run:
        print("   [DRY RUN] Would download VGG16 model\n")
        return
    
    try:
        import torch
        
        print("   Downloading VGG16 from PyTorch Hub...")
        # This will cache to TORCH_HOME/checkpoints/vgg16-397923af.pth
        _ = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        print("   ✅ Successfully downloaded\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return False
    
    return True


def download_inception_model(dry_run=False):
    """Download Inception V3 model for FID metric calculation during validation."""
    print(f"📦 Inception V3 Model (for FID validation metric)")
    
    if dry_run:
        print("   [DRY RUN] Would download Inception V3 model\n")
        return
    
    try:
        import torch
        from pytorch_fid.inception import InceptionV3
        
        print("   Downloading Inception V3 from pytorch-fid...")
        # InceptionV3 will download and cache the model automatically
        # Cache location: TORCH_HOME/checkpoints/pt_inception-2015-12-05-6726825d.pth
        _ = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
        print("   ✅ Successfully downloaded\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return False
    
    return True


def print_cache_info():
    """Print information about the cache directories being used."""
    print_banner("Cache Directory Information")
    
    # Check various HuggingFace cache environment variables
    hf_home = os.getenv('HF_HOME')
    hf_hub_cache = os.getenv('HF_HUB_CACHE')
    transformers_cache = os.getenv('TRANSFORMERS_CACHE')
    torch_home = os.getenv('TORCH_HOME')
    
    if hf_home:
        print(f"✓ HF_HOME: {hf_home}")
    else:
        default_hf = Path.home() / ".cache" / "huggingface"
        print(f"  HF_HOME: Not set (will use default: {default_hf})")
    
    if hf_hub_cache:
        print(f"✓ HF_HUB_CACHE: {hf_hub_cache}")
    else:
        print(f"  HF_HUB_CACHE: Not set (will use HF_HOME/hub)")
    
    if transformers_cache:
        print(f"✓ TRANSFORMERS_CACHE: {transformers_cache}")
    else:
        print(f"  TRANSFORMERS_CACHE: Not set (will use HF_HOME/transformers)")
    
    if torch_home:
        print(f"✓ TORCH_HOME: {torch_home}")
    else:
        default_torch = Path.home() / ".cache" / "torch"
        print(f"  TORCH_HOME: Not set (will use default: {default_torch})")
    
    print("\nℹ️  Tip: Set environment variables for custom cache locations:")
    print("   export HF_HOME=/path/to/huggingface/cache")
    print("   export TORCH_HOME=/path/to/torch/cache")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download all required pretrained models for offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default locations
  python scripts/download_models.py

  # Download to custom locations
  export HF_HOME=/project/shared/models/huggingface
  export TORCH_HOME=/project/shared/models/torch
  python scripts/download_models.py

  # Preview what will be downloaded
  python scripts/download_models.py --dry-run

For more information about cache configuration, see:
https://huggingface.co/docs/huggingface_hub/guides/manage-cache
https://pytorch.org/hub/
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    
    args = parser.parse_args()
    
    print_banner("PLAICraft Model Downloads")
    print("This script downloads all required pretrained models.")
    print("Run this on a login node with internet access.")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be downloaded\n")
    
    print_cache_info()
    
    # Check required packages
    print_banner("Checking Dependencies")
    missing_packages = []
    
    try:
        import torch
        print("✓ torch")
    except ImportError:
        print("✗ torch (missing)")
        missing_packages.append("torch")
    
    try:
        import diffusers
        print("✓ diffusers")
    except ImportError:
        print("✗ diffusers (missing)")
        missing_packages.append("diffusers")
    
    try:
        import transformers
        print("✓ transformers")
    except ImportError:
        print("✗ transformers (missing)")
        missing_packages.append("transformers")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them first:")
        print(f"   pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    # Download models
    print_banner("Downloading Models")
    
    results = []
    results.append(("VAE", download_vae_model(args.dry_run)))
    results.append(("Encodec", download_encodec_model(args.dry_run)))
    results.append(("Wav2Vec2", download_wav2vec2_models(args.dry_run)))
    results.append(("VGG16", download_vgg16_model(args.dry_run)))
    results.append(("Inception V3", download_inception_model(args.dry_run)))
    
    # Summary
    print_banner("Summary")
    
    if args.dry_run:
        print("Dry run completed. No files were downloaded.")
        print("\nRun without --dry-run to actually download the models.")
    else:
        failed = [name for name, result in results if result is False]
        
        if failed:
            print(f"❌ Some models failed to download: {', '.join(failed)}")
            print("\nPlease check the errors above and try again.")
            sys.exit(1)
        else:
            print("✅ All models downloaded successfully!")
            print("\nYou can now run training/inference on compute nodes without internet.")
            print("\nℹ️  Make sure to set the same environment variables in your SLURM/compute jobs:")
            print("   export HF_HOME=/path/to/huggingface/cache")
            print("   export TORCH_HOME=/path/to/torch/cache")


if __name__ == "__main__":
    main()
