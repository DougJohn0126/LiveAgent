#!/bin/bash
#SBATCH --partition=ubcml
#SBATCH --job-name=plai_v1_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --mem=40gb
#SBATCH --output=logs/train/slurm-%j.out
#SBATCH --error=logs/train/slurm-%j.err
#SBATCH --export=ALL

# =============================================================================
# Environment Setup
# =============================================================================
source .env

export WANDB_MODE=offline
export WORLD_SIZE="$SLURM_NTASKS"

# Optional overrides injected by external resubmission tooling.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-rope_autoresume}"
export RUN_DIR="${RUN_DIR:-}"
export RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
export WANDB_RESUME_ID="${WANDB_RESUME_ID:-}"
export WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-must}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"


srun --unbuffered bash -c '
  export JOB_CACHE_ROOT="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/rank_${SLURM_PROCID}"
  export TMPDIR="$JOB_CACHE_ROOT"
  export XDG_DATA_HOME="$JOB_CACHE_ROOT/xdg/data"

  export TORCH_EXTENSIONS_DIR="$JOB_CACHE_ROOT/torch_extensions"
  export TRITON_CACHE_DIR="$JOB_CACHE_ROOT/triton_cache"
  export MPLCONFIGDIR="$JOB_CACHE_ROOT/matplotlib"

  # Only rank 0 should create/own a W&B run in distributed training.
  if [[ "$SLURM_PROCID" != "0" ]]; then
    export WANDB_MODE=disabled
  fi

  if [[ -n "$RUN_DIR" ]]; then
    export WANDB_CACHE_PATH="$RUN_DIR/wandb"
  else
    export WANDB_CACHE_PATH="$JOB_CACHE_ROOT/wandb"
  fi
  export WANDB_DATA_DIR="$WANDB_CACHE_PATH/staging"
  export WANDB_CACHE_DIR="$WANDB_CACHE_PATH/cache"
  export WANDB_ARTIFACT_DIR="$WANDB_CACHE_PATH/downloads"
  export WANDB_DIR="$WANDB_CACHE_PATH/runlogs"

  mkdir -p "$TORCH_EXTENSIONS_DIR" \
           "$TRITON_CACHE_DIR" \
           "$MPLCONFIGDIR" \
           "$XDG_DATA_HOME" \
           "$WANDB_DATA_DIR" \
           "$WANDB_CACHE_DIR" \
           "$WANDB_ARTIFACT_DIR" \
           "$WANDB_DIR"

  TRAIN_ARGS=(experiment="$EXPERIMENT_NAME")
  if [[ -n "$RUN_DIR" ]]; then
    TRAIN_ARGS+=(hydra.run.dir="$RUN_DIR")
  fi
  if [[ -n "$RESUME_CKPT_PATH" ]]; then
    TRAIN_ARGS+=(ckpt_path="$RESUME_CKPT_PATH")
  fi
  if [[ -n "$WANDB_RESUME_ID" ]]; then
    TRAIN_ARGS+=(logger.wandb.id="$WANDB_RESUME_ID")
    TRAIN_ARGS+=(logger.wandb.resume="$WANDB_RESUME_MODE")
  fi
  if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARR=($EXTRA_TRAIN_ARGS)
    TRAIN_ARGS+=("${EXTRA_ARGS_ARR[@]}")
  fi

  singularity exec --nv --containall \
    --env CC=/usr/bin/gcc \
    --env CXX=/usr/bin/g++ \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WORLD_SIZE="$WORLD_SIZE" \
    --env SLURM_PROCID="$SLURM_PROCID" \
    --env SLURM_NTASKS="$SLURM_NTASKS" \
    --env SLURM_LOCALID="$SLURM_LOCALID" \
    --env SLURM_NODEID="$SLURM_NODEID" \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-4}" \
    --env TMPDIR="$TMPDIR" \
    --env XDG_DATA_HOME="$XDG_DATA_HOME" \
    --env TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env WANDB_MODE="$WANDB_MODE" \
    --env WANDB_CACHE_PATH="$WANDB_CACHE_PATH" \
    --env WANDB_DATA_DIR="$WANDB_DATA_DIR" \
    --env WANDB_CACHE_DIR="$WANDB_CACHE_DIR" \
    --env WANDB_ARTIFACT_DIR="$WANDB_ARTIFACT_DIR" \
    --env WANDB_DIR="$WANDB_DIR" \
    --env DATASET_PATH="$DATASET_PATH" \
    --env TRAINING_METADATA_DB_PATH="$TRAINING_METADATA_DB_PATH" \
    --env VALIDATION_METADATA_DB_PATH="$VALIDATION_METADATA_DB_PATH" \
    --env EVALUATION_METADATA_DB_PATH="$EVALUATION_METADATA_DB_PATH" \
    -B "${HOME}" \
    -B "${DATASET_PATH}" \
    -B "$(pwd)" --pwd "$(pwd)" \
    -B "$JOB_CACHE_ROOT":"$JOB_CACHE_ROOT" \
    "${CONTAINER_IMAGE_PATH}" \
    python src/train.py "${TRAIN_ARGS[@]}"
'