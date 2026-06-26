#!/bin/bash
#SBATCH --partition=ubcml
#SBATCH --job-name=plai_v1_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --mem=40gb
#SBATCH --output=logs/eval/slurm-%j.out
#SBATCH --error=logs/eval/slurm-%j.err
#SBATCH --export=ALL,TMPDIR=

source .env

cd "${PROJECT_ROOT}"

export TMPDIR="${SLURM_TMPDIR}/${SLURM_JOB_ID}"
export WANDB_CACHE_PATH="${TMPDIR}/wandb"

mkdir -p "$TMPDIR"
mkdir -p "$WANDB_CACHE_PATH"

export WANDB_DATA_DIR=$WANDB_CACHE_PATH/staging
export WANDB_CACHE_DIR=$WANDB_CACHE_PATH/cache
export WANDB_ARTIFACT_DIR=$WANDB_CACHE_PATH/downloads
export WANDB_DIR=$WANDB_CACHE_PATH/runlogs

mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_DIR"

export TRITON_CACHE_DIR=$TMPDIR/triton_cache
export XDG_CACHE_HOME="${TMPDIR}/.cache"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

srun --unbuffered singularity exec --nv \
    --env CC=/usr/bin/gcc \
    --env CXX=/usr/bin/g++ \
    --env TMPDIR="${TMPDIR}" \
    --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    -B "${HOME}" \
    -B "${DATASET_PATH}" \
    -B "$(pwd)" --pwd "$(pwd)" \
    -B "${TMPDIR}" \
    "${CONTAINER_IMAGE_PATH}" \
    python src/eval.py \
        eval.mode=offline_sample \
        experiment=sample_train_set # the sample_train_set.example.yaml, you need to remove .example