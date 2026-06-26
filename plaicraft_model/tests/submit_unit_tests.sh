#!/bin/bash
#SBATCH --partition=ubcml
#SBATCH --job-name=unit_test_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --mem=40gb
#SBATCH --output=./logs/unit_test/slurm-%j.out
#SBATCH --error=./logs/unit_test/slurm-%j.err
#SBATCH --export=ALL

# =============================================================================
# Environment Setup
# =============================================================================
source .env
working_dir="${PROJECT_ROOT}"
cd ${working_dir}

export TRITON_CACHE_DIR=$HOME/.triton/cache
export TMPDIR=/tmp

# Ensure log directory exists
mkdir -p logs/unit_test

CONTAINERS="${CONTAINERS_PATH}"
SINGULARITY_IMAGE="plaicraft_latest.sif"

# =============================================================================
# Run Unit Tests inside Singularity
# =============================================================================
srun --unbuffered /opt/singularity-4.2.1/bin/singularity exec --nv \
    -B /home -B /scratch-ssd \
    -B /ubc/cs/research/ubc_ml/plaicraft \
    -B /ubc/cs/research/plai-scratch \
    -B /ubc/cs/research/fwtemp/ \
    -B "$(pwd)" --pwd "$(pwd)" \
    -B /etc/ssh:/etc/ssh \
    -B "$(command -v ssh)":/usr/bin/ssh \
    -B "$(command -v scp)":/usr/bin/scp \
    -B "$(command -v ssh-keyscan)":/usr/bin/ssh-keyscan \
    --env PYTHONPATH="${working_dir}/src" \
    "${CONTAINERS}/${SINGULARITY_IMAGE}" \
    bash tests/run_unit_tests.sh