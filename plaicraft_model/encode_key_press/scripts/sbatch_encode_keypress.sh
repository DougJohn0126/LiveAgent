#!/bin/bash
#SBATCH --job-name=plaicraft_encode_keypress_job
#SBATCH --partition=plai
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=/ubc/cs/research/ubc_ml/plaicraft/slurm_logs/encode_keypress/slurm-outputs/slurm-%A_%a.out
#SBATCH --error=/ubc/cs/research/ubc_ml/plaicraft/slurm_logs/encode_keypress/slurm-errors/slurm-%A_%a.err
#SBATCH --export=ALL
#SBATCH --array=1-100

###############################################################################
#                            Environment Setup                                #
###############################################################################

# Load necessary modules or activate environment if needed
# Example:
# module load python/3.8

# Activate the Conda environment
export PATH=/opt/slurm/bin:$PATH
source /ubc/cs/research/plai-scratch/plaicraft/conda/etc/profile.d/conda.sh
conda activate prep-data-3

# Verify Conda environment
echo "Conda root directory: $(conda info --base)"
echo "Activated Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

###############################################################################
#                             CUDA Availability Check                         #
###############################################################################

# Check if CUDA is available
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "CUDA is not available on this node for an unknown reason! Waiting 60 minutes, and exiting job."
    sleep 60
    exit 1
else
    echo "CUDA is available, proceeding with the job!"
fi

###############################################################################
#                              Set Cache Directories                          #
###############################################################################

# Set the cache directories (REQUIRED)
export HF_HOME=/ubc/cs/research/plai-scratch/plaicraft/.cache/huggingface
export TORCH_HOME=/ubc/cs/research/plai-scratch/plaicraft/.cache/torch
export TRANSFORMERS_CACHE=/ubc/cs/research/plai-scratch/plaicraft/.cache/transformers
export PYTORCH_LIGHTNING_HOME=/ubc/cs/research/plai-scratch/plaicraft/.cache/pytorch-lightning
export XDG_CACHE_HOME=/ubc/cs/research/plai-scratch/plaicraft/.cache
export MPLCONFIGDIR=/ubc/cs/research/plai-scratch/plaicraft/.cache/matplotlib

###############################################################################
#                             Retrieve Environment Variables                  #
###############################################################################

# Retrieve environment variables
TOKENS_FILE=${TOKENS_FILE}
PARENT_DIR=${PARENT_DIR}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT}    # Ensure this is set to the path of your model checkpoint

echo "TOKENS_FILE: $TOKENS_FILE"
echo "PARENT_DIR: $PARENT_DIR"
echo "MODEL_CHECKPOINT: $MODEL_CHECKPOINT"

# Validate essential environment variables
if [ -z "$TOKENS_FILE" ]; then
    echo "[ERROR] TOKENS_FILE environment variable is not set."
    exit 1
fi

if [ -z "$PARENT_DIR" ]; then
    echo "[ERROR] PARENT_DIR environment variable is not set."
    exit 1
fi

if [ -z "$MODEL_CHECKPOINT" ]; then
    echo "[ERROR] MODEL_CHECKPOINT environment variable is not set."
    exit 1
fi

###############################################################################
#                             Token Processing                                 #
###############################################################################

# Get the current token based on SLURM_ARRAY_TASK_ID
TOKEN_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TOKENS_FILE")
if [ -z "$TOKEN_LINE" ]; then
    echo "[ERROR] No token found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

# Extract hashedEmail and token from the token line
IFS='/' read -r HASHED_EMAIL TOKEN <<< "$TOKEN_LINE"

echo "Processing token: $HASHED_EMAIL/$TOKEN"

###############################################################################
#                             Define Directories                               #
###############################################################################

# Define the session database path

SESSION_DB_PATH="$PARENT_DIR/processed/$HASHED_EMAIL/$TOKEN/$TOKEN.db"
LOGS_DIR="$PARENT_DIR/logs/$HASHED_EMAIL/$TOKEN"
STATUS_DIR="$PARENT_DIR/status/$HASHED_EMAIL/$TOKEN"

# Create logs and status directories if they don't exist
mkdir -p "$LOGS_DIR"
mkdir -p "$STATUS_DIR"

# Initialize status file to 'processing'
STATUS_FILE="$STATUS_DIR/encode_keypress.status"
echo "processing" > "$STATUS_FILE"

# Define log file paths
ENCODE_KEYPRESS_OUT="$LOGS_DIR/encode_keypress_$SLURM_JOB_ID.out"
ENCODE_KEYPRESS_ERR="$LOGS_DIR/encode_keypress_$SLURM_JOB_ID.err"

###############################################################################
#                             Status Update Function                          #
###############################################################################

# Function to update status
update_status() {
    local status=$1
    echo "$status" > "$STATUS_FILE"
}

###############################################################################
#                             Run Encoding Script                             #
###############################################################################

echo "Starting key_press encoding for token '$HASHED_EMAIL/$TOKEN'..."

python /ubc/cs/research/ubc_ml/plaicraft/plaicraft-data-preprocessing/encode_key_press/main.py \
    --session_db_path "$SESSION_DB_PATH" \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --batch_size 256 \
    --device cuda \
    --verbose \
    > "$ENCODE_KEYPRESS_OUT" 2> "$ENCODE_KEYPRESS_ERR"

EXIT_STATUS=$?

###############################################################################
#                             Update Status Based on Outcome                   #
###############################################################################

if [ $EXIT_STATUS -eq 0 ]; then
    update_status "success"
    echo "[INFO] Successfully encoded key presses for token '$HASHED_EMAIL/$TOKEN'."
else
    update_status "failed"
    echo "[ERROR] key_press encoding failed for token '$HASHED_EMAIL/$TOKEN'. Check logs."
fi

exit $EXIT_STATUS
