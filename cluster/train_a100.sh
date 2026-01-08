#!/bin/bash

#SBATCH --job-name=chroma-train
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --output=logs/jobs/train_%j.out
#SBATCH --error=logs/jobs/train_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================================
# CHROMA Training Script - A100
# ============================================================================

echo "=========================================="
echo "CHROMA Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="

# --- 1. Data Staging -----------------------------------------------------------
DATA_SRC=$HOME/data/
DATA_DST=/scratch/data/

echo "========== STAGING DATA =========="
mkdir -p "$DATA_DST"
echo "DATA_SRC: $DATA_SRC"
echo "DATA_DST: $DATA_DST"

rclone copy "$DATA_SRC" "$DATA_DST" \
            --transfers=20  --checkers=20 \
            --stats=60s --progress

echo "Data copy completed at: $(date)"
echo "=================================="

# --- 2. Setup Environment ---------------------------------------------------
echo ""
echo "========== ENVIRONMENT SETUP =========="

# Activate virtual environment (uv or venv)
if [ -d ".venv" ]; then
    echo "Activating .venv"
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Please run 'uv sync' first."
    exit 1
fi

# --- 3. Training ------------------------------------------------------------
echo ""
echo "========== TRAINING START =========="

# Config file (can be passed as argument or use default)
CONFIG_PATH=${1:-"configs/a100_resent50.yaml"}
echo "Using config: $CONFIG_PATH"

# Optional: resume from checkpoint
CKPT_PATH=${2:-""}
if [ -n "$CKPT_PATH" ]; then
    echo "Resuming from checkpoint: $CKPT_PATH"
    python train.py --config "$CONFIG_PATH" --ckpt_path "$CKPT_PATH"
else
    python train.py --config "$CONFIG_PATH"
fi

EXIT_CODE=$?
echo "=========================================="

# --- 4. Cleanup & Summary ---------------------------------------------------
echo ""
echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "Ended at: $(date)"
echo "=========================================="

# Print GPU memory usage
nvidia-smi

exit $EXIT_CODE

