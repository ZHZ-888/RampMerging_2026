#!/bin/bash
#SBATCH --job-name=tsg_training
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=train-logs/tsg_out_%j.txt
#SBATCH --error=train-logs/tsg_err_%j.txt

set -euo pipefail

# Ensure the script runs from the project root directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# HPC Storage Configuration
# ================================
export RUN_DIR="/nesi/nobackup/uoa04480/TSG_Training_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
mkdir -p train-logs

echo "All TSG logs and models will be saved to: $RUN_DIR"

# ================================
# Load Environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:${SLURM_SUBMIT_DIR}"

# Keep Torch/SUMO CPU usage stable
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ================================
# Run shared TSG multi-scenario training
# ================================
stdbuf -oL python -u run.py \
  --algo mpgc_multi_lane_TSG_training \
  --lr 0.0001 \
  --train_interval 16 \
  --hidden_layer 64 64 \
  --st 6000