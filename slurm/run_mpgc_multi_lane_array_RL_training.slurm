#!/bin/bash
#SBATCH --job-name=mpgc_training            # Job name for tracking
#SBATCH --time=12:00:00                     # Increased time limit for longer simulation
#SBATCH --cpus-per-task=4                   # CPU cores per task for Torch/SUMO
#SBATCH --mem=12G                           # Memory allocation for Replay Buffer
#SBATCH --output=train-logs/out_%A_%a.txt   # Standard output log
#SBATCH --error=train-logs/err_%A_%a.txt     # Error log
# NOTE: Submit with --array=0-8 (for 3 LRs * 3 Seeds)

set -euo pipefail

# Ensure the script runs from the project root directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# Load Environment
# ================================
# Load Python module and activate virtual environment
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Add current directory to PYTHONPATH to ensure internal modules are importable
export PYTHONPATH="${PYTHONPATH:-}:${SLURM_SUBMIT_DIR}"

# Optional: Specify SUMO_HOME if required by the cluster
# export SUMO_HOME="/path/to/your/sumo" 

# Create necessary directories for logging and data
mkdir -p train-logs data/raw/training

# ================================
# Parameter Mapping (Grid Search)
# ================================
LR_VALS=(0.001 0.0005 0.0001)               # Candidate Learning Rates
SEEDS=(21 22 23)                            # Different Random Seeds
AGENT_TYPE="CA"                             # Targeted RL Agent (CA or SA)

NS=${#SEEDS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Calculate indices to map Task ID to specific hyperparameters
lr_idx=$(( TASK_ID / NS ))
s_idx=$(( TASK_ID % NS ))

LR=${LR_VALS[$lr_idx]}
SEED=${SEEDS[$s_idx]}

echo "STARTING TRAINING: ID=$TASK_ID | LR=$LR | SEED=$SEED | AGENT=$AGENT_TYPE"

# ================================
# Execution
# ================================
# Use 'stdbuf -oL' to ensure real-time logging output to the .txt file
# Simulation time set to 120000 (1200 * 100)
stdbuf -oL python run.py \
  --algo mpgc_multi_lane_RL_agent_training \
  --train_agent "$AGENT_TYPE" \
  --lr "$LR" \
  --seed "$SEED" \
  --batch_epoch 16 5 \
  --hidden_layer 32 32 \
  --st 120000 \
  --av_p 0.3