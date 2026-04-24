#!/bin/bash
#SBATCH --job-name=mpgc_training            # Job name for tracking
#SBATCH --time=16:00:00                     # Increased time limit for longer simulation
#SBATCH --cpus-per-task=1                   # CPU cores per task for Torch/SUMO
#SBATCH --mem=2G                            # Memory allocation for Replay Buffer
#SBATCH --output=train-logs/out_%A_%a.txt   # Standard output log
#SBATCH --error=train-logs/err_%A_%a.txt    # Error log
#SBATCH --array=0-11
# # NOTE: Submit with --array=0-11 (for 2 agents * 2 LRs * 3 Seeds)//NOTE: Submit with --array=0-17 (for 2 agents * 3 LRs * 3 Seeds)

set -euo pipefail

# Ensure the script runs from the project root directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# HPC Storage Configuration
# ================================
export RUN_DIR="/nesi/nobackup/uoa04480/RL_Training_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_DIR"
echo "All heavy logs and models will be saved to: $RUN_DIR"

# ================================
# Load Environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:${SLURM_SUBMIT_DIR}"

mkdir -p train-logs

# ================================
# Parameter Mapping (Grid Search)
# ================================
LR_VALS=(0.005 0.01)               # Candidate Learning Rates # LR_VALS=(0.001 0.0005 0.0001)
SEEDS=(21 22 23)                            # Different Random Seeds
AGENT_LIST=("CA" "SA")                      # Fixed array name for indexing

TASK_ID=${SLURM_ARRAY_TASK_ID}

# 1. Determine Agent Type (Task 0-5: CA, Task 6-11: SA)//(Task 0-8: CA, Task 9-17: SA)
agent_idx=$(( TASK_ID / 6 )) # agent_idx=$(( TASK_ID / 9 ))
AGENT_TYPE=${AGENT_LIST[$agent_idx]}        # Correctly reference AGENT_LIST

# 2. Assign AV Penetration Rate based on Agent Type
if [ "$AGENT_TYPE" == "SA" ]; then
    AV_P=0.1                                # SA runs with 10% AV penetration
else
    AV_P=0.3                                # CA runs with 30% AV penetration
fi

# 3. Determine Hyperparameters (LR and Seed)
# rem=$(( TASK_ID % 9 ))
rem=$(( TASK_ID % 6 ))
lr_idx=$(( rem / 3 ))
s_idx=$(( rem % 3 ))

LR=${LR_VALS[$lr_idx]}
SEED=${SEEDS[$s_idx]}

echo "------------------------------------------------"
echo "JOB START: Array ID $TASK_ID"
echo "CONFIG: Agent=$AGENT_TYPE | AV_P=$AV_P | LR=$LR | Seed=$SEED"
echo "------------------------------------------------"

# ================================
# Execution
# ================================
stdbuf -oL python -u run.py \
  --algo mpgc_multi_lane_RL_agent_training \
  --train_agent "$AGENT_TYPE" \
  --lr "$LR" \
  --seed "$SEED" \
  --batch_epoch 16 5 \
  --hidden_layer 32 32 \
  --st 120000 \
  --av_p "$AV_P"