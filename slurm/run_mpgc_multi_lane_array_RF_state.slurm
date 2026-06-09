#!/bin/bash
#SBATCH --job-name=rf_state_data_collection
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-59
#SBATCH --output=output-logs/out_%A_%a.txt
#SBATCH --error=error-logs/err_%A_%a.txt
# 60 jobs: 3 AV penetration rates × 20 random seeds

set -euo pipefail

# Ensure relative paths are resolved from the submission directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# Load Python environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Ensure deterministic CPU behaviour
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create output directories
mkdir -p output-logs error-logs
mkdir -p data/features/tmp

# ================================
# Experiment parameter grid
# ================================
AV_P_VALS=(0.1 0.2 0.3)
SEEDS=(11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

NAV=${#AV_P_VALS[@]}
NS=${#SEEDS[@]}
NT=$((NAV * NS))

# Check SLURM array task ID
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set."
  echo "Submit with: sbatch --array=0-$((NT-1)) run_array_RF_state_data.slurm"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Map task ID to (av_p, seed)
av_idx=$(( TASK_ID / NS ))
s_idx=$(( TASK_ID % NS ))

AV_P=${AV_P_VALS[$av_idx]}
SEED=${SEEDS[$s_idx]}

echo "TASK_ID=$TASK_ID | AV_P=$AV_P | SEED=$SEED"

# ================================
# Output CSV for this task
# ================================
OUT_CSV="data/features/tmp/df_RF_state_data_av${AV_P}_seed${SEED}.csv"

# ================================
# Run RF data collection
# ================================
python run.py \
  --algo mpgc_multi_lane_array_RF_state \
  --av_p "$AV_P" \
  --seed "$SEED" \
  --out_csv "$OUT_CSV"