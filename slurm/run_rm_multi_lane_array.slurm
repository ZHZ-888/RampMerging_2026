#!/bin/bash
#SBATCH --job-name=rm_ml_array            # Job name shown in the queue
#SBATCH --time=04:00:00                  # Walltime limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=4G                         # Memory per task
#SBATCH --output=output-logs/out_%A_%a.txt   # Standard output log
#SBATCH --error=error-logs/err_%A_%a.txt    # Standard error log
# NOTE: --array is specified at submission time

set -euo pipefail

# Ensure relative paths are resolved from the submission directory
cd "$SLURM_SUBMIT_DIR"

# ================================
# Load Python environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Create log directories if not exist
mkdir -p output-logs error-logs

# ================================
# Trajectory (XML) output directory
# Use nobackup due to large file size
# Read by Python via environment variable TRAJ_DIR
# ================================
export TRAJ_DIR="/nesi/nobackup/uoa04480/traj"
mkdir -p "$TRAJ_DIR"
echo "TRAJ_DIR=$TRAJ_DIR"

# ================================
# Experiment parameter grid
# ================================
MAINLINE=1500
RAMPS=(200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400)
SEEDS=(1 2 3 4 5)

NR=${#RAMPS[@]}
NS=${#SEEDS[@]}
NT=$((NR * NS))   # 50 tasks

# Check SLURM array task ID
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-$((NT-1))"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Validate task ID range
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$NT" ]; then
  echo "Error: TASK_ID=$TASK_ID out of range (valid 0..$((NT-1)))"
  exit 1
fi

# Map array task ID to (ramp demand, seed)
r_idx=$(( TASK_ID / NS ))
s_idx=$(( TASK_ID % NS ))

RAMP=${RAMPS[$r_idx]}
SEED=${SEEDS[$s_idx]}

echo "TASK_ID=$TASK_ID | ramp=$RAMP | mainline=$MAINLINE | seed=$SEED"

# ================================
# Per-task CSV output (safe for parallel jobs)
# Each task writes to its own CSV file
# ================================
OUT_CSV="data/raw/tmp/rm_task_${TASK_ID}.csv"
mkdir -p data/raw/tmp

# ================================
# Run RM multi-lane simulation
# ================================
python run.py \
  --algo rm_multi_lane \
  --r_fr "$RAMP" \
  --m_fr "$MAINLINE" \
  --seed "$SEED" \
  --out_csv "$OUT_CSV"
