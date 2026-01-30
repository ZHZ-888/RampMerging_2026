#!/bin/bash
#SBATCH --job-name=mpgc_ml_array          # Job name shown in the queue
#SBATCH --time=00:45:00                  # Walltime limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=4G                         # Memory per task
#SBATCH --output=output-logs/out_%A_%a.txt   # Standard output log
#SBATCH --error=error-logs/err_%A_%a.txt    # Standard error log
# NOTE: Submit with --array=0-14  (3 AV_Ps * 5 Seeds = 15 jobs)

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
AV_P_VALS=(0.1 0.2 0.3) # AV penetartion
SEEDS=(16 17 18 19 20)                         # Random seeds

NAV=${#AV_P_VALS[@]}   # Number of AV Settings (3)
NS=${#SEEDS[@]}        # Number of Seeds (5)
NT=$((NAV * NS))       # Total Tasks (15)

# Check SLURM array task ID
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-$((NT-1))"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Map array task ID to (av_p, seed)
av_idx=$(( TASK_ID / NS ))
s_idx=$(( TASK_ID % NS ))

AV_P=${AV_P_VALS[$av_idx]}
SEED=${SEEDS[$s_idx]}

echo "TASK_ID=$TASK_ID | AV_P=$AV_P | Seed=$SEED | (Indices: av=$av_idx, s=$s_idx)"

# ================================
# Per-task CSV output (safe for parallel jobs)
# Each task writes to its own CSV file
# ================================
OUT_CSV="data/raw/tmp/rf_data_avp${AV_P}_seed${SEED}.csv"
mkdir -p data/raw/tmp

# ================================
# Run mpgc_multi_lane_collect_RF_at_data multi-lane simulation
# ================================
python run.py \
  --algo mpgc_multi_lane_collect_RF_at_data \
  --av_p "$AV_P" \
  --seed "$SEED" \
  --out_csv "$OUT_CSV"
