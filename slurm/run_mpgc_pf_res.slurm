#!/bin/bash
#SBATCH --job-name=pfz_res
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=output-logs/pfz_res_%A_%a.out
#SBATCH --error=error-logs/pfz_res_%A_%a.err

# Submit with:
# sbatch --array=0-89 run_pfz_res.sl

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ================================
# Python environment
# ================================
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate

# Deterministic behavior
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ================================
# Log directories
# ================================
mkdir -p output-logs
mkdir -p error-logs

# ================================
# Trajectory output directory
# ================================
export TRAJ_DIR="/nesi/nobackup/uoa04480/traj"
mkdir -p "$TRAJ_DIR"

echo "TRAJ_DIR=$TRAJ_DIR"

# ================================
# Fixed experiment settings
# ================================
R_FR=0
M_FR=1500

# ================================
# Parameter sweep
# ================================
TSG_MODES=(off fix predict)
AV_P_VALS=(0.1 0.2 0.3)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

NM=${#TSG_MODES[@]}
NAV=${#AV_P_VALS[@]}
NS=${#SEEDS[@]}
NT=$((NM * NAV * NS))

# ================================
# Check task id
# ================================
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    echo "Submit with: sbatch --array=0-$((NT-1)) run_pfz_res.sl"
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}

# ================================
# Map task id -> (av_p, seed)
# ================================
mode_idx=$(( TASK_ID / (NAV * NS) ))
rem=$(( TASK_ID % (NAV * NS) ))
av_idx=$(( rem / NS ))
s_idx=$(( rem % NS ))

TSG_MODE=${TSG_MODES[$mode_idx]}
AV_P=${AV_P_VALS[$av_idx]}
SEED=${SEEDS[$s_idx]}

echo "=================================="
echo "TASK_ID : $TASK_ID"
echo "TSG_MODE: $TSG_MODE"
echo "AV_P    : $AV_P"
echo "SEED    : $SEED"
echo "R_FR    : $R_FR"
echo "M_FR    : $M_FR"
echo "=================================="

# ================================
# Output CSV
# ================================
mkdir -p data/raw/pfz_res

OUT_CSV="data/raw/pfz_res/pfz_res_task_${TASK_ID}.csv"
# ================================
# Run simulation
# ================================
python run.py \
    --algo mpgc_multi_lane \
    --av_p "$AV_P" \
    --r_fr "$R_FR" \
    --m_fr "$M_FR" \
    --seed "$SEED" \
    --tsg_mode "$TSG_MODE" \
    --out_csv "$OUT_CSV"

echo "Finished TASK_ID=$TASK_ID"
