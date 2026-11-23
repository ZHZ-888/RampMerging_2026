#!/bin/bash
#SBATCH --job-name=cvmc_ml
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=output-logs/cvmc_%A_%a.out
#SBATCH --error=error-logs/cvmc_%A_%a.err

# From project root: mkdir -p output-logs error-logs
# sbatch --array=0-389 slurm/run_cvmc_multi_lane_array.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

AV_PS=(0.1 0.2 0.3)
RAMPS=(200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
RAMP_SPR=${RAMP_SPR:-0.7}
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit as an array}
NS=${#SEEDS[@]}
NR=${#RAMPS[@]}
NT=$((${#AV_PS[@]} * NR * NS))
if (( TASK_ID < 0 || TASK_ID >= NT )); then
    echo "Task must be in 0..$((NT-1))"
    exit 1
fi
AV_P=${AV_PS[$((TASK_ID / (NR * NS)))]}
RAMP=${RAMPS[$(((TASK_ID / NS) % NR))]}
SEED=${SEEDS[$((TASK_ID % NS))]}
export PYTHONHASHSEED="$SEED"
export TRAJ_DIR="/nesi/nobackup/uoa04480/traj/cvmc/${SLURM_ARRAY_JOB_ID}"
RESULT_DIR="data/raw/cvmc/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$TRAJ_DIR" "$RESULT_DIR"
python run.py --algo cvmc_multi_lane --av_p "$AV_P" --r_fr "$RAMP" \
    --m_fr 1500 --seed "$SEED" --st 1500 --ramp_spr "$RAMP_SPR" \
    --out_csv "$RESULT_DIR/task_${TASK_ID}.csv" \
    --log_dir "$RESULT_DIR/control_logs"
