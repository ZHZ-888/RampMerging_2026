#!/bin/bash
#SBATCH --job-name=cvmc_ml
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=output-logs/out_%A_%a.txt
#SBATCH --error=error-logs/err_%A_%a.txt

# From the project root, create output-logs and error-logs before submission.
# sbatch --array=0-389%30 slurm/run_cvmc_multi_lane_array.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

AV_PS=(0.1 0.2 0.3)
RAMPS=(200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit as an array}
NR=${#RAMPS[@]}
NS=${#SEEDS[@]}
NT=$((${#AV_PS[@]} * NR * NS))
if (( TASK_ID < 0 || TASK_ID >= NT )); then
    echo "Task ID must be in 0..$((NT - 1))." >&2
    exit 1
fi
AV_P=${AV_PS[$((TASK_ID / (NR * NS)))]}
RAMP=${RAMPS[$(((TASK_ID / NS) % NR))]}
SEED=${SEEDS[$((TASK_ID % NS))]}

module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED="$SEED"

RESULT_DIR=${COMPARISON_RESULT_DIR:-data/raw/main_res}
SAFETY_DIR=data/raw/upstream_safety
export TRAJ_DIR="${COMPARISON_TRAJ_ROOT:-/nesi/nobackup/uoa04480/traj/ramp_platoon_comparison}/cvmc"
mkdir -p "$RESULT_DIR" "$SAFETY_DIR" "$TRAJ_DIR"
OUT_CSV="$RESULT_DIR/cvmc_task_${TASK_ID}.csv"
SAFETY_CSV="$SAFETY_DIR/cvmc_task_${TASK_ID}_UPsafety.csv"
TEMP_DIR=$(mktemp -d "$RESULT_DIR/.cvmc_${TASK_ID}.XXXXXX")
TEMP_CSV="$TEMP_DIR/result.csv"
TEMP_SAFETY_CSV="$TEMP_DIR/upstream_safety.csv"
trap 'rm -f -- "$TEMP_CSV" "$TEMP_SAFETY_CSV"; rmdir -- "$TEMP_DIR"' EXIT

echo "task=$TASK_ID av_p=$AV_P r_fr=$RAMP seed=$SEED"
python run.py --algo cvmc_multi_lane --av_p "$AV_P" --r_fr "$RAMP" \
    --m_fr 1500 --seed "$SEED" --st 1500 \
    --out_csv "$TEMP_CSV" \
    --safety_out_csv "$TEMP_SAFETY_CSV" \
    --log_dir "$TRAJ_DIR/control_logs"
mv -f -- "$TEMP_CSV" "$OUT_CSV"
mv -f -- "$TEMP_SAFETY_CSV" "$SAFETY_CSV"
