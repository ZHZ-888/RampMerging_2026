#!/bin/bash
#SBATCH --job-name=ramp_spr_sensitivity
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=output-logs/ramp_spr_sensitivity_%A_%a.out
#SBATCH --error=error-logs/ramp_spr_sensitivity_%A_%a.err

# From the repository root:
# python -m tools.ramp_spr_sensitivity preflight data/raw/ramp_spr_sensitivity/preflight.csv
# mkdir -p output-logs error-logs
# sbatch --array=0-649%30 slurm/ramp_spr_sensitivity.slurm
# python -m tools.ramp_spr_sensitivity summarize data/raw/ramp_spr_sensitivity
# Retry selected task IDs into the same results directory:
# sbatch --array=ID_LIST%30 slurm/ramp_spr_sensitivity.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this job as an array}
RESULT_DIR=${RESULT_DIR:-data/raw/ramp_spr_sensitivity}
export TRAJ_DIR=${RAMP_SPR_TRAJ_BASE:-/nesi/nobackup/uoa04480/traj/ramp_spr_sensitivity}

RAMP_SPRS=(0.1 0.3 0.5 0.7 0.9)
RAMP_DEMANDS=(200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
NR=${#RAMP_DEMANDS[@]}
NS=${#SEEDS[@]}
NT=$(( ${#RAMP_SPRS[@]} * NR * NS ))
if (( TASK_ID < 0 || TASK_ID >= NT )); then
    echo "Task ID must be in 0–$((NT - 1))." >&2
    exit 1
fi

RAMP_SPR=${RAMP_SPRS[$((TASK_ID / (NR * NS)))]}
R_FR=${RAMP_DEMANDS[$(((TASK_ID / NS) % NR))]}
SEED=${SEEDS[$((TASK_ID % NS))]}

export PYTHONHASHSEED="$SEED"
export COMP_DIR="$TRAJ_DIR/computation"
mkdir -p "$RESULT_DIR" "$TRAJ_DIR"
OUT_CSV="$RESULT_DIR/task_$TASK_ID.csv"
TEMP_DIR=$(mktemp -d "$RESULT_DIR/.task_${TASK_ID}.XXXXXX")
TEMP_CSV="$TEMP_DIR/result.csv"
trap 'rm -f -- "$TEMP_CSV"; rmdir -- "$TEMP_DIR"' EXIT

if python run.py \
    --algo mpgc_multi_lane \
    --av_p 0.1 --r_fr "$R_FR" --m_fr 1500 --seed "$SEED" --st 1500 \
    --ramp_spr "$RAMP_SPR" --fc_mode full --tsg_mode predict \
    --max_team_size 12 --loss_rate 0 --lc \
    --out_csv "$TEMP_CSV"; then
    mv -- "$TEMP_CSV" "$OUT_CSV"
    rm -f -- "$RESULT_DIR/failure_$TASK_ID.txt"
    echo "Finished TASK_ID=$TASK_ID"
else
    status=$?
    printf 'Exit status %s. See error-logs/ramp_spr_sensitivity_%s_%s.err\n' \
        "$status" "${SLURM_ARRAY_JOB_ID:-unknown}" "$TASK_ID" > "$RESULT_DIR/failure_$TASK_ID.txt"
    exit "$status"
fi
