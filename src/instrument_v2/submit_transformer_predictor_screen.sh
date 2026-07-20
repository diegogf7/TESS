#!/bin/bash
# All this code is from Claude
# Submit the transformer-predictor screen (single job, ~1 h).
#   DRY_RUN=1 bash src/instrument_v2/submit_transformer_predictor_screen.sh
#   bash src/instrument_v2/submit_transformer_predictor_screen.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
PARTITION=${PARTITION:-ou_mki_gpu}
COMMON="-p $PARTITION --gres=gpu:1 -N 1 -c 8 --mem=64G -t 03:00:00"
RUNNER=src/instrument_v2/run_transformer_predictor_screen.sh
LOGDIR=logs/instrument_v2_transformer_screen

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN -- would submit: sbatch $COMMON $RUNNER"
  echo "  prereqs: fixed_regional_teacher_v1 teacher+MLP selections must exist"
  echo "  report:  artifacts/instrument_v2/transformer_predictor_screen/final_summary.md"
  exit 0
fi

cd "$REPO"
mkdir -p "$LOGDIR"
JOB=$(sbatch --parsable $COMMON -J tx_screen \
  -o "$LOGDIR/screen_%j.out" \
  --export="ALL,PY=$PY,REPO=$REPO,PYTHONUNBUFFERED=1" "$RUNNER")
echo "submitted: $JOB"
echo "monitor:   tail -f $LOGDIR/screen_$JOB.out"
echo "report:    cat artifacts/instrument_v2/transformer_predictor_screen/final_summary.md"
