#!/bin/bash
# All this code is from Claude
# Submit the area_commonmode_v2 chain. Every stage exits 0 even on refusal,
# so the final report ALWAYS runs.
#   DRY_RUN=1 bash src/instrument_v2/submit_area_commonmode_v2.sh
#   bash src/instrument_v2/submit_area_commonmode_v2.sh
# Stages: diag -> screen[4] -> select -> confirm[3] -> finetune[36] -> final
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
RUN_TAG=acm2_$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/instrument_v2_area_commonmode_v2/$RUN_TAG
ART_DIR=artifacts/instrument_v2/area_commonmode_v2
RUNNER=src/instrument_v2/run_area_commonmode_v2.sh
PARTITION=${PARTITION:-ou_mki_gpu}
COMMON="-p $PARTITION --gres=gpu:1 -N 1 -c 8 --mem=64G"
EXPORTS="ALL,PY=$PY,REPO=$REPO,PYTHONUNBUFFERED=1"

echo "run=$RUN_TAG commit=$(git rev-parse --short HEAD) partition=$PARTITION"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN -- would submit:"
  echo "  diag     $COMMON -t 02:00:00"
  echo "  screen   $COMMON -t 03:00:00 --array=0-3%4    afterok:diag"
  echo "  select   $COMMON -t 00:20:00                  afterok:screen"
  echo "  confirm  $COMMON -t 06:00:00 --array=0-2%3    afterok:select"
  echo "  finetune $COMMON -t 04:00:00 --array=0-35%6   afterok:confirm"
  echo "  final    $COMMON -t 02:00:00                  afterok:finetune"
  exit 0
fi

mkdir -p "$LOGDIR" "$ART_DIR"

DIAG=$(sbatch --parsable $COMMON -t 02:00:00 -J acm2_diag \
  -o "$LOGDIR/diag_%j.out" --export="$EXPORTS,STAGE=diag" "$RUNNER")
SCREEN=$(sbatch --parsable $COMMON -t 03:00:00 -J acm2_screen --array=0-3%4 \
  --dependency=afterok:$DIAG \
  -o "$LOGDIR/screen_%A_%a.out" --export="$EXPORTS,STAGE=screen" "$RUNNER")
SELECT=$(sbatch --parsable $COMMON -t 00:20:00 -J acm2_select \
  --dependency=afterok:$SCREEN \
  -o "$LOGDIR/select_%j.out" --export="$EXPORTS,STAGE=select" "$RUNNER")
CONFIRM=$(sbatch --parsable $COMMON -t 06:00:00 -J acm2_confirm --array=0-2%3 \
  --dependency=afterok:$SELECT \
  -o "$LOGDIR/confirm_%A_%a.out" --export="$EXPORTS,STAGE=confirm" "$RUNNER")
FT=$(sbatch --parsable $COMMON -t 04:00:00 -J acm2_finetune --array=0-35%6 \
  --dependency=afterok:$CONFIRM \
  -o "$LOGDIR/finetune_%A_%a.out" --export="$EXPORTS,STAGE=finetune" "$RUNNER")
FINAL=$(sbatch --parsable $COMMON -t 02:00:00 -J acm2_final \
  --dependency=afterok:$FT \
  -o "$LOGDIR/final_%j.out" --export="$EXPORTS,STAGE=final" "$RUNNER")

cat > "$ART_DIR/job_manifest_$RUN_TAG.json" <<EOF
{
  "run_tag": "$RUN_TAG",
  "git_commit": "$(git rev-parse --short HEAD)",
  "partition": "$PARTITION",
  "log_dir": "$LOGDIR",
  "jobs": {
    "diag":     {"id": "$DIAG"},
    "screen":   {"id": "$SCREEN",  "array": "0-3%4"},
    "select":   {"id": "$SELECT"},
    "confirm":  {"id": "$CONFIRM", "array": "0-2%3"},
    "finetune": {"id": "$FT",      "array": "0-35%6"},
    "final":    {"id": "$FINAL"}
  },
  "final_report": "$ART_DIR/final_summary.md"
}
EOF
echo "submitted: diag=$DIAG screen=$SCREEN select=$SELECT confirm=$CONFIRM finetune=$FT final=$FINAL"
echo "manifest:  $ART_DIR/job_manifest_$RUN_TAG.json"
echo "report:    cat $ART_DIR/final_summary.md"
