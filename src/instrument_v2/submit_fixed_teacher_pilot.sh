#!/bin/bash
# All this code is from Claude
# Submit the fixed-regional-teacher pilot chain (seed 0, validation-only).
#   DRY_RUN=1 bash src/instrument_v2/submit_fixed_teacher_pilot.sh
#   bash src/instrument_v2/submit_fixed_teacher_pilot.sh
# Stages: teacher -> student -> probes -> finetune[9] -> final
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
RUN_TAG=frt_$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/instrument_v2_fixed_teacher/$RUN_TAG
ART_DIR=artifacts/instrument_v2/fixed_regional_teacher_v1
RUNNER=src/instrument_v2/run_fixed_teacher_pilot.sh
PARTITION=${PARTITION:-ou_mki_gpu}
COMMON="-p $PARTITION --gres=gpu:1 -N 1 -c 8 --mem=64G"
EXPORTS="ALL,PY=$PY,REPO=$REPO,PYTHONUNBUFFERED=1"

echo "run=$RUN_TAG commit=$(git rev-parse --short HEAD) partition=$PARTITION"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN -- would submit:"
  echo "  teacher  $COMMON -t 04:00:00"
  echo "  student  $COMMON -t 04:00:00                afterok:teacher"
  echo "  probes   $COMMON -t 01:00:00                afterok:student"
  echo "  finetune $COMMON -t 04:00:00 --array=0-8%3  afterok:probes"
  echo "  final    $COMMON -t 01:00:00                afterok:finetune"
  exit 0
fi

mkdir -p "$LOGDIR" "$ART_DIR"

TEACHER=$(sbatch --parsable $COMMON -t 04:00:00 -J frt_teacher \
  -o "$LOGDIR/teacher_%j.out" --export="$EXPORTS,STAGE=teacher" "$RUNNER")
STUDENT=$(sbatch --parsable $COMMON -t 04:00:00 -J frt_student \
  --dependency=afterok:$TEACHER \
  -o "$LOGDIR/student_%j.out" --export="$EXPORTS,STAGE=student" "$RUNNER")
PROBES=$(sbatch --parsable $COMMON -t 01:00:00 -J frt_probes \
  --dependency=afterok:$STUDENT \
  -o "$LOGDIR/probes_%j.out" --export="$EXPORTS,STAGE=probes" "$RUNNER")
FT=$(sbatch --parsable $COMMON -t 04:00:00 -J frt_finetune --array=0-8%3 \
  --dependency=afterok:$PROBES \
  -o "$LOGDIR/finetune_%A_%a.out" --export="$EXPORTS,STAGE=finetune" "$RUNNER")
FINAL=$(sbatch --parsable $COMMON -t 01:00:00 -J frt_final \
  --dependency=afterok:$FT \
  -o "$LOGDIR/final_%j.out" --export="$EXPORTS,STAGE=final" "$RUNNER")

cat > "$ART_DIR/job_manifest_$RUN_TAG.json" <<EOF
{
  "run_tag": "$RUN_TAG",
  "git_commit": "$(git rev-parse --short HEAD)",
  "partition": "$PARTITION",
  "log_dir": "$LOGDIR",
  "jobs": {
    "teacher":  {"id": "$TEACHER"},
    "student":  {"id": "$STUDENT"},
    "probes":   {"id": "$PROBES"},
    "finetune": {"id": "$FT", "array": "0-8%3"},
    "final":    {"id": "$FINAL"}
  },
  "final_report": "$ART_DIR/final_summary.md"
}
EOF
echo "submitted: teacher=$TEACHER student=$STUDENT probes=$PROBES finetune=$FT final=$FINAL"
echo "monitor:   tail -f $LOGDIR/teacher_$TEACHER.out"
echo "report:    cat $ART_DIR/final_summary.md"
