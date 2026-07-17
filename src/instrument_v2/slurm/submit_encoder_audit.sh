#!/bin/bash
# All this code is from Claude
# Submit the online-vs-EMA encoder audit dependency graph in one shot.
#   DRY_RUN=1 bash src/instrument_v2/slurm/submit_encoder_audit.sh
#   bash src/instrument_v2/slurm/submit_encoder_audit.sh
# Stages: preflight -> (fixed + select_online) -> finetune[27] -> select_ft -> final
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
RUN_ID=${RUN_ID:-abl1_encoder_audit}
LOGDIR=logs/instrument_v2_ablation/$RUN_ID
NEW_RUN=artifacts/instrument_v2/ablation/$RUN_ID
RUNNER=src/instrument_v2/slurm/run_encoder_audit_job.sh
PARTITION=${PARTITION:-ou_mki_gpu}
COMMON="-p $PARTITION --gres=gpu:1 -N 1 -c 8 --mem=64G"

eval "$($PY -m src.instrument_v2.ablation_config counts)"
[ "$ONLINE_FINETUNE_TASKS" -eq 27 ] || { echo "expected 27 online finetune tasks, got $ONLINE_FINETUNE_TASKS"; exit 1; }
echo "task grid OK: $ONLINE_FINETUNE_TASKS online finetune tasks"
echo "RUN_ID=$RUN_ID  commit=$(git rev-parse --short HEAD)  partition=$PARTITION"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN -- would submit:"
  echo "  1 preflight     : $COMMON -t 00:20:00"
  echo "  2 fixed         : $COMMON -t 04:00:00, afterok:preflight"
  echo "  2 select_online : $COMMON -t 08:00:00, afterok:preflight"
  echo "  3 finetune      : $COMMON -t 04:00:00 --array=0-26%4, afterok:select_online"
  echo "  4 select_ft     : $COMMON -t 00:20:00, afterok:finetune"
  echo "  5 final         : $COMMON -t 04:00:00 (FINAL_EVAL=YES), afterok:fixed+select_ft"
  for i in 0 13 26; do
    echo "  finetune[$i]: $($PY -m src.instrument_v2.ablation_config online_finetune $i | tr '\n' ' ')"
  done
  exit 0
fi

mkdir -p "$LOGDIR" "$NEW_RUN"
EXPORTS="ALL,RUN_ID=$RUN_ID,PY=$PY,REPO=$REPO"

PRE=$(sbatch --parsable $COMMON -t 00:20:00 -J aud_preflight \
  -o "$LOGDIR/preflight_%j.out" --export="$EXPORTS,STAGE=preflight" "$RUNNER")
FIXED=$(sbatch --parsable $COMMON -t 04:00:00 -J aud_fixed \
  --dependency=afterok:$PRE \
  -o "$LOGDIR/fixed_%j.out" --export="$EXPORTS,STAGE=fixed" "$RUNNER")
SELON=$(sbatch --parsable $COMMON -t 08:00:00 -J aud_selonline \
  --dependency=afterok:$PRE \
  -o "$LOGDIR/select_online_%j.out" --export="$EXPORTS,STAGE=select_online" "$RUNNER")
FT=$(sbatch --parsable $COMMON -t 04:00:00 -J aud_finetune --array=0-26%4 \
  --dependency=afterok:$SELON \
  -o "$LOGDIR/finetune_%A_%a.out" --export="$EXPORTS,STAGE=finetune" "$RUNNER")
SELFT=$(sbatch --parsable $COMMON -t 00:20:00 -J aud_selft \
  --dependency=afterok:$FT \
  -o "$LOGDIR/select_ft_%j.out" --export="$EXPORTS,STAGE=select_ft" "$RUNNER")
FINAL=$(sbatch --parsable $COMMON -t 04:00:00 -J aud_final \
  --dependency=afterok:$FIXED:$SELFT \
  -o "$LOGDIR/final_%j.out" --export="$EXPORTS,STAGE=final" "$RUNNER")

cat > "$NEW_RUN/job_manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "git_commit": "$(git rev-parse --short HEAD)",
  "partition": "$PARTITION",
  "log_dir": "$LOGDIR",
  "jobs": {
    "preflight":     {"id": "$PRE",   "depends_on": []},
    "fixed":         {"id": "$FIXED", "depends_on": ["$PRE"]},
    "select_online": {"id": "$SELON", "depends_on": ["$PRE"]},
    "finetune":      {"id": "$FT",    "array": "0-26%4", "depends_on": ["$SELON"]},
    "select_ft":     {"id": "$SELFT", "depends_on": ["$FT"]},
    "final":         {"id": "$FINAL", "depends_on": ["$FIXED", "$SELFT"]}
  },
  "final_report": "$NEW_RUN/final_summary.md"
}
EOF
echo "submitted: preflight=$PRE fixed=$FIXED select_online=$SELON finetune=$FT select_ft=$SELFT final=$FINAL"
echo "manifest:  $NEW_RUN/job_manifest.json"
echo "status:    bash src/instrument_v2/slurm/status_encoder_audit.sh"
