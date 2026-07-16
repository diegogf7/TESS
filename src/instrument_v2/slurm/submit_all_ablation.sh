#!/bin/bash
# All this code is from Claude
# Submit the FULL instrument-ablation dependency graph in one shot.
#   DRY_RUN=1 bash src/instrument_v2/slurm/submit_all_ablation.sh   # validate only
#   bash src/instrument_v2/slurm/submit_all_ablation.sh             # submit
# Stages: smoke -> pretrain[12] -> select_pre -> finetune[72] -> select_ft -> final
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
RUN_ID=${RUN_ID:-abl_$(date +%Y%m%d_%H%M%S)}
LOGDIR=logs/instrument_v2_ablation/$RUN_ID
ABL_DIR=artifacts/instrument_v2/ablation/$RUN_ID
RUNNER=src/instrument_v2/slurm/run_ablation_job.sh
PARTITION=${PARTITION:-ou_mki_gpu}
COMMON="-p $PARTITION --gres=gpu:1 -N 1 -c 8 --mem=64G"

# ---- validate the task grid before touching Slurm ----
eval "$($PY -m src.instrument_v2.ablation_config counts)"
[ "$PRETRAIN_TASKS" -eq 12 ] || { echo "expected 12 pretrain tasks, got $PRETRAIN_TASKS"; exit 1; }
[ "$FINETUNE_TASKS" -eq 72 ] || { echo "expected 72 finetune tasks, got $FINETUNE_TASKS"; exit 1; }
echo "task grid OK: $PRETRAIN_TASKS pretrain, $FINETUNE_TASKS finetune"
echo "RUN_ID=$RUN_ID  commit=$(git rev-parse --short HEAD)  partition=$PARTITION"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN -- would submit:"
  echo "  1 smoke        : sbatch $COMMON -t 00:30:00 (STAGE=smoke)"
  echo "  2 pretrain     : sbatch $COMMON -t 04:00:00 --array=0-11%4 (STAGE=pretrain), afterok:smoke"
  echo "  3 select_pre   : sbatch $COMMON -t 08:00:00 (STAGE=select_pre), afterok:pretrain"
  echo "  4 finetune     : sbatch $COMMON -t 04:00:00 --array=0-71%4 (STAGE=finetune), afterok:select_pre"
  echo "  5 select_ft    : sbatch $COMMON -t 00:30:00 (STAGE=select_ft), afterok:finetune"
  echo "  6 final        : sbatch $COMMON -t 04:00:00 (STAGE=final, FINAL_EVAL=YES), afterok:select_pre+select_ft"
  for i in 0 11; do echo "  pretrain[$i]: $($PY -m src.instrument_v2.ablation_config pretrain $i | tr '\n' ' ')"; done
  for i in 0 35 71; do echo "  finetune[$i]: $($PY -m src.instrument_v2.ablation_config finetune $i | tr '\n' ' ')"; done
  exit 0
fi

mkdir -p "$LOGDIR" "$ABL_DIR"
EXPORTS="ALL,RUN_ID=$RUN_ID,PY=$PY,REPO=$REPO"

SMOKE=$(sbatch --parsable $COMMON -t 00:30:00 -J abl_smoke \
  -o "$LOGDIR/smoke_%j.out" --export="$EXPORTS,STAGE=smoke" "$RUNNER")
PRE=$(sbatch --parsable $COMMON -t 04:00:00 -J abl_pretrain --array=0-11%4 \
  --dependency=afterok:$SMOKE \
  -o "$LOGDIR/pretrain_%A_%a.out" --export="$EXPORTS,STAGE=pretrain" "$RUNNER")
SELPRE=$(sbatch --parsable $COMMON -t 08:00:00 -J abl_selpre \
  --dependency=afterok:$PRE \
  -o "$LOGDIR/select_pre_%j.out" --export="$EXPORTS,STAGE=select_pre" "$RUNNER")
FT=$(sbatch --parsable $COMMON -t 04:00:00 -J abl_finetune --array=0-71%4 \
  --dependency=afterok:$SELPRE \
  -o "$LOGDIR/finetune_%A_%a.out" --export="$EXPORTS,STAGE=finetune" "$RUNNER")
SELFT=$(sbatch --parsable $COMMON -t 00:30:00 -J abl_selft \
  --dependency=afterok:$FT \
  -o "$LOGDIR/select_ft_%j.out" --export="$EXPORTS,STAGE=select_ft" "$RUNNER")
FINAL=$(sbatch --parsable $COMMON -t 04:00:00 -J abl_final \
  --dependency=afterok:$SELPRE:$SELFT \
  -o "$LOGDIR/final_%j.out" --export="$EXPORTS,STAGE=final" "$RUNNER")

cat > "$ABL_DIR/job_manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "git_commit": "$(git rev-parse --short HEAD)",
  "partition": "$PARTITION",
  "log_dir": "$LOGDIR",
  "jobs": {
    "smoke":      {"id": "$SMOKE",  "depends_on": []},
    "pretrain":   {"id": "$PRE",    "array": "0-11%4", "depends_on": ["$SMOKE"]},
    "select_pre": {"id": "$SELPRE", "depends_on": ["$PRE"]},
    "finetune":   {"id": "$FT",     "array": "0-71%4", "depends_on": ["$SELPRE"]},
    "select_ft":  {"id": "$SELFT",  "depends_on": ["$FT"]},
    "final":      {"id": "$FINAL",  "depends_on": ["$SELPRE", "$SELFT"]}
  },
  "final_report": "$ABL_DIR/final_summary.md"
}
EOF
echo "submitted: smoke=$SMOKE pretrain=$PRE select_pre=$SELPRE finetune=$FT select_ft=$SELFT final=$FINAL"
echo "manifest:  $ABL_DIR/job_manifest.json"
echo "status:    bash src/instrument_v2/slurm/status_ablation.sh $RUN_ID"
