#!/bin/bash
# All this code is from Claude
# Dispatcher for the online-vs-EMA encoder audit stages.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

export PYTHONUNBUFFERED=1
export RUN_ID=${RUN_ID:-abl1_encoder_audit}
export OLD_RUN=${OLD_RUN:-artifacts/instrument_v2/ablation/abl1}
export NEW_RUN=${NEW_RUN:-artifacts/instrument_v2/ablation/$RUN_ID}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints}
export ABL_CKPT_DIR=${ABL_CKPT_DIR:-$CKPT_DIR/ablation/abl1}      # abl1 pretrain ckpts (read-only)
export AUDIT_CKPT_DIR=${AUDIT_CKPT_DIR:-$CKPT_DIR/ablation/$RUN_ID}
export S14_DATA=${S14_DATA:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet}
export ONLINE_MANIFEST=$NEW_RUN/online_pretrain_selection.json

echo "=== STAGE=$STAGE RUN_ID=$RUN_ID host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-n/a} ==="
echo "git commit: $(git rev-parse --short HEAD)"

case "$STAGE" in
  preflight)
    $PY -m src.tests.test_encoder_audit
    $PY -m src.tests.test_ablation
    for f in "$OLD_RUN/pretrain_selection.json" "$OLD_RUN/finetune_selection.json"; do
      [ -f "$f" ] || { echo "PREFLIGHT FAIL: missing $f" >&2; exit 1; }
    done
    echo "PREFLIGHT_OK"
    ;;
  fixed)
    $PY -m src.instrument_v2.eval_encoder_source_fixed
    ;;
  select_online)
    $PY -m src.instrument_v2.select_online_checkpoints
    ;;
  finetune)
    eval "$($PY -m src.instrument_v2.ablation_config online_finetune "$SLURM_ARRAY_TASK_ID")"
    export INIT_ARM SEED BACKBONE_LR
    $PY -m src.instrument_v2.train_online_finetune
    ;;
  select_ft)
    $PY -m src.instrument_v2.select_online_finetune
    ;;
  final)
    FINAL_EVAL=YES $PY -m src.instrument_v2.eval_encoder_source_final
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2; exit 1
    ;;
esac
echo "=== STAGE $STAGE DONE ==="
