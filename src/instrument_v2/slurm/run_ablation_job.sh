#!/bin/bash
# All this code is from Claude
# Single dispatcher for every stage of the instrument ablation.
# Env: STAGE (smoke|pretrain|select_pre|finetune|select_ft|final), RUN_ID,
#      SLURM_ARRAY_TASK_ID for array stages. Everything runs from the repo root.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

export PYTHONUNBUFFERED=1
export RUN_ID=${RUN_ID:?RUN_ID must be set}
export ABL_DIR=${ABL_DIR:-artifacts/instrument_v2/ablation/$RUN_ID}
export ABL_CKPT_DIR=${ABL_CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/ablation/$RUN_ID}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints}
export S14_DATA=${S14_DATA:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet}
export PRETRAIN_MANIFEST=$ABL_DIR/pretrain_selection.json

echo "=== STAGE=$STAGE RUN_ID=$RUN_ID host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-n/a} ==="
echo "git commit: $(git rev-parse --short HEAD)"

case "$STAGE" in
  smoke)
    # two-batch GPU smoke of both new objectives + a matched-finetune batch
    for OBJ in supcon hybrid; do
      OBJECTIVE=$OBJ CONTRASTIVE_WEIGHT=0.5 SEED=0 EPOCHS=1 MAX_BATCHES=2 \
        ABL_CKPT_DIR=$ABL_CKPT_DIR/smoke ABL_DIR=$ABL_DIR/smoke \
        $PY -m src.instrument_v2.train_sector14_contrastive
    done
    INIT_ARM=random TARGET=camera SEED=0 BACKBONE_LR=1e-4 EPOCHS=1 MAX_BATCHES=2 \
      ABL_CKPT_DIR=$ABL_CKPT_DIR/smoke ABL_DIR=$ABL_DIR/smoke \
      $PY -m src.instrument_v2.train_sector14_matched_finetune
    echo "SMOKE_OK"
    ;;
  pretrain)
    eval "$($PY -m src.instrument_v2.ablation_config pretrain "$SLURM_ARRAY_TASK_ID")"
    export OBJECTIVE SEED CONTRASTIVE_WEIGHT
    $PY -m src.instrument_v2.train_sector14_contrastive
    ;;
  select_pre)
    $PY -m src.instrument_v2.select_pretrain_checkpoints
    ;;
  finetune)
    eval "$($PY -m src.instrument_v2.ablation_config finetune "$SLURM_ARRAY_TASK_ID")"
    export INIT_ARM TARGET SEED BACKBONE_LR
    $PY -m src.instrument_v2.train_sector14_matched_finetune
    ;;
  select_ft)
    $PY -m src.instrument_v2.select_finetune_models
    ;;
  final)
    FINAL_EVAL=YES $PY -m src.instrument_v2.eval_final_ablation
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2; exit 1
    ;;
esac
echo "=== STAGE $STAGE DONE ==="
