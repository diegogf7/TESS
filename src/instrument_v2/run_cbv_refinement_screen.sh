#!/bin/bash -l
# All this code is from Claude
# CBV-refinement pilot: refine the Transformer-JEPA encoder with train-only
# per-chip CBV teacher targets, then run the frozen five-way encoder probe.
# One sequential watchable job. Validation-only, no fine-tuning, no test.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
export PYTHONUNBUFFERED=1
export SEED=0 K=8 K_CBV=64 EPOCHS=10 LR=3e-4 MIN_EPOCHS=4 PATIENCE=3
export EPOCH0_REF=0.4559 EPOCH0_TOL=0.002
export CBV_ART_DIR=artifacts/instrument_v2/cbv_refinement_screen
export CBV_CKPT_DIR=/orcd/scratch/orcd/006/diegogon/checkpoints/cbv_refinement_screen
export TX_CHECKPOINT=/orcd/scratch/orcd/006/diegogon/checkpoints/transformer_encoder_screen/frtstudent_tx_k8_s0_best.pth

echo "=== node $(hostname) | CBV refinement pilot | commit $(git rev-parse --short HEAD) ==="

echo "--- train (CBV-refined teacher targets) ---"
$PY -m src.instrument_v2.train_cbv_refinement_jepa

echo "--- frozen encoder probe ---"
$PY -m src.instrument_v2.eval_cbv_refinement_jepa

echo "=== DONE ==="
