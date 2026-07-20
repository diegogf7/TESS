#!/bin/bash -l
# All this code is from Claude
# Transformer-predictor screen: train the Transformer-predictor student
# (seed 0, <=20 epochs, early stop) then run the frozen five-way probe.
# One job, sequential, everything printed live. Validation-only.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
export PYTHONUNBUFFERED=1

# Encoder-benchmark revision: selection, early stopping, and PASS all use
# the S4D ENCODER output (SELECT_VIEW=online); the transformer predictor is
# training machinery only and is discarded at evaluation. New dirs so the
# first (predicted-selected) screen's results are preserved.
export FRT_ART_DIR=artifacts/instrument_v2/transformer_encoder_screen
export FRT_CKPT_DIR=/orcd/scratch/orcd/006/diegogon/checkpoints/transformer_encoder_screen
export TXS_ART_DIR=$FRT_ART_DIR
export TEACHER_SELECTION=artifacts/instrument_v2/fixed_regional_teacher_v1/selection_regteacher_k8_s0.json
export PREDICTOR=transformer SELECT_VIEW=online
export SEED=0 K=8 EPOCHS=20 MIN_EPOCHS=8 PATIENCE=6

echo "=== node $(hostname) | transformer predictor screen | commit $(git rev-parse --short HEAD) ==="

echo "--- train transformer-predictor student ---"
$PY -m src.instrument_v2.train_fixed_teacher_instrument_jepa

echo "--- frozen five-way probe screen ---"
$PY -m src.instrument_v2.eval_transformer_predictor_screen

echo "=== DONE ==="
