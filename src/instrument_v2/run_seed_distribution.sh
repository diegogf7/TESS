#!/bin/bash -l
# All this code is from Claude
# Seed-distribution screen, one sequential watchable job:
#   1. train fixed-teacher MLP students for seeds 1 and 2 (seed 0 reused)
#   2. probe 8 random S4Ds + all 3 students through one harness
# Warm-start falls back to the seed-0 group-JEPA selection when a seed's own
# selection does not exist (printed loudly).
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
export PYTHONUNBUFFERED=1

export FRT_ART_DIR=artifacts/instrument_v2/seed_distribution_screen
export FRT_CKPT_DIR=/orcd/scratch/orcd/006/diegogon/checkpoints/seed_distribution_screen
export TEACHER_SELECTION=artifacts/instrument_v2/fixed_regional_teacher_v1/selection_regteacher_k8_s0.json
export PREDICTOR=mlp K=8 EPOCHS=30

echo "=== node $(hostname) | seed distribution screen | commit $(git rev-parse --short HEAD) ==="

for SEED in 1 2; do
  GROUP_SEL=artifacts/instrument_v2/group_level/selection_s14groupmean_k8_s${SEED}.json
  if [ ! -f "$GROUP_SEL" ]; then
    echo "NOTE: $GROUP_SEL missing -- warm-starting seed $SEED from the s0 group-JEPA"
    GROUP_SEL=artifacts/instrument_v2/group_level/selection_s14groupmean_k8_s0.json
  fi
  echo "--- train MLP student seed $SEED ---"
  SEED=$SEED GROUP_SELECTION=$GROUP_SEL \
    $PY -m src.instrument_v2.train_fixed_teacher_instrument_jepa
done

echo "--- probe distribution: 8 random S4Ds vs 3 students ---"
SDS_ART_DIR=$FRT_ART_DIR $PY -m src.instrument_v2.seed_distribution_screen

echo "=== DONE ==="
