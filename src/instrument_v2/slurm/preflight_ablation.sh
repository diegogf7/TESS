#!/bin/bash
# All this code is from Claude
# Preflight for the instrument ablation: verifies data, frozen splits,
# existing JEPA epoch checkpoints, imports, and the unit-test suite.
set -euo pipefail
PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

fail() { echo "PREFLIGHT FAIL: $1" >&2; exit 1; }

DATA=${S14_DATA:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet}
[ -f "$DATA" ] || fail "data parquet missing: $DATA"
for f in artifacts/instrument_v2/chip_signal_diagnostic/split_train_tics.txt \
         artifacts/instrument_v2/chip_signal_diagnostic/split_test_tics.txt \
         artifacts/instrument_v2/sector14_jepa/split_val_tics.txt \
         artifacts/instrument_v2/sector14_jepa/grid_range.json; do
  [ -f "$f" ] || fail "split/grid artifact missing: $f"
done
CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints}
for s in 0 1 2; do
  ls "$CKPT_DIR"/s14jepa_shared_s${s}_ep*.pth >/dev/null 2>&1 \
    || fail "missing JEPA epoch checkpoints for seed $s in $CKPT_DIR"
done

$PY -c "import torch, sklearn, matplotlib" || fail "python deps"
$PY -m src.tests.test_ablation || fail "unit tests"
$PY -m src.tests.test_sector14_jepa || fail "sector14 tests"
echo "PREFLIGHT_OK"
