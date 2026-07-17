#!/usr/bin/env bash
# All this code is from Claude
# Static + unit + synthetic-smoke checker for the Instance-to-Subspace JEPA.
# Never reads the real Sector-14 parquet or the real test split: every smoke
# runs on a synthetic parquet in a temp directory. Exits nonzero on failure.
set -euo pipefail

PYBIN="${PYBIN:-python}"
NEW_PY="src/instrument_v2/instance_subspace_jepa.py \
src/instrument_v2/train_instance_subspace_jepa.py \
src/instrument_v2/train_instance_subspace_finetune.py"
NEW_SH="src/instrument_v2/run_instance_subspace_screen.sh \
src/instrument_v2/check_instance_subspace.sh"

echo "== 1/5 py_compile =="
$PYBIN -m py_compile $NEW_PY
echo "OK"

echo "== 2/5 bash -n =="
for s in $NEW_SH; do bash -n "$s"; done
echo "OK"

echo "== 3/5 new unit tests =="
$PYBIN -m src.tests.test_instance_subspace_jepa
$PYBIN -m src.tests.test_instance_subspace_protocol

echo "== 4/5 existing instrument_v2 tests =="
$PYBIN -m src.tests.test_group_level_jepa
$PYBIN -m src.tests.test_sector14_jepa
$PYBIN -m src.tests.test_ablation

echo "== 5/5 synthetic smoke (1 epoch / 2 batches, every arm) =="
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
$PYBIN - <<PY
import numpy as np, os
from src.tests.test_sector14_jepa import synthetic_frame
tmp = "$TMP"
df = synthetic_frame(n_per_chip=48)
df.to_parquet(os.path.join(tmp, "synth.parquet"))
tics = sorted(set(df["TIC"].astype(str)))
os.makedirs(os.path.join(tmp, "split"), exist_ok=True)
with open(os.path.join(tmp, "split", "split_train_tics.txt"), "w") as fh:
    fh.write("\n".join(tics[:614]))
with open(os.path.join(tmp, "split", "split_test_tics.txt"), "w") as fh:
    fh.write("\n".join(tics[614:]))
print(f"synthetic env ready: {len(tics)} stars")
PY

SMOKE_ENV="S14_DATA=$TMP/synth.parquet SPLIT_DIR=$TMP/split BASE_ART_DIR=$TMP/base \
ISJ_ART_DIR=$TMP/art ISJ_CKPT_DIR=$TMP/ckpt NUM_WORKERS=0"
for ARM in mean_to_mean instance_mean instance_mean_var instance_cov; do
  echo "-- smoke pretrain arm=$ARM --"
  env $SMOKE_ENV ISJ_ARM="$ARM" GROUP_SIZE=2 SEED=0 EPOCHS=1 MAX_BATCHES=2 PROBE_EVERY=1 \
    $PYBIN -m src.instrument_v2.train_instance_subspace_jepa | grep -E "epoch 001|RANDOM"
done
echo "-- smoke LP-FT (scratch_direct + scratch_proj) --"
for INIT in scratch_direct scratch_proj; do
  env $SMOKE_ENV ISJ_INIT="$INIT" ISJ_ARM=instance_mean GROUP_SIZE=2 SEED=0 \
    BACKBONE_LR=3e-5 HEAD_EPOCHS=1 FT_EPOCHS=1 MAX_BATCHES=2 BATCH=32 \
    $PYBIN -m src.instrument_v2.train_instance_subspace_finetune | grep -E "LP 001|FT 001"
done

echo
echo "ALL CHECKS PASSED"
echo "confirmation: only synthetic data in $TMP was read; the real test split was never loaded"
echo "submit the validation-only screen with:"
echo '  sbatch -J isj_screen -p pg_mki_aryeh --gres=gpu:1 -N 1 -c 8 --mem=64G -t 12:00:00 \'
echo '    -o isj_screen_%j.out \'
echo '    --wrap="PYTHONUNBUFFERED=1 PATH=/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin:\$PATH bash src/instrument_v2/run_instance_subspace_screen.sh"'
