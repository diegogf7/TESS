#!/bin/bash
#SBATCH --job-name=tglc_twodecoder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 4: train the p64/i4 two-decoder model over all 80 chips.
#
#   sbatch --partition=$GPU_PARTITION disentangle_attempt/orcd/04_train.sbatch
#
# The model, masking, loss and output equations are the completed single-chip run's,
# imported rather than reimplemented.  The job refuses to start if stage 3 did not
# record a PASS, and refuses to overwrite a previous training attempt's artifacts.
set -euo pipefail

REPO="${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}"
source "$REPO/disentangle_attempt/orcd/env.sh"

echo "=== training: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

python -m disentangle_attempt.train_multichip_spread \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --cache-dir "$OUT_DIR/dataset_cache"

echo
echo "=== finished ==="
date
echo "Metrics:  $OUT_DIR/metrics.json"
echo "Per-chip: $OUT_DIR/per_chip_test_metrics.csv"
echo "Plots:    $OUT_DIR/plots"
