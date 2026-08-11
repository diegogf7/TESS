#!/bin/bash
#SBATCH --job-name=tglc_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 3: build the multi-chip patch and cache it.  CPU only.
#
#   sbatch --partition=$CPU_PARTITION disentangle_attempt/orcd/03_prepare.sbatch
#
# This applies the unchanged peer rule -- bands [128,384) and [384,768] px, four peers
# each, minimum peer-to-peer separation 256 px relaxing to 192 then 128 only if a chip
# cannot do better -- to every candidate anchor on every chip, in parallel.
#
# It writes geometry_summary.json with status PASS or FAIL.  The training job refuses
# to start unless that says PASS, so an infeasible geometry never reaches a GPU.
set -euo pipefail

REPO="${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}"
source "$REPO/disentangle_attempt/orcd/env.sh"

echo "=== building multi-chip spread patch ==="
date

python -m disentangle_attempt.prepare_multichip_dataset \
  --config "$CONFIG" \
  --parquet "$DATA_DIR/cross_sector_raw.parquet" \
  --output-dir "$OUT_DIR" \
  --workers "${SLURM_CPUS_PER_TASK:-32}"

echo
echo "=== finished ==="
date
echo "Geometry summary: $OUT_DIR/geometry_summary.json"
echo "Per-chip audit:   $OUT_DIR/geometry/per_chip_summary.csv"
echo "Next: sbatch disentangle_attempt/orcd/04_train.sbatch"
