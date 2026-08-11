#!/bin/bash
#SBATCH --job-name=tglc_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 2: parse the downloaded FITS into per-chip parquet, resolve detector positions
# with tess-point, and report per-chip detector coverage.  No network needed.
#
#   sbatch --partition=$CPU_PARTITION disentangle_attempt/orcd/02_extract.sbatch
#
# The coverage report at the end is the check that matters: every chip must span enough
# of the detector to host the outer [384,768] px peer band.  A chip that fails it would
# silently starve the spread rule later.
set -euo pipefail

# sbatch exports the submitting shell's environment, so REPO carries over when set.
REPO="${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}"
source "$REPO/disentangle_attempt/orcd/env.sh"

echo "=== extract + positions: sectors $SECTORS ==="
date

python -m disentangle_attempt.multichip_acquire \
  --stage extract \
  --data-dir "$DATA_DIR" \
  --sectors "$SECTORS" \
  --chips "$CHIPS" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"

python -m disentangle_attempt.multichip_acquire \
  --stage positions \
  --data-dir "$DATA_DIR" \
  --sectors "$SECTORS"

python -m disentangle_attempt.multichip_acquire \
  --stage coverage \
  --data-dir "$DATA_DIR" \
  --sectors "$SECTORS"

echo
echo "=== finished ==="
date
echo "Coverage table: $DATA_DIR/chip_coverage.csv"
echo "Next: sbatch disentangle_attempt/orcd/03_prepare.sbatch"
