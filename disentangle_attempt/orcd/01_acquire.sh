#!/bin/bash
# Stage 1: index, select and download 80 chips.  RUN ON A LOGIN NODE -- this is the
# only stage that needs outbound network, and ORCD compute nodes generally have none.
#
#   tmux new -s acquire
#   bash disentangle_attempt/orcd/01_acquire.sh
#
# Everything is resumable: re-running skips finished sectors, chips and files.  A chip
# whose downloads had network failures is deliberately NOT marked complete, so a second
# run retries exactly those files and nothing else.
#
# Measured throughput on the static archive endpoint: ~249 files/s, so 240k files is
# roughly 16 minutes plus extraction.  Raise WORKERS if the cluster link is faster;
# the archive did not rate-limit at 128 concurrent requests.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

WORKERS="${WORKERS:-64}"

echo "=== sectors $MC_SECTORS | chips $MC_CHIPS | $MC_STARS_PER_CHIP stars/chip | $WORKERS workers ==="
date

python -m disentangle_attempt.multichip_acquire \
  --stage index \
  --data-dir "$DATA_DIR" \
  --sectors "$MC_SECTORS"

python -m disentangle_attempt.multichip_acquire \
  --stage select \
  --data-dir "$DATA_DIR" \
  --sectors "$MC_SECTORS" \
  --chips "$MC_CHIPS" \
  --stars-per-chip "$MC_STARS_PER_CHIP"

python -m disentangle_attempt.multichip_acquire \
  --stage download \
  --data-dir "$DATA_DIR" \
  --sectors "$MC_SECTORS" \
  --chips "$MC_CHIPS" \
  --workers "$WORKERS"

echo
echo "=== download stage finished ==="
date
echo "If any chip printed '[incomplete: re-run to retry]', run this script again."
echo "Next: sbatch disentangle_attempt/orcd/02_extract.sbatch"
