#!/bin/bash
# Download random TGLC light curves from every primary-mission sector (1-26).
#
# For each sector, MAST publishes a huge shell script of curl commands (one per
# star). We stream that script and reservoir-sample N lines, so the draw is
# uniform across the WHOLE sector (all cameras/CCDs), then run the sampled
# curls in parallel.
#
# The strictest quality filter (TESS_flags==0 & TGLC_flags==0) is applied later
# in the parquet extractor, not here -- so we oversample (8000/sector = 208k)
# to net ~200k usable curves after low-coverage stars are dropped.
#
# Resume-safe: finished sectors leave a .done marker and are skipped on rerun.

set -u
N_PER_SECTOR=8000
PARALLEL=6
OUT=/orcd/scratch/orcd/006/diegogon/tglc_primary

mkdir -p "$OUT"
cd "$OUT"

for i in $(seq 1 26); do
  SEC=$(printf "s%04d" "$i")
  if [ -f "${SEC}.done" ]; then
    echo "=== $SEC already done, skipping ==="
    continue
  fi

  echo "=== $SEC : sampling $N_PER_SECTOR stars ==="
  URL="https://archive.stsci.edu/hlsps/tglc/download_scripts/hlsp_tglc_tess_ffi_${SEC}_tess_v1_llc.sh"
  curl -s "$URL" | awk -v n="$N_PER_SECTOR" '
      BEGIN { srand() }
      /^curl/ {
        c++
        if (c <= n) r[c] = $0
        else { j = int(rand()*c) + 1; if (j <= n) r[j] = $0 }
      }
      END { for (k = 1; k <= n && k <= c; k++) print r[k] }
    ' | sed 's/^curl -f/curl -sf/' > "${SEC}_sample.sh"

  echo "    $(wc -l < "${SEC}_sample.sh") commands; downloading with $PARALLEL workers"
  split -n "l/$PARALLEL" "${SEC}_sample.sh" "${SEC}_chunk_"
  for chunk in "${SEC}_chunk_"*; do
    bash "$chunk" &
  done
  wait
  rm -f "${SEC}_chunk_"*

  echo "    downloaded: $(find "$SEC" -name '*.fits' | wc -l) files"
  touch "${SEC}.done"
done

echo "ALL DONE: $(find . -name '*.fits' | wc -l) total FITS files in $OUT"
