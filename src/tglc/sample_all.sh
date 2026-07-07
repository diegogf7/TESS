#!/bin/bash
# Stream each primary-mission TGLC bulk script (s0001-s0026) and reservoir-
# sample PER_SECTOR download lines per sector. Streams ~30-60 GB total but
# stores only ~50 MB of sampled lines. Run on an Engaging LOGIN node
# (compute nodes have no internet), from the repo root:
#   nohup bash src/tglc/sample_all.sh > sample_all.log 2>&1 &
set -u
OUT=/orcd/scratch/orcd/006/diegogon/tglc/url_lists
BASE=https://archive.stsci.edu/hlsps/tglc/download_scripts
PER_SECTOR=7700          # x 26 sectors ~= 200k curves

mkdir -p "$OUT"
for s in $(seq 1 26); do
  SEC=$(printf "s%04d" "$s")
  if [ -s "$OUT/$SEC.txt" ]; then
    echo "=== $SEC already sampled, skipping ==="
    continue
  fi
  echo "=== $SEC ==="
  curl -s "$BASE/hlsp_tglc_tess_ffi_${SEC}_tess_v1_llc.sh" \
    | python3 -m src.tglc.sample_urls "$PER_SECTOR" "$s" > "$OUT/$SEC.txt"
  wc -l "$OUT/$SEC.txt"
done
echo "=== ALL SECTORS SAMPLED ==="
