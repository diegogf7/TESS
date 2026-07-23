#!/bin/bash -l
# Chip-stratified S14 TOP-UP download to raise per-area counts for the group-32
# config (need >= 256 train stars/area). Same method as the dense set: sample
# an equal number PER cam-CCD chip so inner rings fill up, not just dense ones.
# Run on a LOGIN node (compute nodes have no internet):
#   nohup bash src/tglc/download_s14_dense_v2.sh > download_s14_v2.log 2>&1 &
set -u
OUT=/orcd/scratch/orcd/006/diegogon/tglc_primary
PER_CHIP=${PER_CHIP:-6000}          # x16 chips ~= 96k; ring-1 ~1/10 -> ~350/ring-1-area
PARALLEL=${PARALLEL:-8}
URL="https://archive.stsci.edu/hlsps/tglc/download_scripts/hlsp_tglc_tess_ffi_s0014_tess_v1_llc.sh"
BULK="$OUT/s0014_full.sh"

mkdir -p "$OUT"; cd "$OUT"

# 1) fetch the full S14 bulk list ONCE (all available targets)
if [ ! -s "$BULK" ]; then
  echo "fetching bulk list..."; curl -s "$URL" -o "$BULK"
fi
echo "bulk total curls: $(grep -c '^curl' "$BULK")"
echo "example line: $(grep -m1 '^curl' "$BULK")"     # <-- confirm the cam*-ccd* token here

# 2) chip-stratified selection: PER_CHIP random lines from each of the 16 cam-CCDs
: > s14_v2_download.sh
for cam in 1 2 3 4; do for ccd in 1 2 3 4; do
  n=$(grep "cam${cam}-ccd${ccd}" "$BULK" | grep -c '^curl')
  grep "cam${cam}-ccd${ccd}" "$BULK" | grep '^curl' | shuf | head -n "$PER_CHIP" \
    | sed 's/^curl -f/curl -sf/' >> s14_v2_download.sh
  echo "cam${cam}-ccd${ccd}: available=$n selected<=$PER_CHIP"
done; done
SEL=$(wc -l < s14_v2_download.sh)
echo "TOTAL selected: $SEL curls"
if [ "$SEL" -eq 0 ]; then
  echo "FATAL: 0 selected -- the cam-ccd token in the URL is not 'camN-ccdN'." >&2
  echo "       check 'example line' above and fix the grep pattern." >&2
  exit 1
fi

# 3) download in parallel (dedups against existing FITS by filename)
split -n "l/$PARALLEL" s14_v2_download.sh s14v2_chunk_
for c in s14v2_chunk_*; do bash "$c" & done; wait
rm -f s14v2_chunk_*
echo "=== DONE: $(find s0014 -name '*.fits' 2>/dev/null | wc -l) FITS now in s0014/ ==="
