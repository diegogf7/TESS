#!/bin/bash -l
# Targeted TGLC download of the PhyTS sector-14 stars, selected by their Gaia DR3
# id (TGLC names each file gaiaid-<DR3>). Grep the S14 bulk list for the PhyTS
# GaiaIDs -> download only those FITS. Run on a LOGIN node (compute nodes have no
# internet), from the repo root:
#   nohup bash src/tglc/download_phyts_s14.sh > download_phyts_s14.log 2>&1 &
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
PHYTS=${PHYTS:-/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train_30min.parquet}
BULK=/orcd/scratch/orcd/006/diegogon/tglc_primary/s0014_full.sh
OUT=/orcd/scratch/orcd/006/diegogon/tglc_phyts
PARALLEL=${PARALLEL:-8}

[ -s "$BULK" ] || { echo "FATAL: $BULK missing (fetch the S14 bulk list first)" >&2; exit 1; }
mkdir -p "$OUT"; cd "$OUT"

# 1) PhyTS s14 Gaia DR3 ids -> exact grep patterns "gaiaid-<id>-"
"$PY" - "$PHYTS" <<'PYEOF'
import sys, pandas as pd
p = pd.read_parquet(sys.argv[1]); p = p[p["sector"] == 14]
ids = sorted({int(g) for g in p["GaiaID"].dropna()})
with open("phyts_gaia_patterns.txt", "w") as fh:
    fh.write("".join(f"gaiaid-{i}-\n" for i in ids))
print(f"PhyTS s14 unique GaiaIDs: {len(ids)}", flush=True)
PYEOF

# 2) select their curl lines from the 2.3 GB S14 bulk list (fixed-string grep)
grep -F -f phyts_gaia_patterns.txt "$BULK" | sed 's/^curl -f/curl -sf/' > phyts_download.sh
echo "matched curl lines: $(grep -c '^curl' phyts_download.sh)"

# 3) download in parallel
split -n "l/$PARALLEL" phyts_download.sh phyts_chunk_
for c in phyts_chunk_*; do bash "$c" & done; wait
rm -f phyts_chunk_*
echo "=== DONE: $(find s0014 -name '*.fits' 2>/dev/null | wc -l) FITS in $OUT/s0014 ==="
