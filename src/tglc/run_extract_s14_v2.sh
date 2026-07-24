#!/bin/bash -l
#SBATCH -J extract_s14v2
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH -o extract_s14v2_%j.out
# FITS -> parquet for the group-32 top-up set, then assign area (ring) from
# ra/dec. Produces tglc_raw_cadence_s14_dense_v2.parquet (does not touch the
# old dense parquet). s0014 is the only sector present, so extract-all == S14.
set -euo pipefail
cd /orcd/scratch/orcd/006/diegogon/TESS
PY=/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python
ROOT=/orcd/scratch/orcd/006/diegogon/tglc_primary
RAW=$ROOT/tglc_raw_cadence_s14_dense_v2_raw.parquet
FINAL=$ROOT/tglc_raw_cadence_s14_dense_v2.parquet

echo "=== extract FITS -> raw parquet ==="
FITS_ROOT=$ROOT OUT_PATH=$RAW N_WORKERS=16 "$PY" -m src.tglc.extract_raw_parquet_cadence

echo "=== add area (ring from ra/dec) -> final ==="
"$PY" - <<PYEOF
import pandas as pd
from src.regions.areas import add_area
d = pd.read_parquet("$RAW")
d = d[d["sector"] == 14].reset_index(drop=True)
d = add_area(d)
d.to_parquet("$FINAL")
print("wrote $FINAL | rows", len(d), "| unique areas", d["area"].nunique(),
      "| unique TIC", d["TIC"].nunique(), flush=True)
PYEOF
echo "=== DONE ==="
