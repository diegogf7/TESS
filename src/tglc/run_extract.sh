#!/bin/bash -l
#SBATCH -J tglc_extract
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o tglc_extract_%j.out

# FITS -> parquet extraction for the 208k TGLC primary-mission curves.
# CPU-only job; no GPU needed.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

# the cluster env has never needed astropy before now
python -c "import astropy" 2>/dev/null || pip install astropy

python -m src.tglc.extract_parquet

echo "=== DONE ==="
