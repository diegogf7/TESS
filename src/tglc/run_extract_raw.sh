#!/bin/bash -l
#SBATCH -J extract_raw
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o extract_raw_%j.out

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve

python -m src.tglc.extract_raw_parquet
echo "=== DONE ==="
