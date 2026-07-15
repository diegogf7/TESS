#!/bin/bash -l
#SBATCH -J gap_audit
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o gap_audit_%j.out

# All code by Claude.
# Gap-recoverability audit (task 1) -- CPU only, no GPU needed.

PY=/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python
cd /orcd/scratch/orcd/006/diegogon/TESS

echo "=== node: $(hostname) ==="
export GAP_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_probe20k_area.parquet

$PY -m src.regions.eval_gap_recoverability

echo "=== DONE ==="
