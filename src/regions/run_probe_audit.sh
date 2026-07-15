#!/bin/bash -l
#SBATCH -J probe_audit
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o probe_audit_%j.out

# All code by Claude.
# Audit reprobe (task 2): 6 checkpoints + random-init + rawflux baseline,
# PCA sweep, within-sector + leave-sector-out, bootstrap CIs, paired diffs.

PY=/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python
cd /orcd/scratch/orcd/006/diegogon/TESS

echo "=== node: $(hostname) ==="
nvidia-smi

export JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_probe_s14_100.parquet

$PY -m src.regions.eval_chip_probes_audit

echo "=== DONE ==="
