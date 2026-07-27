#!/bin/bash -l
#SBATCH -J phyts_apec
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o phyts_apec_%j.out
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

echo "=== node: $(hostname) ==="
nvidia-smi || true

"$PY" -m src.instrument_v2.inspect_phyts_aperiodic_eclipse
echo "=== DONE ==="

