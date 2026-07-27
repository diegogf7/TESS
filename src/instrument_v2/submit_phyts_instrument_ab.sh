#!/bin/bash -l
#SBATCH -J phyts_ab
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o phyts_ab_%j.out
# Matched A/B: physics-JEPA classification of PhyTS s14, raw vs instrument-cleaned.
# No training. Prints progress + the final raw / cleaned / difference only.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

echo "=== node: $(hostname) ==="
nvidia-smi || true

"$PY" -m src.instrument_v2.eval_phyts_instrument_ab
echo "=== DONE ==="
