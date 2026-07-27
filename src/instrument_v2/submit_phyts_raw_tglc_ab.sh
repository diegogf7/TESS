#!/bin/bash -l
#SBATCH -J phyts_raw_ab
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o phyts_raw_ab_%j.out
# Matched A/B on RAW TGLC flux (PhyTS labels only): physics classification
# baseline vs instrument-cleaned. No training.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

echo "=== node: $(hostname) ==="
nvidia-smi || true

"$PY" -m src.instrument_v2.eval_phyts_raw_tglc_ab
echo "=== DONE ==="
