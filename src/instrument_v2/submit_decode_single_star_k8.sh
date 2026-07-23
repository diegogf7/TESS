#!/bin/bash -l
#SBATCH -J decode_k8
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o decode_k8_%j.out

set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}

REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

echo "=== node: $(hostname) ==="
nvidia-smi || true

export SEED=${SEED:-0}
export K=${K:-8}
export RIDGE_LAMBDA=${RIDGE_LAMBDA:-1e-2}
export EPOCHS=${EPOCHS:-20}

"$PY" -m src.instrument_v2.decode_single_star_k8
echo "=== DONE ==="

