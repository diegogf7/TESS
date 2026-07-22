#!/bin/bash -l
#SBATCH -J group_cbv_k8_mlp
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o group_cbv_k8_mlp_%j.out

# All code by Claude. Two-stage custom group-level TGLC CBV JEPA, K=8, MLP.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

echo "=== node: $(hostname) ==="
nvidia-smi || true

export SEED=${SEED:-0}
export K=${K:-8}
export RIDGE_LAMBDA=${RIDGE_LAMBDA:-1e-2}
export EPOCHS=${EPOCHS:-15}
export MIN_EPOCHS=${MIN_EPOCHS:-8}
export PATIENCE=${PATIENCE:-4}

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=== DRY_RUN: 1 batch, 1 epoch per stage ==="
  export MAX_BATCHES=1 EPOCHS=1 MIN_EPOCHS=1 PATIENCE=1 NUM_WORKERS=0
fi

"$PY" -m src.instrument_v2.train_group_cbv_k8_mlp
echo "=== DONE ==="
