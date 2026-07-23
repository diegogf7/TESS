#!/bin/bash -l
#SBATCH -J group32_cbv8
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o group32_cbv8_%j.out

# All code by Claude. Two-stage group-CBV MLP JEPA with the DECOUPLED config:
# group size, CBV rank and cadence-validity are three independent knobs.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

# ---- decoupled configuration (this experiment REQUIRES exactly 32 / 8 / 16) ----
export GROUP_SIZE=${GROUP_SIZE:-32}
export CBV_RANK=${CBV_RANK:-8}
export MIN_VALID_STARS=${MIN_VALID_STARS:-16}

# ---- dense data defaults (group size 32 needs the dense parquet; overridable) ---
DENSE=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense.parquet
export S14_DATA=${S14_DATA:-$DENSE}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense}

echo "=== node: $(hostname) ==="
nvidia-smi || true
echo "config: GROUP_SIZE=$GROUP_SIZE CBV_RANK=$CBV_RANK MIN_VALID_STARS=$MIN_VALID_STARS"
echo "data:   S14_DATA=$S14_DATA SPLIT_DIR=$SPLIT_DIR"

if [[ "$GROUP_SIZE" != "32" || "$CBV_RANK" != "8" || "$MIN_VALID_STARS" != "16" ]]; then
    echo "FATAL: config must be exactly GROUP_SIZE=32 CBV_RANK=8 MIN_VALID_STARS=16 " \
         "(got $GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS)" >&2
    exit 1
fi

"$PY" -m src.instrument_v2.train_group_cbv_k8_mlp
echo "=== DONE ==="
