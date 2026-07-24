#!/bin/bash -l
#SBATCH -J clean_ss
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o clean_ss_%j.out
# Post-processing scale-and-subtract cleaner on the frozen group-32 DIRECT
# decoder. No training. Defaults to the v2 dense data + split (overridable).
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

DENSE=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export SEED=${SEED:-0}

echo "=== node: $(hostname) ==="
nvidia-smi || true
echo "data: S14_DATA=$S14_DATA SPLIT_DIR=$SPLIT_DIR"

"$PY" -m src.instrument_v2.scale_subtract_clean
echo "=== DONE ==="
