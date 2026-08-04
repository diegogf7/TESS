#!/bin/bash -l
#SBATCH -J s4d_correction
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o s4d_correction_%j.out


set -euo pipefail
PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

export SEED=${SEED:-0}
export GROUP_SIZE=${GROUP_SIZE:-32}
export N_STARS=${N_STARS:-1000}
export EPOCHS=${EPOCHS:-30}
export LAMBDA_SIZE=${LAMBDA_SIZE:-0.5}       # new loss scale differs -- sweep 0.3/0.5/0.7/1.0 (do NOT assume 0.7)
export MIN_OVERLAP=${MIN_OVERLAP:-64}
export LOSS_MODE=${LOSS_MODE:-topk_fixed_cov}   # topk_fixed_cov | legacy_corr | windowed_group_cov
export TOPK_PEERS=${TOPK_PEERS:-8}
export GROUPING_MODE=${GROUPING_MODE:-random}   # random | nearest (RA/Dec anchor groups)
export REQUIRE_FULL=${REQUIRE_FULL:-1}       # 0 = all available stars/area (dense_v2 has areas < 1000)
DENSE_V2=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE_V2}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export ART_DIR=${ART_DIR:-artifacts/shared_s4d/correction_v1}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/shared_s4d_correction_v1}
export MAX_BATCHES=${MAX_BATCHES:-0}

echo "================ resolved configuration ================"
echo "  node        : $(hostname)"
echo "  git commit  : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  LOSS_MODE/TOPK/LAMBDA/MINOV : $LOSS_MODE/$TOPK_PEERS/$LAMBDA_SIZE/$MIN_OVERLAP"
echo "  N_STARS/EPOCHS/require_full : $N_STARS/$EPOCHS/$REQUIRE_FULL"
echo "  ART/CKPT    : $ART_DIR | $CKPT_DIR"
echo "========================================================"
nvidia-smi || true
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device on $(hostname) -- aborting (add --exclude=$(hostname -s))."; exit 1; }

"$PY" -m src.shared_s4d.train_correction
echo "=== DONE s4d_correction (lambda ${LAMBDA_SIZE}) ==="
