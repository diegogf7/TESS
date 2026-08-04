#!/bin/bash -l
#SBATCH -J s4d_detector
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o s4d_detector_%j.out

# Dedicated launcher for the detector-nearest, 8-token, pairwise-window-cov experiment.
set -euo pipefail
PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

# ---- fixed experiment defaults (override on the command line only if you mean to) ----
export LOSS_MODE=${LOSS_MODE:-pairwise_window_cov}
export GROUPING_MODE=${GROUPING_MODE:-detector_nearest}
export GROUP_SIZE=${GROUP_SIZE:-16}
export N_STARS=${N_STARS:-1000}
export N_TOKENS=${N_TOKENS:-8}
export TOKEN_DIM=${TOKEN_DIM:-32}
export LAMBDA_SIZE=${LAMBDA_SIZE:-0.1}
export REQUIRE_FULL=${REQUIRE_FULL:-0}
export USE_AMP=${USE_AMP:-0}
export SEED=${SEED:-0}
export EPOCHS=${EPOCHS:-30}
export LR=${LR:-3e-4}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
export GROUPS_PER_AREA=${GROUPS_PER_AREA:-100}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
export COLLAPSE_STD=${COLLAPSE_STD:-0.05}
export MIN_OVERLAP=${MIN_OVERLAP:-64}
export MAX_BATCHES=${MAX_BATCHES:-0}
export S14_DATA=${S14_DATA:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2_xy.parquet}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export ART_DIR=${ART_DIR:-artifacts/shared_s4d/correction_v1}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/shared_s4d_correction_v1}

echo "================ resolved configuration ================"
echo "  node        : $(hostname)"
echo "  git commit  : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  LOSS_MODE   : $LOSS_MODE   GROUPING : $GROUPING_MODE   GROUP_SIZE : $GROUP_SIZE"
echo "  TOKENS      : ${N_TOKENS}x${TOKEN_DIM}   LAMBDA_SIZE : $LAMBDA_SIZE   N_STARS : $N_STARS"
echo "  EPOCHS      : $EPOCHS   REQUIRE_FULL : $REQUIRE_FULL   USE_AMP : $USE_AMP   MAX_BATCHES : $MAX_BATCHES"
echo "  LR/WD/GPA   : $LR / $WEIGHT_DECAY / $GROUPS_PER_AREA   EARLY_STOP : $EARLY_STOP_PATIENCE   COLLAPSE_STD : $COLLAPSE_STD"
echo "  S14_DATA    : $S14_DATA"
echo "  ART/CKPT    : $ART_DIR | $CKPT_DIR"
echo "========================================================"

# ---- FAIL FAST if physical detector coordinates are missing/invalid ----
"$PY" - "$S14_DATA" <<'PYCHK' || { echo "FATAL: DETECTOR_X/DETECTOR_Y missing/invalid -- run src/tglc/merge_detector_positions.py"; exit 1; }
import sys, numpy as np, pandas as pd
d = pd.read_parquet(sys.argv[1], columns=["DETECTOR_X", "DETECTOR_Y"])
frac = float(np.isfinite(d.to_numpy()).all(1).mean())
print(f"DETECTOR_X/Y finite fraction: {frac:.5f}", flush=True)
sys.exit(0 if frac >= 0.98 else 1)   # a handful of boundary NaNs are dropped in training
PYCHK

nvidia-smi || true
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device on $(hostname) -- aborting."; exit 1; }

"$PY" -m src.shared_s4d.train_correction
echo "=== DONE s4d_detector (loss ${LOSS_MODE} grouping ${GROUPING_MODE}) ==="
