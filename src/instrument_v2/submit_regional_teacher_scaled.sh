#!/bin/bash -l
#SBATCH -J regteach_g32n1000
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -o regteach_g32n1000_%j.out

# PART A (scaled): regional group teacher, 32-star groups, exactly 1,000 stars
# per area, 1,000 A/B pairs per area per epoch (resampled each epoch), 20 epochs,
# NO early stopping. HARD-FAILS with an area-count report if any area has < 1,000
# training stars. Produces the regteacher_g32_n1000_s0 selection for Part B.
#   sbatch -p mit_normal_gpu src/instrument_v2/submit_regional_teacher_scaled.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

export SEED=${SEED:-0}
export GROUP_SIZE=${GROUP_SIZE:-32}
export CBV_RANK=${CBV_RANK:-8}
export MIN_VALID_STARS=${MIN_VALID_STARS:-16}
export N_STARS=${N_STARS:-1000}
export N_PAIRS=${N_PAIRS:-1000}
export REQUIRE_FULL=${REQUIRE_FULL:-1}       # 0 = use all available stars/area (no 1000 floor)
export EPOCHS=${EPOCHS:-20}
DENSE_V2=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE_V2}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export GROUP_ART_DIR=${GROUP_ART_DIR:-artifacts/instrument_v2/regteacher_g32_n1000_v1}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/regteacher_g32_n1000_v1}
export MAX_BATCHES=${MAX_BATCHES:-0}

echo "================ resolved configuration ================"
echo "  node        : $(hostname)"
echo "  git commit  : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  N_STARS/N_PAIRS/EPOCHS : $N_STARS/$N_PAIRS/$EPOCHS (no early stop)"
echo "  GROUP/RANK/MINVALID    : $GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS"
echo "  ART/CKPT    : $GROUP_ART_DIR | $CKPT_DIR"
echo "========================================================"
nvidia-smi || true
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device on $(hostname) -- aborting (add --exclude=$(hostname -s))."; exit 1; }

"$PY" -m src.instrument_v2.train_regional_teacher_scaled
echo "=== DONE Part A (regteacher_g${GROUP_SIZE}_n${N_STARS}_s${SEED}) ==="
