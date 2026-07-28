#!/bin/bash -l
#SBATCH -J cbv_oracle
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o cbv_oracle_%j.out

# Validation-only ORACLE CEILING test for the K=8 CBV-weight decoder. Fits the
# best-possible 8 weights to the pre-CBV 32-star median and localizes the loss
# (basis capacity vs decoder vs weak-signal). Nothing is trained or rebuilt.
#   sbatch -p mit_normal_gpu src/instrument_v2/submit_eval_cbv_oracle_ceiling.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

export SEED=${SEED:-0}
export GROUP_SIZE=${GROUP_SIZE:-32}
export CBV_RANK=${CBV_RANK:-8}
export MIN_VALID_STARS=${MIN_VALID_STARS:-16}
DENSE_V2=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE_V2}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export GROUP_ART_DIR=${GROUP_ART_DIR:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_qclean_v1}
export DECODER_CKPT=${DECODER_CKPT:-$GROUP_ART_DIR/single_star_weight_decode/decoder.pth}
export CBV_BASES_NPZ=${CBV_BASES_NPZ:-$GROUP_ART_DIR/area_group_cbv_r8_g32_mv16_q16437_tglc0_1607e67857039a07.npz}

sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || echo MISSING; }
STAGE_B="$CKPT_DIR/group_cbv_mlp_g${GROUP_SIZE}_r${CBV_RANK}_mv${MIN_VALID_STARS}_s${SEED}_best.pth"
echo "================ resolved configuration ================"
echo "  node          : $(hostname)"
echo "  git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  STAGE_B_CKPT  : $STAGE_B  ($(sha "$STAGE_B"))"
echo "  DECODER_CKPT  : $DECODER_CKPT  ($(sha "$DECODER_CKPT"))"
echo "  CBV_BASES_NPZ : $CBV_BASES_NPZ  ($(sha "$CBV_BASES_NPZ"))"
echo "========================================================"
nvidia-smi || true

"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device visible on $(hostname) -- aborting (add --exclude=$(hostname -s))."
  exit 1
}

"$PY" -m src.instrument_v2.eval_cbv_oracle_ceiling
echo "=== DONE cbv_oracle ==="
