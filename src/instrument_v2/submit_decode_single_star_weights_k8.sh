#!/bin/bash -l
#SBATCH -J decode_k8_weights
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o decode_k8_weights_%j.out

# All code by Claude. Matched K=8 CBV-WEIGHT decoder (DECODER_MODE=weights):
# frozen instrument latent -> MLP -> 8 weights; template = B_area @ w with the
# fixed train-only area bases. Trains ONLY the MLP; saves to
# artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1/single_star_weight_decode/
# and never touches the existing direct 1024-output decoder.
#   sbatch src/instrument_v2/submit_decode_single_star_weights_k8.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

# ---- the matched-comparison config is FIXED at 32 / 8 / 16 --------------------
export DECODER_MODE=weights
export SEED=${SEED:-0}
export GROUP_SIZE=${GROUP_SIZE:-32}
export CBV_RANK=${CBV_RANK:-8}
export MIN_VALID_STARS=${MIN_VALID_STARS:-16}
export RIDGE_LAMBDA=${RIDGE_LAMBDA:-1e-2}
export EPOCHS=${EPOCHS:-20}

# ---- dense_v2 data env (the footgun: without these the SVD/splits are wrong;
#      these are the EXACT settings the qclean_v1 direct decoder trained with) --
DENSE_V2=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE_V2}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export GROUP_ART_DIR=${GROUP_ART_DIR:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_qclean_v1}

if [[ "$GROUP_SIZE" != "32" || "$CBV_RANK" != "8" || "$MIN_VALID_STARS" != "16" ]]; then
    echo "FATAL: matched comparison requires GROUP_SIZE=32 CBV_RANK=8 MIN_VALID_STARS=16" \
         "(got $GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS)" >&2
    exit 1
fi

STAGE_B=${STAGE_B_CKPT:-$CKPT_DIR/group_cbv_mlp_g${GROUP_SIZE}_r${CBV_RANK}_mv${MIN_VALID_STARS}_s${SEED}_best.pth}
echo "================ resolved configuration ================"
echo "  node          : $(hostname)"
echo "  git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  DECODER_MODE  : $DECODER_MODE"
echo "  G/R/MV        : $GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS  SEED=$SEED EPOCHS=$EPOCHS"
echo "  S14_DATA      : $S14_DATA"
echo "  SPLIT_DIR     : $SPLIT_DIR"
echo "  BASE_ART_DIR  : $BASE_ART_DIR"
echo "  GROUP_ART_DIR : $GROUP_ART_DIR"
echo "  STAGE_B_CKPT  : $STAGE_B"
echo "  stage-B sha256: $(shasum -a 256 "$STAGE_B" 2>/dev/null | cut -d' ' -f1 || echo MISSING)"
echo "========================================================"
nvidia-smi || true

"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device visible on $(hostname) -- aborting instead of training on CPU."
  exit 1
}

"$PY" -m src.instrument_v2.decode_single_star_k8
echo "=== DONE weight decoder seed $SEED ==="
