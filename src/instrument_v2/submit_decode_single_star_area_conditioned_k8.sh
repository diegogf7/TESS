#!/bin/bash -l
#SBATCH -J cbv_areacond
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o cbv_areacond_%j.out

# Area-conditioned K=8 CBV-weight decoder: decoder_input = concat(latent, area
# one-hot). Trains ONLY the MLP decoder, then compares it to the existing GLOBAL
# weight decoder with the oracle-ceiling metrics. Frozen instrument JEPA, CBV
# bases, teacher/student/predictor are untouched.
#   sbatch -p mit_normal_gpu src/instrument_v2/submit_decode_single_star_area_conditioned_k8.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

# same fixed config + dense_v2 env as the global weight decoder (footgun: without
# these the SVD/splits differ from the trained bases)
export DECODER_MODE=weights_area_conditioned
export SEED=${SEED:-0}
export GROUP_SIZE=${GROUP_SIZE:-32}
export CBV_RANK=${CBV_RANK:-8}
export MIN_VALID_STARS=${MIN_VALID_STARS:-16}
export RIDGE_LAMBDA=${RIDGE_LAMBDA:-1e-2}
export EPOCHS=${EPOCHS:-20}
DENSE_V2=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE_V2}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export GROUP_ART_DIR=${GROUP_ART_DIR:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_qclean_v1}
export CBV_BASES_NPZ=${CBV_BASES_NPZ:-$GROUP_ART_DIR/area_group_cbv_r8_g32_mv16_q16437_tglc0_1607e67857039a07.npz}

if [[ "$GROUP_SIZE" != "32" || "$CBV_RANK" != "8" || "$MIN_VALID_STARS" != "16" ]]; then
    echo "FATAL: matched comparison requires 32/8/16 (got $GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS)" >&2
    exit 1
fi

sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || echo MISSING; }
STAGE_B="$CKPT_DIR/group_cbv_mlp_g${GROUP_SIZE}_r${CBV_RANK}_mv${MIN_VALID_STARS}_s${SEED}_best.pth"
echo "================ resolved configuration ================"
echo "  node          : $(hostname)"
echo "  git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  DECODER_MODE  : $DECODER_MODE"
echo "  STAGE_B_CKPT  : $STAGE_B  ($(sha "$STAGE_B"))"
echo "  GLOBAL DECODER: $GROUP_ART_DIR/single_star_weight_decode/decoder.pth  ($(sha "$GROUP_ART_DIR/single_star_weight_decode/decoder.pth"))"
echo "  CBV_BASES_NPZ : $CBV_BASES_NPZ  ($(sha "$CBV_BASES_NPZ"))"
echo "========================================================"
nvidia-smi || true

"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device visible on $(hostname) -- aborting (add --exclude=$(hostname -s))."
  exit 1
}

echo "=== train area-conditioned decoder ==="
"$PY" -m src.instrument_v2.decode_single_star_k8
echo "=== compare vs global decoder (oracle-ceiling metrics) ==="
"$PY" -m src.instrument_v2.eval_cbv_area_conditioned
echo "=== DONE cbv_areacond ==="
