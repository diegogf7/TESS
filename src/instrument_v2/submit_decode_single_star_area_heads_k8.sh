#!/bin/bash -l
#SBATCH -J cbv_areaheads
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o cbv_areaheads_%j.out

# Area-HEAD K=8 CBV-weight decoder: shared trunk + one 8-weight head per area,
# warm-started from the global decoder. Decoder-only: instrument JEPA, teacher,
# student, predictor, physics JEPA and CBV bases are all frozen. One job:
#   (1) train the area-head decoder
#   (2) 3-way decoder comparison (global | one-hot | heads), oracle-ceiling metrics
#   (3) matched PhyTS physics A/B with the area-head cleaning (frozen ms16 JEPA)
#   sbatch -p mit_normal_gpu src/instrument_v2/submit_decode_single_star_area_heads_k8.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

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
    echo "FATAL: requires 32/8/16 (got $GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS)" >&2; exit 1
fi

sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || echo MISSING; }
GLOBAL=$GROUP_ART_DIR/single_star_weight_decode/decoder.pth
echo "================ resolved configuration ================"
echo "  node          : $(hostname)"
echo "  git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  GLOBAL DECODER: $GLOBAL  ($(sha "$GLOBAL"))"
echo "  ONE-HOT DEC   : $GROUP_ART_DIR/single_star_weight_decode_area_conditioned/decoder.pth  ($(sha "$GROUP_ART_DIR/single_star_weight_decode_area_conditioned/decoder.pth"))"
echo "  CBV_BASES_NPZ : $CBV_BASES_NPZ  ($(sha "$CBV_BASES_NPZ"))"
echo "========================================================"
nvidia-smi || true
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device visible on $(hostname) -- aborting (add --exclude=$(hostname -s))."; exit 1; }

echo "=== (1) train area-head decoder ==="
DECODER_MODE=weights_area_heads "$PY" -m src.instrument_v2.decode_single_star_k8

echo "=== (2) 3-way decoder comparison (global | one-hot | heads) ==="
"$PY" -m src.instrument_v2.eval_cbv_area_heads

echo "=== (3) matched PhyTS physics A/B, area-head cleaning (frozen ms16 JEPA) ==="
CLEAN_MODE=cbv AREA_CONDITIONED=1 \
AREA_DECODER=$GROUP_ART_DIR/single_star_weight_decode_area_heads/decoder.pth \
JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth \
JEPA_READOUT=mean_std JEPA_NTOKENS=16 \
TGLC_PATH=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_phyts_s14.parquet \
OUT_DIR=artifacts/instrument_v2/phyts_raw_tglc_cbv_areaheads \
"$PY" -m src.instrument_v2.eval_phyts_raw_tglc_ab

echo "=== DONE cbv_areaheads ==="
