#!/bin/bash -l
#SBATCH -J instr_dyn1000
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o instr_dyn1000_%j.out

# Retrain the Sector-14 instrument Stage-B student with dynamic 1000-context/area
# sampling (frozen teacher + frozen K=8 CBVs), then retrain the area-head decoder
# on the new encoder and report Pearson/R2/neg-rate by camera. One job:
#   (1) train_instrument_dynamic_groups     -> new encoder+predictor
#   (2) decode DECODER_MODE=weights         -> new GLOBAL decoder (warm-start source)
#   (3) decode DECODER_MODE=weights_area_heads -> new area-head decoder
#   (4) eval_dynamic_area_heads             -> median/Q1 Pearson, Q1 R2, neg-rate, by camera
#   sbatch -p mit_normal_gpu src/instrument_v2/submit_instrument_dynamic_groups.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

export SEED=${SEED:-0}
export GROUP_SIZE=${GROUP_SIZE:-32}
export CBV_RANK=${CBV_RANK:-8}
export MIN_VALID_STARS=${MIN_VALID_STARS:-16}
export N_CONTEXT=${N_CONTEXT:-1000}
export EPOCHS=${EPOCHS:-15}
DENSE_V2=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE_V2}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export SRC_ART_DIR=${SRC_ART_DIR:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1}
export GROUP_ART_DIR=${GROUP_ART_DIR:-artifacts/instrument_v2/custom_group32_cbv8_mlp_dynamic1000_v1}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_dynamic1000_v1}
export MAX_BATCHES=${MAX_BATCHES:-0}

TAG=group_cbv_mlp_dyn${N_CONTEXT}_g${GROUP_SIZE}_r${CBV_RANK}_mv${MIN_VALID_STARS}_s${SEED}
NEW_CKPT=$CKPT_DIR/${TAG}_best.pth

echo "================ resolved configuration ================"
echo "  node          : $(hostname)"
echo "  git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  N_CONTEXT     : $N_CONTEXT   GROUP/RANK/MINVALID: $GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS"
echo "  SRC (teacher+CBVs): $SRC_ART_DIR"
echo "  NEW ART / CKPT: $GROUP_ART_DIR | $CKPT_DIR"
echo "  new encoder   : $NEW_CKPT"
echo "========================================================"
nvidia-smi || true
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device on $(hostname) -- aborting (add --exclude=$(hostname -s))."; exit 1; }

echo "=== (1) dynamic 1000-context Stage-B retrain ==="
"$PY" -m src.instrument_v2.train_instrument_dynamic_groups

echo "=== (2) global weight decoder on the new encoder (warm-start source) ==="
DECODER_MODE=weights STAGE_B_CKPT=$NEW_CKPT "$PY" -m src.instrument_v2.decode_single_star_k8

echo "=== (3) area-head decoder on the new encoder (warm-started from new global) ==="
DECODER_MODE=weights_area_heads STAGE_B_CKPT=$NEW_CKPT \
GLOBAL_DECODER_CKPT=$GROUP_ART_DIR/single_star_weight_decode/decoder.pth \
"$PY" -m src.instrument_v2.decode_single_star_k8

echo "=== (4) by-camera Pearson/R2/neg-rate report ==="
STAGE_B_CKPT=$NEW_CKPT "$PY" -m src.instrument_v2.eval_dynamic_area_heads

echo "=== DONE instr_dyn1000 ==="
