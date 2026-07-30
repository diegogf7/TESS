#!/bin/bash -l
#SBATCH -J partb_dyn1000
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o partb_dyn1000_%j.out

# PART B -- DO NOT SUBMIT UNTIL PART A (regteacher_g32_n1000_s0) FINISHES OK.
# Retrains the Stage-B student on the NEW frozen Part-A teacher: exactly 1,000
# context stars per area per epoch, each with a dynamic 32-star same-area group
# (context excluded), resampled every epoch -> 1,000 context-group examples per
# area per epoch. Group size 32, MIN_VALID=16, CBV rank 8 (unchanged). Then
# retrains the area-head decoder and reports by camera.
#   sbatch -p mit_normal_gpu src/instrument_v2/submit_partb_dynamic_from_scaled_teacher.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

export SEED=${SEED:-0}
export GROUP_SIZE=${GROUP_SIZE:-32}          # group size 32
export CBV_RANK=${CBV_RANK:-8}               # CBV rank 8 -- NOT 32
export MIN_VALID_STARS=${MIN_VALID_STARS:-16}
export N_CONTEXT=${N_CONTEXT:-1000}
export REQUIRE_FULL=${REQUIRE_FULL:-1}       # 0 = keep areas < N_CONTEXT (use all available stars/area)
export EPOCHS=${EPOCHS:-15}
DENSE_V2=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE_V2}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
# the NEW Part-A teacher selection (regteacher_g32_n1000_s0)
export TEACHER_SELECTION=${TEACHER_SELECTION:-artifacts/instrument_v2/regteacher_g32_n1000_v1/selection_regteacher_g32_n1000_s0.json}
# new Part-B outputs (never overwrite the qclean_v1 pipeline)
export GROUP_ART_DIR=${GROUP_ART_DIR:-artifacts/instrument_v2/custom_group32_cbv8_from_scaled_teacher_v1}
export CKPT_DIR=${CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_from_scaled_teacher_v1}

if [ ! -f "$TEACHER_SELECTION" ]; then
  echo "FATAL: Part-A selection not found: $TEACHER_SELECTION -- run Part A first." >&2; exit 1
fi
if [[ "$CBV_RANK" != "8" || "$GROUP_SIZE" != "32" || "$MIN_VALID_STARS" != "16" ]]; then
  echo "FATAL: must be GROUP_SIZE=32 CBV_RANK=8 MIN_VALID_STARS=16 (got $GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS)" >&2; exit 1
fi

TAG=group_cbv_mlp_dyn${N_CONTEXT}_g${GROUP_SIZE}_r${CBV_RANK}_mv${MIN_VALID_STARS}_s${SEED}
NEW_CKPT=$CKPT_DIR/${TAG}_best.pth
echo "================ resolved configuration ================"
echo "  node          : $(hostname)"
echo "  git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  TEACHER_SEL   : $TEACHER_SELECTION"
echo "  N_CONTEXT/GROUP/RANK/MINVALID: $N_CONTEXT/$GROUP_SIZE/$CBV_RANK/$MIN_VALID_STARS"
echo "  ART/CKPT      : $GROUP_ART_DIR | $CKPT_DIR"
echo "========================================================"
nvidia-smi || true
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device on $(hostname) -- aborting (add --exclude=$(hostname -s))."; exit 1; }

echo "=== Part B (1): dynamic 1000-context Stage-B student on new frozen teacher ==="
"$PY" -m src.instrument_v2.train_instrument_dynamic_groups
echo "=== Part B (2): global weight decoder (warm-start source) ==="
DECODER_MODE=weights STAGE_B_CKPT=$NEW_CKPT "$PY" -m src.instrument_v2.decode_single_star_k8
echo "=== Part B (3): area-head decoder ==="
DECODER_MODE=weights_area_heads STAGE_B_CKPT=$NEW_CKPT \
GLOBAL_DECODER_CKPT=$GROUP_ART_DIR/single_star_weight_decode/decoder.pth \
"$PY" -m src.instrument_v2.decode_single_star_k8
echo "=== Part B (4): by-camera Pearson/R2/neg-rate report ==="
STAGE_B_CKPT=$NEW_CKPT "$PY" -m src.instrument_v2.eval_dynamic_area_heads
echo "=== DONE Part B ==="
