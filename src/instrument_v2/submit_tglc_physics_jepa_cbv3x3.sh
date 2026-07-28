#!/bin/bash -l
#SBATCH -J tglc_jepa_cbv3x3
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o tglc_jepa_cbv3x3_%j.out

# All code by Claude. Matched 3x3 physics-JEPA comparison: raw vs direct-cleaned
# vs CBV-weight-cleaned. Reuses the frozen tglc_physics_jepa_ab raw/direct
# arrays, init and checkpoints (hard-fails when stale); trains ONLY the cbv arm.
# STAGE in {prepare, train, evaluate}:
#   sbatch --export=ALL,STAGE=prepare,SEED=0  src/instrument_v2/submit_tglc_physics_jepa_cbv3x3.sh
#   sbatch --export=ALL,STAGE=train,SEED=0    src/instrument_v2/submit_tglc_physics_jepa_cbv3x3.sh
#   sbatch --export=ALL,STAGE=evaluate,SEED=0 src/instrument_v2/submit_tglc_physics_jepa_cbv3x3.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

# physics-JEPA architecture: MUST equal the raw/direct arms (submit_tglc_physics_jepa_ab.sh)
export JEPA_NTOKENS=${JEPA_NTOKENS:-16}
export JEPA_READOUT=${JEPA_READOUT:-mean_std}
export JEPA_PREDICTOR=${JEPA_PREDICTOR:-transformer}
export JEPA_MASK_RATIO=${JEPA_MASK_RATIO:-0.5}

STAGE=${STAGE:?set STAGE=prepare|train|evaluate}
SEED=${SEED:-0}
# same data/instrument inputs as the frozen ab experiment
PRETRAIN_PATH=${PRETRAIN_PATH:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet}
EVAL_TGLC_PATH=${EVAL_TGLC_PATH:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_phyts_s14.parquet}
INST_CKPT=${INST_CKPT:-/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_qclean_v1/group_cbv_mlp_g32_r8_mv16_s0_best.pth}
DECODER_CKPT=${DECODER_CKPT:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1/single_star_decode/decoder.pth}
GRID_RANGE=${GRID_RANGE:-artifacts/instrument_v2/sector14_jepa_dense_v2/grid_range.json}
# cbv-specific artifacts
GROUP_ART_DIR=${GROUP_ART_DIR:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1}
WEIGHT_DECODER_CKPT=${WEIGHT_DECODER_CKPT:-$GROUP_ART_DIR/single_star_weight_decode/decoder.pth}
CBV_BASES_NPZ=${CBV_BASES_NPZ:-}
export PRETRAIN_PATH EVAL_TGLC_PATH INST_CKPT DECODER_CKPT GRID_RANGE
export GROUP_ART_DIR WEIGHT_DECODER_CKPT CBV_BASES_NPZ
# NOTE: never export OUT_DIR here -- it would relocate the frozen ab experiment's
# artifacts; the cbv dirs are CBV3X3_OUT_DIR / CBV3X3_CKPT_DIR (defaults in the module).

sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || echo MISSING; }
echo "================ resolved configuration ================"
echo "  node                : $(hostname)"
echo "  git commit          : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  STAGE / SEED        : $STAGE / $SEED"
echo "  PRETRAIN_PATH       : $PRETRAIN_PATH"
echo "  EVAL_TGLC_PATH      : $EVAL_TGLC_PATH"
echo "  INST_CKPT           : $INST_CKPT"
echo "    sha256            : $(sha "$INST_CKPT")"
echo "  DIRECT DECODER_CKPT : $DECODER_CKPT"
echo "    sha256            : $(sha "$DECODER_CKPT")"
echo "  WEIGHT_DECODER_CKPT : $WEIGHT_DECODER_CKPT"
echo "    sha256            : $(sha "$WEIGHT_DECODER_CKPT")"
echo "  CBV_BASES_NPZ       : ${CBV_BASES_NPZ:-auto (unique glob in $GROUP_ART_DIR)}"
echo "  GRID_RANGE          : $GRID_RANGE"
echo "  JEPA config         : NTOKENS=$JEPA_NTOKENS READOUT=$JEPA_READOUT PREDICTOR=$JEPA_PREDICTOR MASK_RATIO=$JEPA_MASK_RATIO"
echo "========================================================"
nvidia-smi || true

"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device visible on $(hostname) -- aborting instead of training on CPU."
  exit 1
}

"$PY" -m src.instrument_v2.run_tglc_physics_jepa_cbv3x3 --stage "$STAGE" --seed "$SEED"
echo "=== DONE $STAGE seed $SEED ==="
