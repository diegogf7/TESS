#!/bin/bash -l
#SBATCH -J tglc_jepa_ab
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o tglc_jepa_ab_%A_%a.out
# Matched raw-vs-cleaned physics-JEPA pretraining A/B on raw TGLC (Sector 14).
# STAGE in {prepare, train, evaluate}. For train, the array task picks the arm
# (0=raw, 1=cleaned). Examples:
#   sbatch -p mit_normal_gpu --export=ALL,STAGE=prepare,SEED=0 src/instrument_v2/submit_tglc_physics_jepa_ab.sh
#   sbatch -p mit_normal_gpu --array=0-1%2 --export=ALL,STAGE=train,SEED=0 src/instrument_v2/submit_tglc_physics_jepa_ab.sh
#   sbatch -p mit_normal_gpu --export=ALL,STAGE=evaluate,SEED=0 src/instrument_v2/submit_tglc_physics_jepa_ab.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

# architecture/config for build_latent_jepa (identical for both arms)
export JEPA_NTOKENS=${JEPA_NTOKENS:-16}
export JEPA_READOUT=${JEPA_READOUT:-mean_std}
export JEPA_PREDICTOR=${JEPA_PREDICTOR:-transformer}
export JEPA_MASK_RATIO=${JEPA_MASK_RATIO:-0.5}

STAGE=${STAGE:?set STAGE=prepare|train|evaluate}
SEED=${SEED:-0}
PRETRAIN_PATH=${PRETRAIN_PATH:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet}
EVAL_TGLC_PATH=${EVAL_TGLC_PATH:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_phyts_s14.parquet}
INST_CKPT=${INST_CKPT:-/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_qclean_v1/group_cbv_mlp_g32_r8_mv16_s0_best.pth}
DECODER_CKPT=${DECODER_CKPT:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1/single_star_decode/decoder.pth}
GRID_RANGE=${GRID_RANGE:-artifacts/instrument_v2/sector14_jepa_dense_v2/grid_range.json}
export PRETRAIN_PATH EVAL_TGLC_PATH INST_CKPT DECODER_CKPT GRID_RANGE

ARM="${ARM:-}"                              # honor an exported ARM (raw|cleaned|cbv); do NOT wipe it
ARGS="--stage $STAGE --seed $SEED"
if [ "$STAGE" = "train" ]; then
  case "${SLURM_ARRAY_TASK_ID:-none}" in    # an array task still maps 0=raw 1=cleaned
    0) ARM=raw ;; 1) ARM=cleaned ;;
  esac
  if [ -z "$ARM" ]; then                     # never silently default to raw -- it would overwrite the raw ckpt
    echo "FATAL: train stage needs ARM=raw|cleaned|cbv (or array task 0/1)" >&2; exit 1
  fi
  ARGS="$ARGS --arm $ARM"
fi

echo "================ resolved configuration ================"
echo "  node          : $(hostname)"
echo "  STAGE         : $STAGE"
echo "  SEED          : $SEED"
echo "  ARM           : ${ARM:-n/a}   (array task ${SLURM_ARRAY_TASK_ID:-n/a}: 0=raw 1=cleaned)"
echo "  PRETRAIN_PATH : $PRETRAIN_PATH"
echo "  EVAL_TGLC_PATH: $EVAL_TGLC_PATH"
echo "  INST_CKPT     : $INST_CKPT"
echo "  DECODER_CKPT  : $DECODER_CKPT"
echo "  GRID_RANGE    : $GRID_RANGE"
echo "  JEPA config   : NTOKENS=$JEPA_NTOKENS READOUT=$JEPA_READOUT PREDICTOR=$JEPA_PREDICTOR MASK_RATIO=$JEPA_MASK_RATIO"
echo "  git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "========================================================"
nvidia-smi || true

# fail fast if this node has no usable GPU (some ou_mki nodes hand out no device)
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: no CUDA device visible on $(hostname) -- aborting instead of training on CPU."
  echo "       resubmit, and add --exclude=$(hostname -s) to skip this node."
  exit 1
}

# The arm is passed on the command line ($ARGS has --arm). Clear the ARM env var
# so it can't leak into unrelated modules -- src/instrument_v2/train_sector14_jepa
# reads os.environ["ARM"] for its GRID arm and asserts it is "shared"/"legacy",
# so an exported ARM=cbv would crash that import.
DONE_ARM="${ARM:-}"
unset ARM
"$PY" -m src.instrument_v2.run_tglc_physics_jepa_ab $ARGS
echo "=== DONE $STAGE ${DONE_ARM:-} seed $SEED ==="
