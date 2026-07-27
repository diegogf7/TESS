#!/bin/bash -l
#SBATCH -J phyts_ctrl
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o phyts_ctrl_%j.out
# Frozen 2x2 preprocessing diagnostic (native vs grid) x (raw vs cleaned).
# No training, no model/decoder modification. Run:
#   sbatch -p mit_normal_gpu \
#     --export=ALL,TGLC_PATH=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_phyts_s14.parquet,JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth,JEPA_READOUT=mean_std,JEPA_NTOKENS=16 \
#     src/instrument_v2/submit_phyts_native_cleaning_controls.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

# resolve the exact frozen config used for raw=0.629 / cleaned=0.645
export JEPA_CKPT=${JEPA_CKPT:-/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth}
export JEPA_READOUT=${JEPA_READOUT:-mean_std}
export JEPA_NTOKENS=${JEPA_NTOKENS:-16}
export TGLC_PATH=${TGLC_PATH:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_phyts_s14.parquet}
export INST_CKPT=${INST_CKPT:-/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_qclean_v1/group_cbv_mlp_g32_r8_mv16_s0_best.pth}
export DECODER_CKPT=${DECODER_CKPT:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1/single_star_decode/decoder.pth}
export GRID_RANGE=${GRID_RANGE:-artifacts/instrument_v2/sector14_jepa_dense_v2/grid_range.json}

echo "================ resolved configuration ================"
echo "  node        : $(hostname)"
echo "  PY          : $PY"
echo "  REPO        : $REPO"
echo "  JEPA_CKPT   : $JEPA_CKPT"
echo "  JEPA_READOUT: $JEPA_READOUT"
echo "  JEPA_NTOKENS: $JEPA_NTOKENS"
echo "  TGLC_PATH   : $TGLC_PATH"
echo "  INST_CKPT   : $INST_CKPT"
echo "  DECODER_CKPT: $DECODER_CKPT"
echo "  GRID_RANGE  : $GRID_RANGE"
echo "========================================================"
nvidia-smi || true

"$PY" -m src.instrument_v2.eval_phyts_native_cleaning_controls
echo "=== DONE ==="
