#!/bin/bash -l
#SBATCH -J phyts_ft_ab
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o phyts_ft_ab_%A_%a.out
# Matched SUPERVISED fine-tuning A/B (raw vs instrument-cleaned TGLC). One array
# task per seed (SLURM_ARRAY_TASK_ID). Physics encoder + head are fine-tuned;
# the instrument JEPA + decoder stay frozen. Run:
#   sbatch -p ou_mki_gpu --array=0-2%3 \
#     --export=ALL,TGLC_PATH=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_phyts_s14.parquet \
#     src/instrument_v2/submit_finetune_phyts_raw_tglc_ab.sh
# then aggregate: $PY -m src.instrument_v2.finetune_phyts_raw_tglc_ab --aggregate-only
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

# the exact physics model behind raw=0.629 / cleaned=0.645 (env-overridable, not hard-coded in .py)
export JEPA_CKPT=${JEPA_CKPT:-/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_ms16.pth}
export JEPA_READOUT=${JEPA_READOUT:-mean_std}
export JEPA_NTOKENS=${JEPA_NTOKENS:-16}

echo "=== node: $(hostname) | seed(array)=${SLURM_ARRAY_TASK_ID:-0} ==="
echo "    JEPA_CKPT=$JEPA_CKPT READOUT=$JEPA_READOUT NTOKENS=$JEPA_NTOKENS"
nvidia-smi || true

"$PY" -m src.instrument_v2.finetune_phyts_raw_tglc_ab
echo "=== DONE seed ${SLURM_ARRAY_TASK_ID:-0} ==="
