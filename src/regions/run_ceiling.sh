#!/bin/bash -l
#SBATCH -J ceiling
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-2
#SBATCH -o ceiling_%A_%a.out

# Supervised single-curve ceiling: one sector, one target, three seeds.
# Usage:
#   sbatch --export=ALL,TESS_SECTOR=14,TESS_TARGET=camera src/regions/run_ceiling.sh
# TESS_TARGET in {camera, camccd, area}. Array task id = seed.


#all code by claude code
PY=/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python
cd /orcd/scratch/orcd/006/diegogon/TESS

: "${TESS_SECTOR:?set TESS_SECTOR via --export}"
: "${TESS_TARGET:?set TESS_TARGET via --export}"

export TESS_SECTOR TESS_TARGET
export JEPA_SEED=$SLURM_ARRAY_TASK_ID
export JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/ceiling_${TESS_TARGET}_sec${TESS_SECTOR}_s${JEPA_SEED}.pth

echo "=== node: $(hostname) | sector ${TESS_SECTOR} | target ${TESS_TARGET} | seed ${JEPA_SEED} ==="
nvidia-smi

$PY -m src.regions.train_instrument_ceiling

echo "=== DONE ==="
