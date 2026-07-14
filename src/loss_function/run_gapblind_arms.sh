#!/bin/bash -l
#SBATCH -J gapblind_arms
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-5
#SBATCH -o gapblind_arms_%A_%a.out

# Gap-blind A/B: 2 pairing arms x 3 seeds through train_instrument_gapblind.
# Tasks 0-2 = sector pairing (seeds 0,1,2); tasks 3-5 = chip pairing (seeds 0,1,2).

PY=/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python
cd /orcd/scratch/orcd/006/diegogon/TESS

ARMS=("sector" "sector,camera,ccd")
export JEPA_GROUP=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
export JEPA_SEED=$((SLURM_ARRAY_TASK_ID % 3))

TAG=sector
if [ "$JEPA_GROUP" != "sector" ]; then TAG=chip; fi

export JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_train_40k_area.parquet
export JEPA_VAL=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val_area.parquet
export JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_gapblind_${TAG}_s${JEPA_SEED}.pth

echo "=== node: $(hostname) | task ${SLURM_ARRAY_TASK_ID} | group ${JEPA_GROUP} | seed ${JEPA_SEED} ==="
nvidia-smi
mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

echo "=== TRAIN (gap-blind, ${TAG} pairing, seed ${JEPA_SEED}) ==="
$PY -m src.loss_function.train_instrument_gapblind

echo "=== CHIP PROBES (within-sector / leave-sector-out) ==="
JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val_area.parquet \
    $PY -m src.regions.eval_chip_probes

echo "=== SECTOR PROBE (health check) ==="
JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val_area.parquet \
    $PY -m src.loss_function.gapblind_probe_raw

echo "=== DONE ==="
