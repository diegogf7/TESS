#!/bin/bash -l
#SBATCH -J instrument_chip
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o instrument_chip_%j.out

# Instrument JEPA, pairing = same (sector, camera, ccd), different star.
# A/B against instrument_jepa_raw_edges.pth (same data, same loss, pairing only).

PY=/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python

echo "=== node: $(hostname) ==="
nvidia-smi

export JEPA_GROUP=sector,camera,ccd
export JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_train_40k_area.parquet
export JEPA_VAL=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val_area.parquet
export JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_chip.pth

echo "=== TRAIN (chip-paired instrument JEPA on RAW flux) ==="
$PY -m src.worked_folder.instrument.train_instrument_jepa

echo "=== CAMERA/CCD PROBE ==="
JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val_area.parquet \
    $PY -m src.regions.eval_camera_ccd

echo "=== SECTOR PROBE (comparability with raw baseline) ==="
JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val_area.parquet \
    $PY -m src.loss_function.gapblind_probe_raw
echo "=== DONE ==="
