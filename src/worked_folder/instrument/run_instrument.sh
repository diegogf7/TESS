#!/bin/bash -l
#SBATCH -J instrument_gapmask
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o instrument_gapmask_%j.out

# Gap-masked retrain: full curves (no splitting), gap tokens get zero weight
# in the loss so the encoder can't score points off the placeholder zeros.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

# new checkpoint -- old instrument_jepa.pth stays untouched as the baseline
export JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_gapmask.pth

echo "=== TRAIN (gap-masked instrument JEPA) ==="
python -m src.worked_folder.instrument.train_instrument_jepa

echo "=== EVAL (sector / class probes) ==="
python -m src.worked_folder.instrument.eval_instrument_jepa

echo "=== ZERO-FLUX CONTROL (did the shortcut die?) ==="
python -m src.worked_folder.analysis.zero_flux_control

echo "=== DONE ==="
