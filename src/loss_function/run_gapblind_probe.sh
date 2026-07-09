#!/bin/bash -l
#SBATCH -J gapblind_probe
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o gapblind_probe_%j.out

# Diagnostic only, no training: decomposes each checkpoint's sector signal
# into mask-borne vs flux-borne. Safe to run any time, changes nothing.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== PROBE: instrument_jepa.pth (original, gap-attached baseline) ==="
JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa.pth \
    python -m src.loss_function.gapblind_probe

echo "=== PROBE: instrument_jepa_gapmask.pth (masked-loss retrain) ==="
JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_gapmask.pth \
    python -m src.loss_function.gapblind_probe

echo "=== DONE ==="
