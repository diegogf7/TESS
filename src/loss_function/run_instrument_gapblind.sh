#!/bin/bash -l
#SBATCH -J instrument_gapblind
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o instrument_gapblind_%j.out

# Gap-blind retrain: same model, same masked loss, but the encoders never see
# gap placement (infilled inputs, no masks). New checkpoint; nothing existing
# gets overwritten.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

export JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_gapblind.pth

echo "=== TRAIN (gap-blind instrument JEPA) ==="
python -m src.loss_function.train_instrument_gapblind

echo "=== GAP-BLIND PROBE (flux-borne sector signal is the verdict) ==="
python -m src.loss_function.gapblind_probe

echo "=== DONE ==="
