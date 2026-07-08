#!/bin/bash -l
#SBATCH -J instrument_split
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o instrument_split_%j.out

# Gap-split retrain: same cross-star instrument JEPA, but every view is a
# single contiguous observed segment (no gaps, no placeholders) so the
# encoder cannot use gap placement as the sector shortcut.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

export JEPA_SPLIT_GAPS=1
export JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_split.pth

echo "=== TRAIN (gap-split instrument JEPA) ==="
python -m src.worked_folder.instrument.train_instrument_jepa

echo "=== EVAL (gap-split probe) ==="
python -m src.worked_folder.instrument.eval_instrument_jepa

echo "=== DONE ==="
