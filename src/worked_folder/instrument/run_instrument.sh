#!/bin/bash -l
#SBATCH -J instrument_jepa
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o instrument_jepa_%j.out

# --- cross-star instrument JEPA: pretrain only (eval script comes next) ---
cd /orcd/scratch/orcd/006/diegogon/TESS

conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

echo "=== TRAIN (cross-star instrument JEPA) ==="
python -m src.worked_folder.instrument.train_instrument_jepa

echo "=== DONE ==="
