#!/bin/bash -l
#SBATCH -J latent_jepa
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o latent_jepa_%j.out

# --- run detached on a GPU node: train then eval the latent-space JEPA ---
cd /orcd/scratch/orcd/006/diegogon/TESS

conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

echo "=== TRAIN (latent-space JEPA pretraining) ==="
python -m src.bot_folder.train_jepa && \
echo "=== EVAL (freeze + probe SECTOR/CLASS) ===" && \
python -m src.bot_folder.eval_jepa

echo "=== DONE ==="
