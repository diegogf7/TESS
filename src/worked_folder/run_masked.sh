#!/bin/bash -l
#SBATCH -J masked_s4d
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o masked_s4d_%j.out

# --- run detached on a GPU node: train then eval, sequentially ---
cd /orcd/scratch/orcd/006/diegogon/TESS

conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

# fresh re-run baseline -> own checkpoint, so it never overwrites the saved 0.473 model
export MASKED_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/masked_s4d_rerun.pth

echo "=== TRAIN (self-supervised masked pretraining) -> $MASKED_CKPT ==="
python -m src.worked_folder.train_masked && \
echo "=== EVAL (freeze + probe SECTOR/CLASS) ===" && \
python -m src.worked_folder.eval_masked

echo "=== DONE ==="
