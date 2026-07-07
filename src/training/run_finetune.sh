#!/bin/bash -l
#SBATCH -J jepa_finetune
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o jepa_finetune_%j.out

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

python -m src.training.train_finetune_jepa

echo "=== DONE ==="
