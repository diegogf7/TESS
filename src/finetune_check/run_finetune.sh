#!/bin/bash -l
#SBATCH -J finetune_check
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o finetune_check_%j.out

cd /orcd/scratch/orcd/006/diegogon/TESS

conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

echo "1: fine-tune from ms16 JEPA init "
FINETUNE_INIT=jepa FINETUNE_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/ft_jepa_init.pth \
    python -m src.finetune_check.train_finetune

echo "2: from-scratch control (same arch, same data)"
FINETUNE_INIT=random FINETUNE_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/ft_random_init.pth \
    python -m src.finetune_check.train_finetune

echo "finished "
