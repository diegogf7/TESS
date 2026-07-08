#!/bin/bash -l
#SBATCH -J physics_tglc200k
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -o physics_tglc200k_%j.out

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

# ms16 winning config from the predictor A/B
export JEPA_NTOKENS=16
export JEPA_READOUT=mean_std
export JEPA_MASK_RATIO=0.5
export JEPA_PREDICTOR=transformer

# new checkpoint, new data -- nothing existing gets overwritten
export JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/latent_jepa_tglc200k.pth
export JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_pretrain_train.parquet
export JEPA_VAL=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_pretrain_val.parquet

echo "=== TRAIN physics JEPA on TGLC 200k ==="
python -m src.worked_folder.physics.train_jepa

echo "=== DONE ==="
