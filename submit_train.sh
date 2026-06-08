#!/bin/bash
#SBATCH --job-name=tess_mae
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/orcd/scratch/orcd/006/diegogon/logs/tess_mae_%j.out
#SBATCH --error=/orcd/scratch/orcd/006/diegogon/logs/tess_mae_%j.err

source $SCRATCH/miniforge3/etc/profile.d/conda.sh
conda activate lightcurve

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints
mkdir -p /orcd/scratch/orcd/006/diegogon/logs

cd /orcd/scratch/orcd/006/diegogon/TESS/src

python train.py
