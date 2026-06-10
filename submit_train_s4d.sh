#!/bin/bash
#SBATCH --job-name=tess_s4d
#SBATCH --partition=orcd_ug_pg_mki_aryeh_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/orcd/scratch/orcd/006/diegogon/logs/tess_s4d_%j.out
#SBATCH --error=/orcd/scratch/orcd/006/diegogon/logs/tess_s4d_%j.err

source $SCRATCH/miniforge3/etc/profile.d/conda.sh
conda activate lightcurve

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints
mkdir -p /orcd/scratch/orcd/006/diegogon/logs

cd /orcd/scratch/orcd/006/diegogon/TESS/src

python train_s4d.py
