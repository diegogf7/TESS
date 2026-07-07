#!/bin/bash -l
#SBATCH -J physics_pair
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o physics_pair_%j.out

# --- cross-sector physics JEPA: same-star/diff-sector context -> anchor ---
cd /orcd/scratch/orcd/006/diegogon/TESS

conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints

echo "=== TRAIN (cross-sector physics-pair JEPA, readout=mean_std) ==="
JEPA_READOUT=mean_std python -m src.worked_folder.archive.train_physics_pair_jepa && \
echo "=== EVAL (probe: CLASS want high, SECTOR want low) ===" && \
JEPA_READOUT=mean_std python -m src.worked_folder.archive.eval_physics_pair_jepa

echo "=== DONE ===" 
