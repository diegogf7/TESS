#!/bin/bash -l
#SBATCH -J gapblind_sweep
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o gapblind_sweep_%j.out

# Collapse-fix sweep: gap-blind training with the variance penalty on the
# encoder's RAW context tokens, at three weights. Each arm gets its own
# checkpoint and its own probe. Watch the "latent std" lines: healthy is
# O(0.5-1), the broken run sat at 0.006.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi

mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints
CKPT_DIR=/orcd/scratch/orcd/006/diegogon/checkpoints

for VARW in 0.05 0.1 0.5; do
    TAG=$(echo "varw${VARW}" | tr -d '.')
    export JEPA_VARW=$VARW
    export JEPA_CKPT=$CKPT_DIR/instrument_jepa_gapblind_${TAG}.pth

    echo "=== TRAIN (gap-blind, var fix, weight $VARW) ==="
    python -m src.loss_function.train_instrument_gapblind

    echo "=== PROBE (weight $VARW) ==="
    python -m src.loss_function.gapblind_probe
done

echo "=== DONE ==="
