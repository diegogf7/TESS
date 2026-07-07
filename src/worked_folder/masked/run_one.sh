#!/bin/bash -l
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o run_one_%j.out

# Generic single-config launcher for seed-confirmation runs. Everything is
# driven by env vars passed via `sbatch --export`, so one script covers both
# models and any seed/checkpoint:
#   MODEL=jepa   + JEPA_NTOKENS / JEPA_READOUT / JEPA_CKPT
#   MODEL=masked + MASKED_CKPT
# Each job trains then evals (freeze + KNN probe SECTOR/CLASS) into its own ckpt.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname)  MODEL=$MODEL ==="
echo "    JEPA_NTOKENS=$JEPA_NTOKENS JEPA_READOUT=$JEPA_READOUT JEPA_CKPT=$JEPA_CKPT"
echo "    MASKED_CKPT=$MASKED_CKPT"

if [ "$MODEL" = "jepa" ]; then
    python -m src.worked_folder.physics.train_jepa && \
    echo "=== EVAL ===" && \
    python -m src.worked_folder.physics.eval_jepa
elif [ "$MODEL" = "masked" ]; then
    python -m src.worked_folder.masked.train_masked && \
    echo "=== EVAL ===" && \
    python -m src.worked_folder.masked.eval_masked
else
    echo "ERROR: set MODEL=jepa or MODEL=masked"; exit 1
fi

echo "=== DONE ($MODEL) ==="
