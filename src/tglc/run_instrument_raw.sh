#!/bin/bash -l
#SBATCH -J instrument_raw
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o instrument_raw_%j.out

# Gapmask instrument JEPA on RAW TGLC flux (systematics included, no
# quality filters). The question: can it learn sector from flux alone?

echo "=== node: $(hostname) ==="
nvidia-smi

export JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_train_40k.parquet
export JEPA_VAL=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val.parquet
export JEPA_CKPT=/orcd/scratch/orcd/006/diegogon/checkpoints/instrument_jepa_raw.pth

echo "=== TRAIN (gapmask instrument JEPA on RAW flux) ==="
python -m src.worked_folder.instrument.train_instrument_jepa

echo "=== GAP-BLIND PROBE (raw val set) ==="
JEPA_DATA=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_val.parquet \
    python -m src.loss_function.gapblind_probe_raw

echo "=== DONE ==="
