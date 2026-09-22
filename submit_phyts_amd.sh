#!/bin/bash
#SBATCH --job-name=vicreg_phyts
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Full protocol on the AMD MI210s: Part 1 -> Part 2 -> classification.
# Full architecture from ARCHITECTURE.md (d_model=128, n_layers=4), long enough
# for the variance constraint to actually be satisfied.
set -u
cd "${WORK}/TESS"
PY="${WORK}/venv_rocm/bin/python"
export PYTHONUNBUFFERED=1
export TESS_DEVICE=cuda

NPZ="${NPZ:-artifacts/vicreg_jepa/phyts_s15.npz}"
OUT="${OUT:-artifacts/vicreg_jepa/amd}"
DROP="${DROP:-}"
mkdir -p "$OUT"

echo "=== node $(hostname) ==="
$PY -c "import torch;print('torch',torch.__version__,'gpu',torch.cuda.get_device_name(0))"
echo "git $(git rev-parse --short HEAD)"

echo "=== PART 1 (full arch) ==="
$PY -m src.vicreg_jepa.train --part 1 --source real --npz "$NPZ" \
  --steps1 8000 --eval-every 500 --val-batches 12 --seed 0 \
  --out "$OUT/part1" || exit 1

P1="$OUT/part1/part1/part1_best.pt"

echo "=== PART 2 (all loss weights = 1) ==="
# Every weight at 1. Measured gradient shares are then ~79/13/8/1 (inv/cor/var/cov)
# and the total gradient norm sits near 1.3, so the clipper does not bind.
# Previously mu grew to 130, pushed the total norm to a median of 14.7 against a
# clip of 1.0, and 98.5% of steps were scaled to 6.8% of their size -- throttling
# every term including the ones meant to stop collapse.
$PY -m src.vicreg_jepa.train_part2 --npz "$NPZ" --part1 "$P1" \
  --out "$OUT/part2" \
  --steps 20000 --batch-size 256 \
  --d-model 128 --n-layers 4 --d-state 64 --dropout 0.1 \
  --eval-every 500 --val-n 1024 \
  --phi 1 --lam 1 --mu 1 --nu 1 --grad-clip 5.0 || exit 1

echo "=== CLASSIFICATION ==="
BEST="$OUT/part2/part2_best.pt"
if [ -f "$BEST" ]; then
  $PY -m src.vicreg_jepa.classify --npz "$NPZ" --part1 "$P1" --part2 "$BEST" \
    --random-control ${DROP:+--drop-class "$DROP"} --out "$OUT/classification.json"
else
  echo "NO VALID (non-collapsed) PART 2 CHECKPOINT"
  $PY -m src.vicreg_jepa.classify --npz "$NPZ" --part1 "$P1" \
    --random-control ${DROP:+--drop-class "$DROP"} --out "$OUT/classification.json"
fi
echo "=== DONE ==="
