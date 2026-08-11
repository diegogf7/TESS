#!/bin/bash
# Check the ORCD environment BEFORE starting a 22 GB download or queuing a GPU job.
# Run on a login node:   bash disentangle_attempt/orcd/00_preflight.sh
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "repo:      $REPO"
echo "data dir:  $DATA_DIR"
echo "output:    $OUT_DIR"
echo "python:    $(which python)"
echo

fail=0

echo "--- required packages ---"
python - <<'PY'
import importlib, sys
required = [
    ("numpy", None), ("pandas", None), ("pyarrow", None), ("yaml", "pyyaml"),
    ("requests", None), ("matplotlib", None), ("astropy", None),
    ("tess_stars2px", "tess-point"), ("torch", None),
]
missing = []
for module, pip_name in required:
    try:
        m = importlib.import_module(module)
        print(f"  ok      {module:16s} {getattr(m, '__version__', '')}")
    except Exception as exc:
        missing.append(pip_name or module)
        print(f"  MISSING {module:16s} ({type(exc).__name__})")
if missing:
    print("\ninstall with:\n  pip install " + " ".join(sorted(set(missing))))
    sys.exit(1)
PY
[ $? -ne 0 ] && fail=1

echo
echo "--- GPU visibility (expect 'no CUDA' on a login node; the GPU job checks again) ---"
python -c "
import torch
print('  torch.cuda.is_available():', torch.cuda.is_available())
print('  device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
" 2>&1 | sed 's/^/  /'

echo
echo "--- outbound network to the archive ---"
if curl -sf -o /dev/null --max-time 25 -r 0-1023 \
    "https://archive.stsci.edu/hlsps/tglc/download_scripts/hlsp_tglc_tess_ffi_s0001_tess_v1_llc.sh"; then
  echo "  ok      archive.stsci.edu reachable"
else
  echo "  FAILED  archive.stsci.edu unreachable -- stage 01 must run somewhere with"
  echo "          outbound HTTPS (a login node), not on a compute node."
  fail=1
fi

echo
echo "--- quota (need ~22 GB AND headroom in the FILE-COUNT quota) ---"
echo "  filesystem free space:"
df -h "$(dirname "$DATA_DIR")" | tail -1 | sed 's/^/    /'
echo "  filesystem free inodes:"
df -i "$(dirname "$DATA_DIR")" | tail -1 | sed 's/^/    /'
echo
echo "  YOUR quota (the shared filesystem above is not the limit that bites):"
if command -v lfs >/dev/null 2>&1 && lfs quota -u "$USER" "$DATA_DIR" 2>/dev/null | tail -3 | sed 's/^/    /'; then
  :
elif quota -s 2>/dev/null | tail -4 | sed 's/^/    /'; then
  :
else
  echo "    (no quota command found -- read the QUOTA REPORT printed at login)"
fi
echo
echo "  This job writes ~240,000 light-curve files. With --purge-fits (the default in"
echo "  01_acquire.sh) only one chip of ~3,000 files exists at a time, so the steady"
echo "  cost is ~80 parquet chunks. Without it you need ~240,000 free files."

echo
if [ "$fail" -eq 0 ]; then
  echo "PREFLIGHT OK -- proceed to 01_acquire.sh"
else
  echo "PREFLIGHT FAILED -- fix the items above first"
fi
exit $fail
