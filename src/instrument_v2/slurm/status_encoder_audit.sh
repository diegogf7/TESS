#!/bin/bash
# All this code is from Claude
# Status of the encoder audit: bash src/instrument_v2/slurm/status_encoder_audit.sh
set -euo pipefail
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
RUN_ID=${1:-abl1_encoder_audit}
NEW_RUN=artifacts/instrument_v2/ablation/$RUN_ID
MANIFEST=$NEW_RUN/job_manifest.json
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST"; exit 1; }

echo "RUN_ID: $RUN_ID"
echo "commit: $(python3 -c "import json;print(json.load(open('$MANIFEST'))['git_commit'])")"
echo "logs:   $(python3 -c "import json;print(json.load(open('$MANIFEST'))['log_dir'])")"
echo
for stage in preflight fixed select_online finetune select_ft final; do
  ID=$(python3 -c "import json;print(json.load(open('$MANIFEST'))['jobs']['$stage']['id'])")
  echo "--- $stage (job $ID) ---"
  sacct -j "$ID" --format=JobID%18,State%14,Elapsed,ExitCode --noheader 2>/dev/null \
    | grep -v '\.extern\|\.batch\|^$' || echo "  (no sacct info yet)"
done
echo
if [ -f "$NEW_RUN/final_summary.md" ]; then
  echo "FINAL REPORT READY: $NEW_RUN/final_summary.md"
else
  echo "final report not written yet: $NEW_RUN/final_summary.md"
fi
