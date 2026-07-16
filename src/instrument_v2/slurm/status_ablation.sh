#!/bin/bash
# All this code is from Claude
# Status of an ablation run: bash src/instrument_v2/slurm/status_ablation.sh <RUN_ID>
set -euo pipefail
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
RUN_ID=${1:?usage: status_ablation.sh RUN_ID}
ABL_DIR=artifacts/instrument_v2/ablation/$RUN_ID
MANIFEST=$ABL_DIR/job_manifest.json
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST"; exit 1; }

echo "RUN_ID: $RUN_ID"
echo "commit: $(python3 -c "import json;print(json.load(open('$MANIFEST'))['git_commit'])")"
echo "logs:   $(python3 -c "import json;print(json.load(open('$MANIFEST'))['log_dir'])")"
echo
for stage in smoke pretrain select_pre finetune select_ft final; do
  ID=$(python3 -c "import json;print(json.load(open('$MANIFEST'))['jobs']['$stage']['id'])")
  echo "--- $stage (job $ID) ---"
  sacct -j "$ID" --format=JobID%18,State%14,Elapsed,ExitCode --noheader 2>/dev/null \
    | grep -v '\.extern\|\.batch\|^$' || echo "  (no sacct info yet)"
done
echo
FAILED=$(sacct --noheader --format=JobID%18,State -j \
  "$(python3 -c "import json;m=json.load(open('$MANIFEST'));print(','.join(j['id'] for j in m['jobs'].values()))")" \
  2>/dev/null | grep -cE "FAILED|CANCELLED|TIMEOUT|OUT_OF_ME" || true)
echo "failed/cancelled task lines: $FAILED"
if [ -f "$ABL_DIR/final_summary.md" ]; then
  echo "FINAL REPORT READY: $ABL_DIR/final_summary.md"
else
  echo "final report not written yet: $ABL_DIR/final_summary.md"
fi
