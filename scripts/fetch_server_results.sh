#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-intern_euijin@165.132.112.154}"
REMOTE_REPO="${REMOTE_REPO:-~/GPU-Workload-Bottleneck-Analyzer}"
REMOTE_PIM_RESULTS="${REMOTE_PIM_RESULTS:-~/pim-tools/pim-results}"
OUTPUT_NAME="${1:-gpu_profile_with_sait_pim}"

mkdir -p outputs profiles simulators

echo "Fetching analyzer output: $OUTPUT_NAME"
scp -r "$SERVER:$REMOTE_REPO/outputs/$OUTPUT_NAME" ./outputs/

echo "Fetching GPU profile CSV"
scp "$SERVER:$REMOTE_REPO/profiles/gpu_profile.csv" ./profiles/

echo "Fetching SAIT PIM simulation CSV if present"
scp "$SERVER:$REMOTE_REPO/simulators/sait_pim_simulation.csv" ./simulators/ || true

echo "Fetching raw SAIT PIMSimulator logs if present"
scp -r "$SERVER:$REMOTE_PIM_RESULTS" ./simulators/ || true

echo "Fetched results into local outputs/, profiles/, and simulators/."
