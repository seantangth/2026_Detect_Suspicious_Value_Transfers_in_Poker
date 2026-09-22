#!/usr/bin/env bash
# Assemble the directory layout in which the cloud-stage scripts were run (see CLOUD_STAGES.md):
#   <root>/cloud  the files of 7_reproduce/lambda_0912/cloud and 7_reproduce/lambda_0913/cloud, side by side
#   <root>/data   raw/ = the competition files; processed/ = four tables written by step s01 of run_all.sh
# Usage: 7_reproduce/cloud_layout.sh <root>        (run step s01 of run_all.sh first)
# Then:  export TPDS_CLOUD_ROOT=<root>; cd <root>/cloud; python3 build_tables.py --data <root>/data --out <root>/tables; ...
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
R=${1:?usage: cloud_layout.sh <root>}
RAW="$REPO/1_data/raw/detect-suspicious-value-transfers-in-poker"
PROC="$REPO/1_data/processed"
for f in action_ctx seat_l0 player_baselines cand_pairs_development; do
  [ -f "$PROC/$f.parquet" ] || { echo "missing $PROC/$f.parquet: run step s01 of run_all.sh first" >&2; exit 1; }
done
mkdir -p "$R/cloud" "$R/data/raw" "$R/data/processed"
for f in "$REPO"/7_reproduce/lambda_0912/cloud/* "$REPO"/7_reproduce/lambda_0913/cloud/*; do if [ -f "$f" ]; then cp "$f" "$R/cloud/"; fi; done
cp "$RAW"/*.parquet "$RAW"/*.csv "$R/data/raw/"
for f in action_ctx seat_l0 player_baselines cand_pairs_development; do cp "$PROC/$f.parquet" "$R/data/processed/"; done
echo "layout ready in $R"
