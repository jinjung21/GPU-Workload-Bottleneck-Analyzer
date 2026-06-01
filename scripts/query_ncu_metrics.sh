#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NCU="${NCU:-$CUDA_HOME/bin/ncu}"
OUT_DIR="${OUT_DIR:-profiles/ncu}"

mkdir -p "$OUT_DIR"

if [[ ! -x "$NCU" ]]; then
  echo "ncu not found at $NCU" >&2
  exit 1
fi

"$NCU" --query-metrics > "$OUT_DIR/available_metrics.txt"

for pattern in hit stall warp branch dram l2 l1tex occupancy transaction sector; do
  grep -i "$pattern" "$OUT_DIR/available_metrics.txt" > "$OUT_DIR/available_metrics_${pattern}.txt" || true
done

echo "Wrote metric query files under $OUT_DIR"
echo "Start with:"
echo "less $OUT_DIR/available_metrics_hit.txt"
echo "less $OUT_DIR/available_metrics_stall.txt"
echo "less $OUT_DIR/available_metrics_warp.txt"
