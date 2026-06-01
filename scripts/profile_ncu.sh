#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NCU="${NCU:-$CUDA_HOME/bin/ncu}"
OUT_DIR="${OUT_DIR:-profiles/ncu}"
LAUNCH_SKIP="${LAUNCH_SKIP:-1}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
NCU_EXTRA_ARGS="${NCU_EXTRA_ARGS:-}"

mkdir -p "$OUT_DIR"

if [[ ! -x "$NCU" ]]; then
  echo "ncu not found at $NCU" >&2
  exit 1
fi

benchmarks=(
  vector_add
  saxpy
  random_gather
  reduction
  scan
  matrix_transpose
  gemv
  matrix_mul_tiled
  cublas_sgemm
)

for benchmark in "${benchmarks[@]}"; do
  if [[ ! -x "build/$benchmark" ]]; then
    echo "Missing benchmark executable: build/$benchmark" >&2
    echo "Run scripts/build_benchmarks.sh first." >&2
    exit 1
  fi
done

for benchmark in "${benchmarks[@]}"; do
  exe="build/$benchmark"
  log="$OUT_DIR/${benchmark}_ncu.txt"
  echo "Profiling $benchmark with Nsight Compute"
  extra_args=()
  if [[ -n "$NCU_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=($NCU_EXTRA_ARGS)
  fi
  "$NCU" \
    --target-processes all \
    --launch-skip "$LAUNCH_SKIP" \
    --launch-count "$LAUNCH_COUNT" \
    "${extra_args[@]}" \
    "$exe" > "$log"
done

echo "Wrote Nsight Compute reports: $OUT_DIR/*_ncu.txt"
echo
echo "Next command:"
echo "python3 scripts/parse_ncu_reports.py --input-dir $OUT_DIR --output profiles/ncu_metrics.csv"
echo
echo "Optional detailed run example:"
echo "NCU_EXTRA_ARGS=\"--section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats\" bash scripts/profile_ncu.sh"
