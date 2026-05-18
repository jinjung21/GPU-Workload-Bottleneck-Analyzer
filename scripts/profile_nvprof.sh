#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVPROF="${NVPROF:-$CUDA_HOME/bin/nvprof}"
OUT_DIR="${OUT_DIR:-profiles}"
OUT_CSV="$OUT_DIR/gpu_profile.csv"

mkdir -p "$OUT_DIR"

if [[ ! -x "$NVPROF" ]]; then
  echo "nvprof not found at $NVPROF" >&2
  exit 1
fi

if [[ ! -x build/vector_add || ! -x build/random_gather || ! -x build/matrix_mul_tiled || ! -x build/cublas_sgemm ]]; then
  echo "Benchmarks are missing. Run scripts/build_benchmarks.sh first." >&2
  exit 1
fi

echo "kernel_name,runtime_ms,flops,dram_read_bytes,dram_write_bytes,memory_access_pattern,notes" > "$OUT_CSV"

for benchmark in vector_add random_gather matrix_mul_tiled cublas_sgemm; do
  exe="build/$benchmark"
  log="$OUT_DIR/${benchmark}_nvprof.log"
  echo "Profiling $benchmark"
  "$exe" --csv >> "$OUT_CSV"
  "$NVPROF" --log-file "$log" "$exe" > /dev/null
done

echo "Wrote analyzer CSV: $OUT_CSV"
echo "Wrote nvprof logs: $OUT_DIR/*_nvprof.log"
echo
echo "Next command:"
echo "python3 main.py --input $OUT_CSV --output-dir outputs/gpu_profile_rtx2080ti --hardware-name \"RTX 2080 Ti\" --peak-flops 13450000000000 --peak-memory-bandwidth 616000000000"
