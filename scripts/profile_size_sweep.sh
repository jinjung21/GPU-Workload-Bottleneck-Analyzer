#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="${PROFILE_DIR:-profiles/size_sweep}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/size_sweep}"
PAPER_BASELINE="${PAPER_BASELINE:-paper_baselines/gpu_benchmark_metadata.csv}"
PIM_SIMULATION="${PIM_SIMULATION:-simulators/sait_pim_simulation.csv}"
HARDWARE_NAME="${HARDWARE_NAME:-RTX 2080 Ti}"
PEAK_FLOPS="${PEAK_FLOPS:-13450000000000}"
PEAK_MEMORY_BANDWIDTH="${PEAK_MEMORY_BANDWIDTH:-616000000000}"

mkdir -p "$PROFILE_DIR" "$OUTPUT_DIR"

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

scales=(
  "small:1048576:512"
  "medium:4194304:1024"
  "large:16777216:1536"
)

for spec in "${scales[@]}"; do
  IFS=":" read -r label vector_n matrix_n <<< "$spec"
  out_csv="$PROFILE_DIR/gpu_profile_${label}.csv"
  echo "kernel_name,runtime_ms,flops,dram_read_bytes,dram_write_bytes,memory_access_pattern,notes" > "$out_csv"

  echo "Profiling size sweep scale=$label vector_n=$vector_n matrix_n=$matrix_n"
  for benchmark in "${benchmarks[@]}"; do
    exe="build/$benchmark"
    case "$benchmark" in
      vector_add|saxpy|random_gather|reduction|scan)
        "$exe" --csv --n "$vector_n" >> "$out_csv"
        ;;
      *)
        "$exe" --csv --n "$matrix_n" >> "$out_csv"
        ;;
    esac
  done

  main_args=(
    python3 main.py
    --input "$out_csv"
    --paper-baseline "$PAPER_BASELINE"
    --output-dir "$OUTPUT_DIR/$label"
    --hardware-name "$HARDWARE_NAME"
    --peak-flops "$PEAK_FLOPS"
    --peak-memory-bandwidth "$PEAK_MEMORY_BANDWIDTH"
  )
  if [[ -f "$PIM_SIMULATION" ]]; then
    main_args+=(--pim-simulation "$PIM_SIMULATION")
  fi
  "${main_args[@]}"
done

python3 scripts/summarize_size_sweep.py \
  --input-dir "$PROFILE_DIR" \
  --paper-baseline "$PAPER_BASELINE" \
  --output-csv "$OUTPUT_DIR/size_sweep_summary.csv" \
  --output-markdown "$OUTPUT_DIR/size_sweep_summary.md" \
  --output-plot "$OUTPUT_DIR/size_sweep.png" \
  --hardware-name "$HARDWARE_NAME" \
  --peak-flops "$PEAK_FLOPS" \
  --peak-memory-bandwidth "$PEAK_MEMORY_BANDWIDTH"

echo "Wrote size sweep profiles: $PROFILE_DIR/gpu_profile_*.csv"
echo "Wrote size sweep reports: $OUTPUT_DIR/{small,medium,large}/reports/analysis_report.md"
echo "Wrote size sweep summary: $OUTPUT_DIR/size_sweep_summary.md"
echo "Wrote size sweep plot: $OUTPUT_DIR/size_sweep.png"
