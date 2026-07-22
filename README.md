# GPU Workload Bottleneck Analyzer

A reproducible research prototype for diagnosing GPU kernel bottlenecks and identifying workloads that may benefit from Processing-in-Memory (PIM) or Near-Memory Processing (NMP).

The project connects measured CUDA runtime, Roofline analysis, NVIDIA Nsight Compute counters, a cache/stall-aware cost model, SAIT PIMSimulator results, and problem-size sweeps in one end-to-end pipeline.

> This is an architecture exploration and candidate-selection tool. It does not claim measured PIM silicon performance or production-grade prediction accuracy.

## Final Project Snapshot

| Item | Result |
|:--|:--|
| Target GPU | NVIDIA GeForce RTX 2080 Ti |
| CUDA workloads | 9 |
| Current policy model | `feature_cost_v6` |
| Automated tests | 31 passed |
| Final decisions | 7 PIM/NMP candidates, 2 GPU-resident workloads |
| Measured GPU-only runtime | 10.963 ms |
| Modeled v6 policy runtime | 4.638 ms |
| Modeled policy speedup | 2.36x |
| PIM simulator coverage | 4/9 workloads (44.4%) |
| Size sweep | Small, medium, and large inputs |

The 2.36x value is a policy-model estimate. Four selected workloads use simulator-reported speedup normalized to measured GPU runtime; three selected workloads fall back to the analytical v6 cost model.

## Portfolio Reports

- [English portfolio technical report](output/pdf/gpu_pim_bottleneck_analyzer_portfolio_report_en.pdf)
- [Korean portfolio technical report](output/pdf/gpu_pim_bottleneck_analyzer_portfolio_report.pdf)

Both reports document the project motivation, architecture, model evolution, experimental setup, results, limitations, and exact reproduction commands.

## System Pipeline

```text
CUDA event runtime + theoretical FLOPs/bytes
                  |
                  v
       Roofline bottleneck analysis
                  |
                  v
   Nsight Compute cache/stall evidence
                  |
                  v
      feature_cost_v6 offload policy
                  |
          +-------+-------+
          |               |
          v               v
 analytical PIM cost   SAIT PIMSimulator
          |               |
          +-------+-------+
                  |
                  v
 GPU-runtime-scaled end-to-end evaluation
                  |
                  v
       size sweep + figures + report
```

The implementation keeps measured, simulated, and estimated values separate so that every final result can be traced back to its evidence source.

## Research Questions

1. Can GPU kernels be separated into bandwidth-bound, compute-bound, and underutilized/latency-bound groups using measurable features?
2. Can a cache-, stall-, reuse-, and transfer-aware model produce more reasonable PIM/NMP decisions than arithmetic-intensity-only rules?
3. Can GPU measurements and PIM simulator results be combined without directly mixing unrelated clock domains?
4. Are the decisions stable across multiple problem sizes?

## Roofline Analysis

The first stage uses the classic Roofline model:

```text
arithmetic_intensity = FLOPs / DRAM_bytes
attainable_performance = min(
    peak_flops,
    peak_memory_bandwidth * arithmetic_intensity
)
ridge_point = peak_flops / peak_memory_bandwidth
```

For the RTX 2080 Ti configuration used in this project:

```text
peak_flops = 13.45e12 FLOP/s
peak_memory_bandwidth = 616e9 byte/s
ridge_point = 21.8344 FLOP/Byte
```

Interpretation:

- **Memory-bound / bandwidth-bound:** performance is close to the memory-bandwidth roof.
- **Compute-bound:** arithmetic intensity is above the ridge point and compute throughput is the primary limit.
- **Underutilized / latency-bound:** achieved performance is far below the applicable roof; cache misses, irregular access, dependencies, divergence, occupancy, or synchronization may be responsible.

Roofline position alone does not prove PIM suitability. A tiled GEMM may have low apparent arithmetic intensity while still benefiting heavily from L2/shared-memory reuse. An irregular gather may fail to saturate DRAM bandwidth because memory latency, rather than bandwidth, is the dominant problem.

## Model Evolution

| Model | Main addition | Problem addressed |
|:--|:--|:--|
| `heuristic_v1` | AI, traffic, bandwidth pressure, irregularity score | Initial interpretable baseline |
| `analytical_v2` | PIM memory, compute, transfer, and synchronization cost | Converts suitability into a timing opportunity |
| `feature_cost_v3` | Numeric GPU/PIM features and risk gates | Reduces false positives from low AI alone |
| `feature_cost_v4` | Data-reuse proxy and dense-compute gate | Keeps high-reuse GEMM workloads on the GPU |
| `feature_cost_v5` | Basic Nsight Compute DRAM/SM/cache pressure | Replaces some metadata proxies with measured counters |
| `feature_cost_v6` | Cache hit, scheduler supply, stalls, control risk, counter coverage | Distinguishes locality, bandwidth, and latency causes |

### `feature_cost_v6`

The final model can use the following optional Nsight Compute features:

```text
ncu_sol_dram_pct
ncu_sol_l1_tex_pct
ncu_sol_l2_pct
ncu_sm_util_pct
ncu_achieved_occupancy_pct
ncu_l1_hit_rate_pct
ncu_l2_hit_rate_pct
ncu_global_load_efficiency_pct
ncu_global_store_efficiency_pct
ncu_warp_execution_efficiency_pct
ncu_branch_efficiency_pct
ncu_memory_stall_pct
ncu_long_scoreboard_stall_pct
ncu_short_scoreboard_stall_pct
ncu_barrier_stall_pct
ncu_eligible_warps_per_scheduler
ncu_registers_per_thread
ncu_l2_read_transactions
ncu_l2_write_transactions
```

Missing counters remain `NA`; they are not interpreted as measured zeros. The model computes `ncu_feature_coverage` and scales its v6 correction by the available-counter ratio. This prevents a partial profile from producing an overconfident adjustment.

The final RTX 2080 Ti profile contains the most useful cache, scheduler, and long-scoreboard signals, but Nsight Compute 2020.1.1 does not provide every modern metric. The report therefore marks the run as partial NCU coverage.

## Workload Suite

| Workload | Intended behavior | Final role |
|:--|:--|:--|
| `vector_add` | Streaming memory bandwidth | PIM/NMP candidate |
| `saxpy` | Streaming vector update | PIM/NMP candidate |
| `random_gather` | Irregular memory latency | PIM/NMP candidate |
| `reduction` | Bandwidth-sensitive collective | PIM/NMP candidate |
| `scan` | Prefix-sum collective with synchronization | PIM/NMP candidate |
| `matrix_transpose` | Memory-dominated layout transformation | PIM/NMP candidate |
| `gemv` | Low-reuse matrix-vector multiplication | PIM-positive control |
| `matrix_mul_tiled` | Reuse-aware custom GEMM | GPU negative control |
| `cublas_sgemm` | Optimized dense compute | GPU negative control |

Representative measured observations:

- `vector_add` and `saxpy`: approximately 555 GB/s and 90% of the modeled Roofline memory limit.
- `random_gather`: approximately 38.8 GB/s and 6.3% Roofline utilization, with very high long-scoreboard stall.
- `matrix_mul_tiled`: approximately 98.34% L2 hit rate, indicating strong GPU locality despite low Roofline arithmetic intensity.
- `cublas_sgemm`: arithmetic intensity around 341 FLOP/Byte and approximately 10.85 TFLOP/s, providing a strong compute-bound control.

## Final Decisions

```text
PIM/NMP exploration candidates:
  vector_add
  saxpy
  random_gather
  reduction
  scan
  matrix_transpose
  gemv

Remain on GPU:
  matrix_mul_tiled
  cublas_sgemm
```

The GEMV positive control and both GEMM negative controls pass their expected calibration roles.

## PIM Simulator Integration

External simulator results use a simulator-agnostic CSV schema:

```text
kernel_name
simulator
simulated_pim_time_ms
simulated_pim_cycles
simulated_baseline_cycles
simulated_speedup
cycle_time_ns
notes
```

SAIT PIMSimulator logs are converted with:

```bash
python3 scripts/parse_sait_pim_logs.py \
  --log-dir ~/pim-tools/pim-results \
  --output simulators/sait_pim_simulation.csv \
  --cycle-time-ns 1.0
```

### Clock-domain normalization

Simulator cycles and GPU milliseconds must not be directly added because they belong to different architectures and timing domains.

The final implementation preserves raw simulator timing for traceability:

```text
raw_simulator_time_ms = simulated_pim_cycles * cycle_time_ns / 1e6
```

For cross-domain end-to-end evaluation, it uses the simulator's internal baseline-to-PIM ratio:

```text
simulated_scaled_pim_time_ms = measured_gpu_runtime_ms / simulated_speedup
```

If no simulator mapping exists for a selected workload, the evaluation falls back to the common `feature_cost_v6` analytical PIM estimate and reports the fallback count.

## Evidence Tiers

The report uses evidence tiers rather than statistical confidence intervals:

| Tier | Evidence |
|:--|:--|
| A | Measured GPU runtime + NCU counters + mapped PIM simulator result |
| B | Measured GPU runtime + NCU counters + analytical PIM estimate |
| C | Measured GPU runtime + analytical PIM estimate |

Tier A is the strongest evidence available in this project, but it is still simulator-backed rather than measured on PIM silicon.

## Problem-Size Sweep

The size sweep checks whether decisions are stable beyond a single input size.

```text
small  : vector_n=1,048,576   matrix_n=512
medium : vector_n=4,194,304   matrix_n=1024
large  : vector_n=16,777,216  matrix_n=1536
```

Run it with:

```bash
bash scripts/profile_size_sweep.sh
```

Generated artifacts:

```text
profiles/size_sweep/gpu_profile_small.csv
profiles/size_sweep/gpu_profile_medium.csv
profiles/size_sweep/gpu_profile_large.csv
outputs/size_sweep/{small,medium,large}/reports/analysis_report.md
outputs/size_sweep/size_sweep_summary.csv
outputs/size_sweep/size_sweep_summary.md
outputs/size_sweep/size_sweep.png
```

The current suite preserves the same 7 PIM/NMP to 2 GPU decision split at all three sizes. This demonstrates internal size stability, not generalization to unseen applications.

## Repository Structure

```text
benchmarks/          CUDA microbenchmarks and argument parsing
data/                Sample and proxy profiles
docs/                Model assumptions and research notes
output/pdf/          English and Korean portfolio reports
paper_baselines/     PrIM-inspired and local workload metadata
profiles/            GPU and Nsight Compute profile inputs
scripts/             Build, profile, parse, sweep, fetch, and report tools
simulators/          Simulator adapter CSV inputs
src/                 Analysis, models, plotting, and reporting modules
tests/               Automated unit and integration-style tests
main.py              End-to-end analyzer entry point
```

## Local Analysis Setup

CUDA is not required to analyze an existing profile CSV.

```bash
git clone https://github.com/jinjung21/GPU-Workload-Bottleneck-Analyzer.git
cd GPU-Workload-Bottleneck-Analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
python3 main.py
```

After profiling on the GPU server or fetching the generated profile files, run the complete RTX 2080 Ti analysis with:

```bash
python3 main.py \
  --input profiles/gpu_profile.csv \
  --paper-baseline paper_baselines/gpu_benchmark_metadata.csv \
  --pim-simulation simulators/sait_pim_simulation.csv \
  --ncu-metrics profiles/ncu_metrics.csv \
  --output-dir outputs/gpu_profile_with_sait_pim_ncu \
  --hardware-name "RTX 2080 Ti" \
  --peak-flops 13450000000000 \
  --peak-memory-bandwidth 616000000000
```

## Complete GPU Server Run

The validated server environment is:

```text
OS: Ubuntu 20.04
GPU: NVIDIA GeForce RTX 2080 Ti
Driver: 535.230.02
CUDA toolkit: 11.0
Nsight Compute: 2020.1.1
Host compiler for nvcc: /usr/bin/g++-9
```

Run the following commands from the server repository:

```bash
cd ~/GPU-Workload-Bottleneck-Analyzer
git pull
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
bash scripts/build_benchmarks.sh
bash scripts/profile_nvprof.sh
NCU_EXTRA_ARGS="--section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats" bash scripts/profile_ncu.sh
python3 scripts/parse_ncu_reports.py --input-dir profiles/ncu --output profiles/ncu_metrics.csv
python3 scripts/parse_sait_pim_logs.py --log-dir ~/pim-tools/pim-results --output simulators/sait_pim_simulation.csv --cycle-time-ns 1.0
python3 main.py --input profiles/gpu_profile.csv --paper-baseline paper_baselines/gpu_benchmark_metadata.csv --pim-simulation simulators/sait_pim_simulation.csv --ncu-metrics profiles/ncu_metrics.csv --output-dir outputs/gpu_profile_with_sait_pim_ncu --hardware-name "RTX 2080 Ti" --peak-flops 13450000000000 --peak-memory-bandwidth 616000000000
bash scripts/profile_size_sweep.sh
```

`profile_ncu.sh` skips one warm-up launch and profiles one representative launch by default. This avoids profiling every repeated benchmark iteration.

If a section name is unsupported by the installed Nsight Compute version, first run the basic command:

```bash
bash scripts/profile_ncu.sh
```

Available metrics can be inspected with:

```bash
bash scripts/query_ncu_metrics.sh
```

## Fetch Server Results to macOS

From the local repository:

```bash
cd /Users/kingjung/Desktop/gpu-bottleneck-analyzer
git pull
bash scripts/fetch_server_results.sh gpu_profile_with_sait_pim_ncu
open outputs/gpu_profile_with_sait_pim_ncu/reports/analysis_report.md
open outputs/size_sweep/size_sweep_summary.md
open outputs/size_sweep/size_sweep.png
open output/pdf/gpu_pim_bottleneck_analyzer_portfolio_report_en.pdf
```

The `outputs/` directory is intentionally ignored by Git because profiler runs can accumulate many generated files. Keep important runs as experimental evidence, and archive or remove old runs only when storage becomes an issue.

## Output Files

Each complete analyzer run generates:

```text
outputs/<run_name>/figures/roofline.png
outputs/<run_name>/figures/model_comparison.png
outputs/<run_name>/figures/end_to_end.png
outputs/<run_name>/reports/analysis_report.md
```

## Input Profile Schema

```text
kernel_name
runtime_ms
flops
dram_read_bytes
dram_write_bytes
memory_access_pattern
notes
```

`runtime_ms` is measured with CUDA events in the GPU benchmark workflow. FLOPs and DRAM-byte counts are theoretical counts derived from each benchmark implementation.

## Testing

```bash
python3 -m pytest
```

Current status:

```text
31 passed
```

Tests cover CSV parsing, Roofline analysis, classification, model comparison, NCU parsing, simulator attachment, clock-domain normalization, analytical fallback behavior, calibration controls, and end-to-end policy evaluation.

## Interpretation of Model Metrics

Precision, recall, F1, and accuracy are computed on the same nine labeled workloads used to guide model design. They measure **calibration-set alignment**, not held-out generalization accuracy.

A publishable accuracy claim would require:

- Frozen model thresholds before evaluation
- Unseen application workloads
- Additional GPU architectures
- Broader NCU counter coverage
- Preferably measurements on a real PIM platform

## Current Limitations

- No runtime has been measured on physical PIM silicon.
- The workload suite contains nine synthetic or microbenchmark-style kernels.
- SAIT PIMSimulator mappings cover four of the nine workloads.
- Analytical PIM bandwidth, throughput, transfer, and synchronization parameters remain calibration assumptions.
- Nsight Compute 2020.1.1 provides only partial v6 counter coverage.
- Theoretical FLOP and DRAM-byte counts may differ from physical transactions.
- End-to-end estimates do not fully model data placement, orchestration, programming, and deployment overhead.
- Size stability does not replace held-out application validation.

## Recommended Next Research Step

The most reasonable next step is not another feature-heavy model version. The current thresholds should be frozen and evaluated on unseen application-level workloads, then repeated on another GPU architecture. Real UPMEM or other PIM hardware measurements should be added before making hardware-level speedup claims.

## References

- Samuel Williams, Andrew Waterman, and David Patterson, ["Roofline: An Insightful Visual Performance Model for Multicore Architectures"](https://doi.org/10.1145/1498765.1498785), CACM, 2009.
- NVIDIA, [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/2020.3/ProfilingGuide/index.html).
- Juan Gómez-Luna et al., ["Benchmarking a New Paradigm: An Experimental Analysis of a Real Processing-in-Memory Architecture"](https://arxiv.org/abs/2105.03814), IEEE Access, 2022.
- Samsung Advanced Institute of Technology, [SAITPublic/PIMSimulator](https://github.com/SAITPublic/PIMSimulator).
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/).

## License and Use

This repository is intended for research, education, portfolio review, and architecture exploration. Validate model assumptions against the target hardware and workload before using the output for engineering decisions.
