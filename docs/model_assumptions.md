# Model Assumptions and Thresholds

This document separates model components with strong external basis from
temporary project assumptions that must be calibrated with more measurements.

## Hardware Roofline Boundary

The primary memory-bound vs compute-bound boundary follows the classic Roofline
model:

```text
attainable_performance = min(peak_flops, peak_memory_bandwidth * arithmetic_intensity)
ridge_point = peak_flops / peak_memory_bandwidth
```

Interpretation:

```text
arithmetic_intensity < ridge_point  -> memory-bandwidth region
arithmetic_intensity >= ridge_point -> compute-throughput region
```

This is not an arbitrary project threshold. It is the intersection between the
sloped memory-bandwidth roof and the flat compute-throughput roof. NVIDIA
Nsight Compute's Roofline documentation uses the same concept and refers to the
ridge point as the partition between the memory-bound and compute-bound regions.

For the RTX 2080 Ti configuration used in this project:

```text
peak_flops = 13.45e12 FLOP/s
peak_memory_bandwidth = 616e9 byte/s
ridge_point = 21.8344 FLOP/byte
```

## What the Ridge Point Does Not Prove

The ridge point is a first-order performance bound, not a complete diagnosis.

A kernel with low arithmetic intensity may still fail to saturate DRAM bandwidth
because of:

```text
irregular memory access
poor coalescing
cache misses
memory latency
branch divergence
low occupancy
synchronization
```

This project therefore keeps a separate `underutilized / latency-bound`
classification when Roofline utilization is low. In future Nsight Compute
counter-based profiling, this should be refined using metrics for occupancy,
cache behavior, warp execution efficiency, and DRAM/L2 throughput.

## PIM/NMP Candidate Thresholds

The current PIM/NMP thresholds are project assumptions, not final scientific
constants:

```text
heuristic_v1 candidate threshold: PIM/NMP score >= 60
analytical_v2 speedup threshold: estimated speedup >= 1.10
feature_cost_v3 speedup threshold: estimated speedup >= 1.10
feature_cost_v3 memory pressure floor: memory_pressure >= 0.25
feature_cost_v3 risk cutoff: risk_score < 0.75
```

These thresholds are intentionally documented as calibration targets. They are
used to compare model behavior on proxy workloads, but should be retuned when
more measured GPU profiles or PIM/NMP reference data become available.

## Current Evidence Levels

| Component | Current basis | Confidence |
|:--|:--|:--|
| Ridge point boundary | Roofline model / hardware specs | High |
| RTX 2080 Ti peak FLOP/s and bandwidth | Published hardware specifications | Medium-high |
| CUDA benchmark runtime | Measured on RTX 2080 Ti with CUDA events | High |
| Benchmark FLOPs/DRAM bytes | Theoretical count from benchmark structure | Medium |
| PIM/NMP score thresholds | Project heuristic | Low-medium |
| analytical_v2 / feature_cost_v3 speedup | Analytical opportunity estimate | Low-medium |
| PrIM workload labels | Literature-inspired workload categories | Medium |

## Key References

- Williams, Waterman, and Patterson, "Roofline: An Insightful Visual
  Performance Model for Multicore Architectures", Communications of the ACM,
  2009.
- NVIDIA Nsight Compute Profiling Guide, Roofline chart and memory/compute
  workload analysis sections.
- NVIDIA technical blog, "Accelerating HPC Applications with NVIDIA Nsight
  Compute Roofline Analysis".
- PrIM / UPMEM benchmark work from CMU SAFARI for processing-in-memory workload
  characterization.
