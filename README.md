# GPU Workload Bottleneck Analyzer

Roofline 모델을 기반으로 GPU kernel profiling 데이터를 분석해 memory-bound, compute-bound, underutilized/latency-bound 병목을 분류하고, PIM/NMP에 적합한 workload 후보를 찾는 Python 기반 연구용 프로토타입입니다.

현재 버전은 두 가지 입력 경로를 지원합니다.

- Proxy/profile CSV 분석: CUDA 없이 macOS나 일반 Linux 환경에서 분석 모델과 report pipeline을 검증합니다.
- CUDA benchmark profiling: RTX 2080 Ti 서버에서 CUDA benchmark를 실행하고 실제 GPU runtime 기반 CSV를 생성합니다.

## Project Goal

이 프로젝트의 목적은 GPU workload의 병목을 정량화하고, PIM/NMP offloading 후보를 여러 baseline model과 비교 가능한 방식으로 선별하는 것입니다. 현재는 Roofline metric, PrIM-inspired proxy baseline, CUDA event runtime, `nvprof` log를 조합해 초기 end-to-end pipeline을 구성했습니다. 향후 Nsight Compute counter 권한이 확보되면 cache hit rate, occupancy, warp divergence, memory transaction metric까지 feature로 확장할 예정입니다.

## Roofline Model

Roofline 모델은 workload의 연산량과 메모리 이동량을 함께 사용해 성능 상한을 추정합니다.

```text
arithmetic_intensity = FLOPs / DRAM_bytes
attainable_performance = min(peak_flops, peak_memory_bandwidth * arithmetic_intensity)
ridge_point = peak_flops / peak_memory_bandwidth
```

`arithmetic_intensity`가 `ridge_point`보다 낮으면 메모리 대역폭의 영향을 크게 받는 memory-bound workload로 볼 수 있습니다. 반대로 높으면 연산 성능의 영향을 크게 받는 compute-bound workload로 볼 수 있습니다.

## Memory-Bound vs Compute-Bound

- Memory-bound: 연산기보다 메모리 대역폭 또는 데이터 이동이 병목입니다.
- Compute-bound: 메모리보다 연산 처리량이 병목입니다.
- Underutilized / latency-bound: Roofline 상한 대비 실제 성능이 낮아 latency, divergence, occupancy 문제가 의심됩니다.

## PIM/NMP Connection

PIM/NMP는 데이터 이동 비용이 큰 workload에서 유리할 수 있습니다. 이 도구는 낮은 arithmetic intensity, 큰 DRAM traffic, irregular memory access pattern을 기준으로 PIM/NMP 적합성을 rule-based heuristic으로 판단합니다.

## PIM/NMP Suitability Score

각 kernel은 0-100점의 PIM/NMP suitability score를 받습니다. 점수는 특정 kernel 이름이 아니라 Roofline 기반 feature를 조합해 계산합니다.

```text
score =
  low_arithmetic_intensity_component
  + dram_traffic_component
  + bandwidth_pressure_component
  + irregular_access_component
```

- Low arithmetic intensity: DRAM에서 데이터를 가져온 뒤 수행하는 연산량이 적을수록 높은 점수
- DRAM traffic: 읽고 쓰는 전체 DRAM byte가 클수록 높은 점수
- Bandwidth pressure: peak memory bandwidth 대비 실제 사용 bandwidth가 높을수록 높은 점수
- Irregular access: random gather, embedding lookup처럼 locality가 낮은 access pattern이면 추가 점수

점수 해석은 다음과 같습니다.

```text
85-100: strong NMP/PIM candidate
60-84 : PIM-friendly
35-59 : possible PIM candidate
0-34  : low PIM priority
```

## Analytical PIM/NMP Model Comparison

PrIM 2022 benchmark suite는 UPMEM 기반 real-world PIM architecture를 평가하기 위해 제안된 16개 memory-bound workload 모음입니다. 이 프로젝트는 PrIM workload metadata와 compute/communication-heavy negative control workload를 `paper_baselines/prim2022_workloads.csv`에 저장하고, 여러 후보 선별 모델을 비교합니다.

현재 비교는 실제 PIM speedup 재현이 아닙니다. UPMEM hardware가 없기 때문에, 지금 단계에서는 literature-labeled proxy workload를 사용해 다음을 검증합니다.

```text
PrIM이 PIM benchmark로 선정한 positive workload
negative control로 둔 compute/communication-heavy workload
vs.
각 모델이 예측한 PIM/NMP candidate 여부
```

비교 대상 모델은 다음과 같습니다.

```text
ai_only       : arithmetic intensity가 ridge point보다 낮으면 candidate
traffic_only  : DRAM traffic이 크면 candidate
heuristic_v1  : 기존 PIM/NMP score가 60점 이상이면 candidate
analytical_v2 : PIM memory/compute/host-transfer/sync time을 추정하고 risk gate 적용
```

`analytical_v2`, `feature_cost_v3`, `feature_cost_v4`는 다음 형태의 추정식을 사용합니다.

```text
estimated_pim_time =
  pim_memory_time
  + pim_compute_time
  + host_transfer_time
  + synchronization_time

estimated_speedup = gpu_proxy_runtime / estimated_pim_time
```

`feature_cost_v3`는 metadata label만으로 gate를 거는 대신, profile에서 얻은 numeric feature와 임시 paper metadata를 결합해 GPU/PIM cost를 비교합니다. `feature_cost_v4`는 여기에 data reuse proxy를 추가해서 cache/shared-memory reuse가 큰 dense compute kernel을 PIM 후보에서 더 강하게 제외합니다.

```text
memory_pressure
bandwidth_pressure
compute_pressure
irregularity
operation_complexity
communication_intensity
partitionability
host_transfer_sensitivity
data_reuse_potential
```

이 모델은 `SORT`처럼 낮은 arithmetic intensity와 irregular access만 보면 좋아 보이지만 communication/host-transfer risk가 큰 workload를 false positive로 분류하지 않도록 설계되었습니다.
`feature_cost_v4`는 `matrix_mul_tiled`처럼 bandwidth를 많이 쓰더라도 data reuse가 큰 GEMM 계열 workload를 naive PIM candidate로 보지 않도록 설계되었습니다.

The report also includes an end-to-end policy estimate:

```text
GPU-only total runtime
vs.
AI-only offload policy
vs.
traffic-only offload policy
vs.
heuristic_v1
vs.
feature_cost_v4
```

Each policy decides which kernels to offload. The runtime comparison then uses
the same `feature_cost_v4` PIM/NMP time estimate for all selected kernels, so
the table measures offload decision quality under a common cost model.

All model assumptions and temporary thresholds are documented in `docs/model_assumptions.md`.

PrIM proxy profile로 model comparison report를 생성하려면 다음 명령을 사용합니다.

```bash
python main.py \
  --input data/prim2022_proxy_profile.csv \
  --paper-baseline paper_baselines/prim2022_workloads.csv \
  --output-dir outputs/prim2022_model_v3
```

`data/prim2022_proxy_profile.csv`는 논문 수치를 복사한 measured profile이 아니라, PrIM workload category와 negative control workload를 현재 analyzer에 통과시키기 위한 qualitative proxy input입니다. 향후 Nsight Compute performance counter 권한이 확보되면 이 파일은 measured counter 기반 CSV로 대체해야 합니다.

## Current GPU Server Status

현재 확인한 GPU 서버 환경은 다음과 같습니다.

```text
OS: Ubuntu 20.04
GPU: NVIDIA GeForce RTX 2080 Ti
Driver: 535.230.02
CUDA runtime reported by driver: 12.2
CUDA toolkit: 11.0
Nsight Compute: 2020.1.1
Nsight Systems: 2020.3.2
Host compiler for nvcc: /usr/bin/g++-9
```

Nsight Compute는 설치되어 있지만 일반 사용자 계정에서는 performance counter 권한이 막혀 있습니다.

```text
ERR_NVGPUCTRPERM
```

따라서 현재 실제 GPU profiling path는 다음처럼 구성했습니다.

```text
CUDA event runtime + theoretical FLOPs/bytes + nvprof raw log
```

첫 실제 RTX 2080 Ti run에서는 `vector_add`, `random_gather`, `matrix_mul_tiled`, `cublas_sgemm`가 실행되었습니다. 이후 benchmark suite를 확장해 streaming, reduction, transpose, irregular access, dense GEMM 계열을 함께 비교합니다.

```text
vector_add        : memory-bound / bandwidth-bound
random_gather     : underutilized / latency-bound
matrix_mul_tiled  : memory-bound / bandwidth-bound in the current implementation
```

`matrix_mul_tiled`는 compute-bound baseline으로 기대했지만, 현재 구현과 problem size에서는 arithmetic intensity가 RTX 2080 Ti ridge point보다 낮게 나왔습니다. 따라서 cuBLAS SGEMM을 강한 compute-bound 기준 benchmark로 추가했고, `feature_cost_v4`는 high-reuse GEMM 계열을 naive PIM candidate에서 제외합니다.

## Setup

```bash
cd gpu-bottleneck-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

다른 profiling CSV나 hardware peak 값을 지정할 수도 있습니다.

```bash
python main.py \
  --input data/sample_profile.csv \
  --output-dir outputs \
  --hardware-name "Example GPU" \
  --peak-flops 15000000000000 \
  --peak-memory-bandwidth 900000000000
```

테스트는 다음처럼 실행합니다.

```bash
python -m pytest
```

## CUDA Benchmark Profiling

GPU 서버에서는 CUDA benchmark를 빌드하고 실제 GPU runtime 기반 CSV를 생성할 수 있습니다. 서버의 CUDA 11.0 toolkit은 GCC 9와 맞춰야 하므로, script는 기본적으로 `/usr/local/cuda/bin/nvcc`와 `/usr/bin/g++-9`를 사용합니다.

```bash
bash scripts/build_benchmarks.sh
bash scripts/profile_nvprof.sh
```

현재 benchmark suite는 다음 기준 workload를 포함합니다.

```text
vector_add        : streaming memory-bound
saxpy             : streaming vector update
random_gather     : irregular memory / latency-bound
reduction         : bandwidth-sensitive parallel primitive
matrix_transpose  : data-layout transformation dominated by memory movement
gemv              : matrix-vector multiplication; matches SAIT PIMSimulator GEMV primitive
matrix_mul_tiled  : initial GEMM baseline; not yet a strong compute-bound reference
cublas_sgemm      : optimized cuBLAS GEMM compute-throughput reference
```

`scripts/profile_nvprof.sh`는 다음을 생성합니다.

```text
profiles/gpu_profile.csv
profiles/vector_add_nvprof.log
profiles/saxpy_nvprof.log
profiles/random_gather_nvprof.log
profiles/reduction_nvprof.log
profiles/matrix_transpose_nvprof.log
profiles/gemv_nvprof.log
profiles/matrix_mul_tiled_nvprof.log
profiles/cublas_sgemm_nvprof.log
```

생성된 CSV는 바로 analyzer 입력으로 사용할 수 있습니다.

```bash
python3 main.py \
  --input profiles/gpu_profile.csv \
  --paper-baseline paper_baselines/gpu_benchmark_metadata.csv \
  --output-dir outputs/gpu_profile_rtx2080ti \
  --hardware-name "RTX 2080 Ti" \
  --peak-flops 13450000000000 \
  --peak-memory-bandwidth 616000000000
```

이 profiling path는 Nsight Compute performance counter 권한이 없는 서버에서도 동작하도록 설계되었습니다. CUDA event로 runtime을 측정하고, benchmark 코드에서 theoretical FLOPs/DRAM bytes를 함께 기록합니다. `nvprof` log는 raw profiler evidence로 저장됩니다.

## PIM Simulator Integration

실제 PIM hardware가 없어도, 오픈소스 PIM simulator 결과를 analyzer에 연결할 수 있습니다. 외부 simulator를 이 repository 안에 직접 vendoring하지 않고, simulator output을 공통 CSV schema로 변환해서 입력합니다.

Simulator CSV schema:

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

`simulated_pim_time_ms` can be provided directly, or computed from `simulated_pim_cycles * cycle_time_ns / 1e6`.

SAIT PIMSimulator logs can be converted with:

```bash
python3 scripts/parse_sait_pim_logs.py \
  --log-dir ~/pim-tools/pim-results \
  --output simulators/sait_pim_simulation.csv \
  --cycle-time-ns 1.0
```

예시:

```bash
python3 main.py \
  --input profiles/gpu_profile.csv \
  --paper-baseline paper_baselines/gpu_benchmark_metadata.csv \
  --pim-simulation simulators/sample_pim_simulation.csv \
  --output-dir outputs/gpu_profile_with_pim_sim \
  --hardware-name "RTX 2080 Ti" \
  --peak-flops 13450000000000 \
  --peak-memory-bandwidth 616000000000
```

`--pim-simulation`을 사용하면 end-to-end policy estimate에서 offload된 kernel은 simulated PIM runtime을 우선 사용합니다. Simulator coverage가 없는 kernel은 analytical estimate로 fallback하고, GPU에 남긴 kernel은 measured GPU runtime을 그대로 사용합니다.

## Cache Metrics

Cache behavior is important for distinguishing true DRAM bandwidth bottlenecks from cache locality or latency problems. The current server blocks Nsight Compute performance counters, so the analyzer cannot yet ingest L1/L2 hit rates, memory transactions, occupancy, or warp divergence.

Current handling:

```text
Available now:
- arithmetic intensity
- achieved bandwidth
- roofline utilization
- irregular vs regular access metadata
- CUDA event runtime

Not available yet:
- L1/L2 cache hit rate
- DRAM transaction count
- warp execution efficiency
- occupancy
- memory latency counters
```

Until those counters are available, `feature_cost_v4` uses `data_reuse_potential` as a proxy for cache/shared-memory reuse. When counter access is enabled, measured cache metrics should replace or calibrate this proxy feature.

## Outputs

실행 후 다음 파일이 생성됩니다.

```text
outputs/figures/roofline.png
outputs/reports/analysis_report.md
```

## Input CSV Schema

```text
kernel_name
runtime_ms
flops
dram_read_bytes
dram_write_bytes
memory_access_pattern
notes
```

## Current Limitations

- Nsight Compute performance counter 기반 metric은 아직 사용하지 않습니다.
- `PIM/NMP score`는 아직 calibration 전의 heuristic component를 포함합니다.
- `feature_cost_v3`/`feature_cost_v4`는 실제 PIM hardware 측정값이 아니라 analytical opportunity estimate입니다.
- `--pim-simulation` 결과는 simulator 기반 결과이며 실제 PIM silicon 측정값은 아닙니다.
- Roofline을 초과하는 측정값은 단위 오류 또는 hardware config 오류 가능성이 있으므로 report에 별도 표시합니다.
- 현재 CUDA benchmark의 FLOPs/DRAM bytes는 benchmark 구조에서 계산한 theoretical count입니다.
- 현재 CUDA benchmark suite는 작으며, 더 다양한 memory access pattern과 problem size sweep이 필요합니다.

## Future Work

- 더 다양한 실제 GPU benchmark metadata와 size sweep 추가
- Nsight Compute performance counter 권한 확보 후 CSV parser 추가
- cache hit rate, memory latency, occupancy, warp divergence metric 반영
- PIM/NMP model threshold와 risk parameter를 논문/실측 데이터 기반으로 calibration
- SAIT PIMSimulator 또는 Ramulator-PIM adapter 추가
- PrIM/UPMEM case study와 실제 RTX 2080 Ti profiling 결과 비교
