# GPU Workload Bottleneck Analyzer

Roofline 모델을 기반으로 GPU kernel profiling 데이터를 분석해 memory-bound, compute-bound, underutilized/latency-bound 병목을 분류하고, PIM/NMP에 적합한 workload 후보를 찾는 Python 기반 연구용 프로토타입입니다.

현재 버전은 macOS에서 실행 가능하도록 fake profiling CSV만 사용합니다. CUDA, Nsight Compute, NVIDIA GPU는 필요하지 않습니다.

## Project Goal

이 프로젝트의 목적은 실제 GPU profiler가 없는 환경에서도 GPU workload 병목 분석 파이프라인을 먼저 설계하고 구현하는 것입니다. 나중에 Linux GPU 서버를 사용할 수 있게 되면 Nsight Compute CSV를 입력으로 받아 같은 분석 엔진을 재사용하도록 확장할 수 있습니다.

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

현재 비교는 실제 PIM speedup 재현이 아닙니다. 아직 UPMEM hardware나 GPU profiling 결과가 없기 때문에, 지금 단계에서는 literature-labeled proxy workload를 사용해 다음을 검증합니다.

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

`analytical_v2`와 `feature_cost_v3`는 다음 형태의 추정식을 사용합니다.

```text
estimated_pim_time =
  pim_memory_time
  + pim_compute_time
  + host_transfer_time
  + synchronization_time

estimated_speedup = gpu_proxy_runtime / estimated_pim_time
```

`feature_cost_v3`는 metadata label만으로 gate를 거는 대신, profile에서 얻은 numeric feature와 임시 paper metadata를 결합해 GPU/PIM cost를 비교합니다.

```text
memory_pressure
bandwidth_pressure
compute_pressure
irregularity
operation_complexity
communication_intensity
partitionability
host_transfer_sensitivity
```

이 모델은 `SORT`처럼 낮은 arithmetic intensity와 irregular access만 보면 좋아 보이지만 communication/host-transfer risk가 큰 workload를 false positive로 분류하지 않도록 설계되었습니다.

PrIM proxy profile로 model comparison report를 생성하려면 다음 명령을 사용합니다.

```bash
python main.py \
  --input data/prim2022_proxy_profile.csv \
  --paper-baseline paper_baselines/prim2022_workloads.csv \
  --output-dir outputs/prim2022_model_v3
```

`data/prim2022_proxy_profile.csv`는 논문 수치를 복사한 measured profile이 아니라, PrIM workload category와 negative control workload를 현재 analyzer에 통과시키기 위한 qualitative proxy input입니다. 실제 GPU 서버를 사용할 수 있게 되면 이 파일은 Nsight Compute로 측정한 CSV로 대체해야 합니다.

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

현재 benchmark suite는 세 개의 기준 workload로 시작합니다.

```text
vector_add        : streaming memory-bound
random_gather     : irregular memory-bound
matrix_mul_tiled  : compute-bound
```

`scripts/profile_nvprof.sh`는 다음을 생성합니다.

```text
profiles/gpu_profile.csv
profiles/vector_add_nvprof.log
profiles/random_gather_nvprof.log
profiles/matrix_mul_tiled_nvprof.log
```

생성된 CSV는 바로 analyzer 입력으로 사용할 수 있습니다.

```bash
python3 main.py \
  --input profiles/gpu_profile.csv \
  --output-dir outputs/gpu_profile_rtx2080ti \
  --hardware-name "RTX 2080 Ti" \
  --peak-flops 13450000000000 \
  --peak-memory-bandwidth 616000000000
```

이 profiling path는 Nsight Compute performance counter 권한이 없는 서버에서도 동작하도록 설계되었습니다. CUDA event로 runtime을 측정하고, benchmark 코드에서 theoretical FLOPs/DRAM bytes를 함께 기록합니다. `nvprof` log는 raw profiler evidence로 저장됩니다.

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
- PIM/NMP 판단은 논문 기반 정량 모델이 아니라 초기 rule-based heuristic입니다.
- Roofline을 초과하는 측정값은 단위 오류 또는 hardware config 오류 가능성이 있으므로 report에 별도 표시합니다.
- 현재 CUDA benchmark의 FLOPs/DRAM bytes는 benchmark 구조에서 계산한 theoretical count입니다.

## Future Work

- Nsight Compute CSV parser 추가
- CUDA benchmark 추가
- Linux GPU 서버에서 실제 profiling 결과 수집
- cache hit rate, memory latency, occupancy, warp divergence metric 반영
- PIM/NMP 관련 논문 case study와 비교 검증
- 이력서용 프로젝트로 발전시키려면 synthetic benchmark suite, Nsight Compute parser, 논문 baseline 재현표를 우선 추가
