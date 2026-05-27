# PIM Simulator Integration

This directory is for simulator adapters and simulator output artifacts.

The project does not vendor or install external PIM simulators. On lab servers,
clone/build external simulators only in a separate user-owned directory after
confirming with the server owner or senior lab member.

## Supported Adapter Schema

Convert simulator output into this CSV schema before passing it to `main.py`:

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

`simulated_pim_time_ms` can be provided directly. If it is omitted, the analyzer
computes it from `simulated_pim_cycles * cycle_time_ns / 1e6`.

Example:

```csv
kernel_name,simulator,simulated_pim_cycles,simulated_baseline_cycles,simulated_speedup,cycle_time_ns,notes
vector_add,SAIT-PIMSimulator,3349,6651,1.98597,1.0,PIMBenchFixture.add
gemv,SAIT-PIMSimulator,13166,36082,2.74054,1.0,PIMBenchFixture.gemv
```

SAIT PIMSimulator logs can be converted with:

```bash
python3 scripts/parse_sait_pim_logs.py \
  --log-dir ~/pim-tools/pim-results \
  --output simulators/sait_pim_simulation.csv \
  --cycle-time-ns 1.0
```

Run the analyzer with simulated PIM timing:

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

When `--pim-simulation` is provided, the end-to-end policy table uses simulated
PIM runtime where available, falls back to analytical estimates for uncovered
kernels, and keeps measured GPU runtime for non-offloaded kernels.

## Candidate Backends

- SAITPublic/PIMSimulator: HBM2-PIM style simulator based on DRAMSim2.
- CMU-SAFARI/ramulator-pim: trace-driven Ramulator-based PIM infrastructure.
- UPMEM SDK simulator: useful for functional UPMEM-style PIM programming.

Initial integration should target simple matched primitives first, such as
`vector_add`/`saxpy` through ADD and `gemv` through GEMV, before attempting graph
or sparse workloads.
