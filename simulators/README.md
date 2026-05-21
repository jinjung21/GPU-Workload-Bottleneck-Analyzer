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
notes
```

Example:

```csv
kernel_name,simulator,simulated_pim_time_ms,notes
vector_add,SAIT-PIMSimulator,0.050,prototype adapter output
saxpy,SAIT-PIMSimulator,0.060,prototype adapter output
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
PIM runtime for offloaded kernels and measured GPU runtime for non-offloaded
kernels.

## Candidate Backends

- SAITPublic/PIMSimulator: HBM2-PIM style simulator based on DRAMSim2.
- CMU-SAFARI/ramulator-pim: trace-driven Ramulator-based PIM infrastructure.
- UPMEM SDK simulator: useful for functional UPMEM-style PIM programming.

Initial integration should target one simple streaming workload first, such as
`vector_add` or `saxpy`, before attempting graph or sparse workloads.
