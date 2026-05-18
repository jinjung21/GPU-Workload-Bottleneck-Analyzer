# Profiles

Generated GPU profiling artifacts are written here by `scripts/profile_nvprof.sh`.

The generated CSV follows the analyzer schema:

```text
kernel_name,runtime_ms,flops,dram_read_bytes,dram_write_bytes,memory_access_pattern,notes
```

`*_nvprof.log` files are kept as raw profiler evidence for each benchmark run.

Generated profile artifacts are intentionally ignored by Git. Keep important
experimental results in a separate documented report or result summary before
sharing.
