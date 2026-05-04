from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "kernel_name",
    "runtime_ms",
    "flops",
    "dram_read_bytes",
    "dram_write_bytes",
    "memory_access_pattern",
    "notes",
}


def load_profile_csv(path: str | Path) -> pd.DataFrame:
    """Load profiling-style CSV data and validate the expected columns."""

    csv_path = Path(path)
    profile = pd.read_csv(csv_path)
    if profile.empty:
        raise ValueError(f"Profile CSV is empty: {csv_path}")

    missing = REQUIRED_COLUMNS - set(profile.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {csv_path}: {missing_text}")

    numeric_columns = ["runtime_ms", "flops", "dram_read_bytes", "dram_write_bytes"]
    for column in numeric_columns:
        profile[column] = pd.to_numeric(profile[column], errors="raise")

    if (profile["runtime_ms"] <= 0).any():
        raise ValueError("runtime_ms must be positive for all kernels")

    if (profile["flops"] < 0).any():
        raise ValueError("flops must be non-negative for all kernels")

    byte_columns = ["dram_read_bytes", "dram_write_bytes"]
    if (profile[byte_columns] < 0).any().any():
        raise ValueError("DRAM byte columns must be non-negative for all kernels")

    byte_total = profile["dram_read_bytes"] + profile["dram_write_bytes"]
    if (byte_total <= 0).any():
        raise ValueError("Total DRAM bytes must be positive for all kernels")

    profile["memory_access_pattern"] = profile["memory_access_pattern"].str.lower().str.strip()
    return profile
