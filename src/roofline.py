import pandas as pd

from .config import HardwareConfig


def add_roofline_metrics(profile: pd.DataFrame, hardware: HardwareConfig) -> pd.DataFrame:
    """Add Roofline-derived metrics to the profile dataframe."""

    result = profile.copy()
    result["runtime_s"] = result["runtime_ms"] / 1_000.0
    result["dram_bytes"] = result["dram_read_bytes"] + result["dram_write_bytes"]
    result["arithmetic_intensity"] = result["flops"] / result["dram_bytes"]
    result["achieved_flops"] = result["flops"] / result["runtime_s"]
    result["achieved_bandwidth"] = result["dram_bytes"] / result["runtime_s"]
    result["attainable_performance"] = result["arithmetic_intensity"].map(
        lambda intensity: min(
            hardware.peak_flops,
            hardware.peak_memory_bandwidth * intensity,
        )
    )
    result["roofline_utilization_raw"] = result.apply(
        lambda row: _safe_div(row["achieved_flops"], row["attainable_performance"]),
        axis=1,
    )
    result["exceeds_roofline"] = result["roofline_utilization_raw"] > 1.0
    result["roofline_utilization"] = result["roofline_utilization_raw"].clip(upper=1.0)
    result["ridge_point"] = hardware.ridge_point
    return result


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator
