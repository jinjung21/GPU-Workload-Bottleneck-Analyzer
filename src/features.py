from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import HardwareConfig


@dataclass(frozen=True)
class WorkloadFeatures:
    arithmetic_intensity: float
    memory_pressure: float
    compute_pressure: float
    bandwidth_pressure: float
    traffic_gb: float
    irregularity: float
    compute_complexity: float
    communication_intensity: float
    partitionability: float
    host_transfer_sensitivity: float


def extract_workload_features(
    profile_row: pd.Series,
    hardware: HardwareConfig,
    metadata_row: pd.Series | None = None,
) -> WorkloadFeatures:
    """Build numeric features for GPU/PIM cost modeling.

    Most features come directly from profile metrics. Paper metadata is used
    only as a temporary proxy for fields that require real profiler counters
    later, such as communication and partitionability.
    """

    intensity = float(profile_row["arithmetic_intensity"])
    roofline_ratio = intensity / hardware.ridge_point if hardware.ridge_point > 0 else 0.0
    achieved_flops = float(profile_row["achieved_flops"])
    achieved_bandwidth = float(profile_row["achieved_bandwidth"])
    dram_bytes = float(profile_row["dram_bytes"])
    access_pattern = str(profile_row["memory_access_pattern"]).lower()

    return WorkloadFeatures(
        arithmetic_intensity=intensity,
        memory_pressure=_clamp01(1.0 - min(roofline_ratio, 1.0)),
        compute_pressure=_clamp01(achieved_flops / hardware.peak_flops),
        bandwidth_pressure=_clamp01(achieved_bandwidth / hardware.peak_memory_bandwidth),
        traffic_gb=dram_bytes / 1e9,
        irregularity=1.0 if access_pattern == "irregular" else 0.0,
        compute_complexity=_metadata_level(metadata_row, "operation_complexity", {"low": 0.2, "medium": 0.55, "high": 1.0}, 0.5),
        communication_intensity=_metadata_level(
            metadata_row,
            "communication_intensity",
            {"low": 0.1, "medium": 0.5, "high": 1.0},
            0.4,
        ),
        partitionability=_metadata_level(metadata_row, "partitionability", {"low": 0.2, "medium": 0.6, "high": 1.0}, 0.6),
        host_transfer_sensitivity=_metadata_level(
            metadata_row,
            "host_transfer_sensitivity",
            {"low": 0.1, "medium": 0.5, "high": 1.0},
            0.5,
        ),
    )


def _metadata_level(
    metadata_row: pd.Series | None,
    column: str,
    mapping: dict[str, float],
    default: float,
) -> float:
    if metadata_row is None or column not in metadata_row:
        return default
    return mapping.get(str(metadata_row[column]).strip().lower(), default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
