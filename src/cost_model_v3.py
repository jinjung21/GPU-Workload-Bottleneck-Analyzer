from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import HardwareConfig
from .features import WorkloadFeatures, extract_workload_features


@dataclass(frozen=True)
class FeatureCostModelConfig:
    name: str = "feature_cost_v3"
    pim_memory_bandwidth: float = 4.0e12
    pim_scalar_throughput: float = 1.0e12
    pim_vector_throughput: float = 8.0e12
    host_pim_bandwidth: float = 512e9
    candidate_speedup_threshold: float = 1.10
    min_memory_pressure: float = 0.25


def estimate_feature_cost_v3(
    profile_row: pd.Series,
    hardware: HardwareConfig,
    metadata_row: pd.Series | None = None,
    config: FeatureCostModelConfig | None = None,
) -> dict[str, float | str | bool]:
    """Estimate GPU-vs-PIM opportunity using numeric workload features."""

    cfg = config or FeatureCostModelConfig()
    features = extract_workload_features(profile_row, hardware, metadata_row)
    runtime_s = float(profile_row["runtime_s"])
    flops = float(profile_row["flops"])
    dram_bytes = float(profile_row["dram_bytes"])

    gpu_compute_time = flops / hardware.peak_flops
    gpu_memory_time = dram_bytes / hardware.peak_memory_bandwidth
    gpu_latency_penalty = runtime_s * (0.03 + 0.12 * features.irregularity + 0.05 * features.communication_intensity)
    predicted_gpu_time = max(gpu_compute_time, gpu_memory_time) + gpu_latency_penalty

    locality_factor = _locality_factor(features)
    pim_memory_time = dram_bytes * locality_factor / cfg.pim_memory_bandwidth
    pim_compute_time = flops / _pim_throughput(features, cfg)
    host_transfer_time = dram_bytes * _host_transfer_factor(features) / cfg.host_pim_bandwidth
    communication_time = runtime_s * _communication_factor(features)
    predicted_pim_time = pim_memory_time + pim_compute_time + host_transfer_time + communication_time

    speedup = predicted_gpu_time / predicted_pim_time if predicted_pim_time > 0 else 0.0
    risk_score = _risk_score(features)
    predicted_candidate = (
        speedup >= cfg.candidate_speedup_threshold
        and features.memory_pressure >= cfg.min_memory_pressure
        and risk_score < 0.75
    )

    return {
        "predicted_gpu_time_ms": predicted_gpu_time * 1000.0,
        "estimated_pim_time_ms": predicted_pim_time * 1000.0,
        "estimated_speedup": speedup,
        "predicted_candidate": predicted_candidate,
        "risk_score": risk_score,
        "risk": _risk_summary(features),
        "feature_summary": _feature_summary(features),
    }


def _pim_throughput(features: WorkloadFeatures, config: FeatureCostModelConfig) -> float:
    blend = features.compute_complexity
    return config.pim_scalar_throughput * (1.0 - blend) + config.pim_vector_throughput * blend


def _locality_factor(features: WorkloadFeatures) -> float:
    base = 0.18 + 0.22 * features.host_transfer_sensitivity
    irregular_penalty = 0.12 * features.irregularity
    communication_penalty = 0.10 * features.communication_intensity
    return min(0.90, base + irregular_penalty + communication_penalty)


def _host_transfer_factor(features: WorkloadFeatures) -> float:
    return min(0.35, 0.01 + 0.18 * features.host_transfer_sensitivity + 0.05 * features.communication_intensity)


def _communication_factor(features: WorkloadFeatures) -> float:
    partition_penalty = 1.0 - features.partitionability
    return 0.01 + 0.10 * features.communication_intensity + 0.08 * partition_penalty


def _risk_score(features: WorkloadFeatures) -> float:
    return min(
        1.0,
        0.35 * features.compute_complexity
        + 0.30 * features.communication_intensity
        + 0.20 * features.host_transfer_sensitivity
        + 0.15 * (1.0 - features.partitionability),
    )


def _risk_summary(features: WorkloadFeatures) -> str:
    risks = []
    if features.compute_complexity >= 0.85:
        risks.append("compute-heavy")
    if features.communication_intensity >= 0.85:
        risks.append("high communication")
    if features.host_transfer_sensitivity >= 0.85:
        risks.append("host transfer sensitive")
    if features.partitionability <= 0.3:
        risks.append("poor partitionability")
    return ", ".join(risks) if risks else "low model risk"


def _feature_summary(features: WorkloadFeatures) -> str:
    return (
        f"memory_pressure={features.memory_pressure:.2f}, "
        f"bandwidth_pressure={features.bandwidth_pressure:.2f}, "
        f"irregularity={features.irregularity:.2f}, "
        f"risk={_risk_score(features):.2f}"
    )
