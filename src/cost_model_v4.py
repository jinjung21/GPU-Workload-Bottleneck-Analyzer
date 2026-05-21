from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import HardwareConfig
from .features import WorkloadFeatures, extract_workload_features


@dataclass(frozen=True)
class ReuseAwareCostModelConfig:
    name: str = "feature_cost_v4"
    pim_memory_bandwidth: float = 4.0e12
    pim_scalar_throughput: float = 1.0e12
    pim_vector_throughput: float = 8.0e12
    host_pim_bandwidth: float = 512e9
    candidate_speedup_threshold: float = 1.10
    min_memory_pressure: float = 0.25
    max_risk_score: float = 0.75


def estimate_feature_cost_v4(
    profile_row: pd.Series,
    hardware: HardwareConfig,
    metadata_row: pd.Series | None = None,
    config: ReuseAwareCostModelConfig | None = None,
) -> dict[str, float | str | bool]:
    """Estimate GPU-vs-PIM opportunity with a cache/reuse-aware penalty.

    This model does not read cache counters directly. Until Nsight Compute
    counters are available, it uses workload metadata and arithmetic intensity
    as a proxy for reuse that is likely to favor GPU caches/shared memory.
    """

    cfg = config or ReuseAwareCostModelConfig()
    features = extract_workload_features(profile_row, hardware, metadata_row)
    runtime_s = float(profile_row["runtime_s"])
    flops = float(profile_row["flops"])
    dram_bytes = float(profile_row["dram_bytes"])

    predicted_gpu_time = runtime_s
    pim_memory_time = dram_bytes * _pim_memory_residency_factor(features) / cfg.pim_memory_bandwidth
    pim_compute_time = flops / _reuse_aware_pim_throughput(features, cfg)
    host_transfer_time = dram_bytes * _host_transfer_factor(features) / cfg.host_pim_bandwidth
    communication_time = runtime_s * _communication_factor(features)
    reuse_migration_time = runtime_s * _reuse_migration_factor(features)
    predicted_pim_time = (
        pim_memory_time
        + pim_compute_time
        + host_transfer_time
        + communication_time
        + reuse_migration_time
    )

    speedup = predicted_gpu_time / predicted_pim_time if predicted_pim_time > 0 else 0.0
    risk_score = _risk_score(features)
    predicted_candidate = (
        speedup >= cfg.candidate_speedup_threshold
        and features.memory_pressure >= cfg.min_memory_pressure
        and risk_score < cfg.max_risk_score
        and not _dense_reuse_negative_control(features)
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


def _reuse_aware_pim_throughput(features: WorkloadFeatures, config: ReuseAwareCostModelConfig) -> float:
    blend = features.compute_complexity
    base = config.pim_scalar_throughput * (1.0 - blend) + config.pim_vector_throughput * blend
    reuse_penalty = 1.0 - 0.65 * features.data_reuse_potential * features.compute_complexity
    return max(config.pim_scalar_throughput * 0.05, base * reuse_penalty)


def _pim_memory_residency_factor(features: WorkloadFeatures) -> float:
    base = 0.16 + 0.20 * features.host_transfer_sensitivity
    irregular_penalty = 0.10 * features.irregularity
    communication_penalty = 0.10 * features.communication_intensity
    reuse_penalty = 0.22 * features.data_reuse_potential
    return min(0.95, base + irregular_penalty + communication_penalty + reuse_penalty)


def _host_transfer_factor(features: WorkloadFeatures) -> float:
    return min(
        0.45,
        0.01
        + 0.18 * features.host_transfer_sensitivity
        + 0.05 * features.communication_intensity
        + 0.08 * features.data_reuse_potential,
    )


def _communication_factor(features: WorkloadFeatures) -> float:
    partition_penalty = 1.0 - features.partitionability
    return 0.01 + 0.10 * features.communication_intensity + 0.08 * partition_penalty


def _reuse_migration_factor(features: WorkloadFeatures) -> float:
    return 1.00 * features.data_reuse_potential * features.compute_complexity


def _risk_score(features: WorkloadFeatures) -> float:
    return min(
        1.0,
        0.30 * features.compute_complexity
        + 0.22 * features.communication_intensity
        + 0.18 * features.host_transfer_sensitivity
        + 0.14 * (1.0 - features.partitionability)
        + 0.16 * features.data_reuse_potential,
    )


def _dense_reuse_negative_control(features: WorkloadFeatures) -> bool:
    return features.data_reuse_potential >= 0.75 and features.compute_complexity >= 0.50


def _risk_summary(features: WorkloadFeatures) -> str:
    risks = []
    if features.compute_complexity >= 0.85:
        risks.append("compute-heavy")
    if features.data_reuse_potential >= 0.75:
        risks.append("high data reuse")
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
        f"reuse={features.data_reuse_potential:.2f}, "
        f"risk={_risk_score(features):.2f}"
    )
