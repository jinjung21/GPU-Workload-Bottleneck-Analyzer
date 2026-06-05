from __future__ import annotations

import pandas as pd

from .config import HardwareConfig
from .cost_model_v4 import estimate_feature_cost_v4


def estimate_feature_cost_v5(
    profile_row: pd.Series,
    hardware: HardwareConfig,
    metadata_row: pd.Series | None = None,
) -> dict[str, float | str | bool]:
    """Counter-aware PIM/NMP model.

    v5 starts from the reuse-aware v4 cost model and then calibrates the
    decision with measured Nsight Compute signals when they are present.
    """

    base = estimate_feature_cost_v4(profile_row, hardware, metadata_row)
    if not _has_ncu_metrics(profile_row):
        result = dict(base)
        result["risk"] = f"{base['risk']}; no NCU counters"
        result["feature_summary"] = f"{base['feature_summary']}, ncu=missing"
        return result

    dram = _pct(profile_row, "ncu_sol_dram_pct", fallback="ncu_memory_util_pct", default=0.0)
    memory = _pct(profile_row, "ncu_memory_util_pct", fallback="ncu_sol_dram_pct", default=0.0)
    sm = _pct(profile_row, "ncu_sm_util_pct")
    l2 = _pct(profile_row, "ncu_sol_l2_pct")
    l1 = _pct(profile_row, "ncu_sol_l1_tex_pct")
    occupancy = _pct(profile_row, "ncu_achieved_occupancy_pct")
    arithmetic_intensity = float(profile_row["arithmetic_intensity"])

    memory_bound_signal = max(dram, memory)
    if sm is not None:
        memory_bound_signal *= 1.0 - min(0.95, sm)
    cache_reuse_signal = max(l1 or 0.0, l2 or 0.0) * min(1.0, arithmetic_intensity / max(hardware.ridge_point, 1e-12))
    compute_signal = (sm or 0.0) * min(1.0, arithmetic_intensity / max(hardware.ridge_point, 1e-12))
    occupancy_risk = 0.0 if occupancy is None else max(0.0, 0.45 - occupancy)

    calibration = 1.0 - 0.22 * memory_bound_signal + 0.30 * cache_reuse_signal + 0.25 * compute_signal
    calibration += 0.15 * occupancy_risk
    calibrated_pim_time_ms = max(0.0, float(base["estimated_pim_time_ms"]) * calibration)
    gpu_time_ms = float(base["predicted_gpu_time_ms"])
    speedup = gpu_time_ms / calibrated_pim_time_ms if calibrated_pim_time_ms > 0 else 0.0

    ncu_risk_score = min(
        1.0,
        0.35 * compute_signal
        + 0.25 * cache_reuse_signal
        + 0.20 * occupancy_risk
        + 0.20 * max(0.0, (sm or 0.0) - max(dram, memory)),
    )
    predicted_candidate = (
        bool(base["predicted_candidate"])
        and speedup >= 1.10
        and memory_bound_signal >= 0.20
        and ncu_risk_score < 0.60
    )

    return {
        "predicted_gpu_time_ms": gpu_time_ms,
        "estimated_pim_time_ms": calibrated_pim_time_ms,
        "estimated_speedup": speedup,
        "predicted_candidate": predicted_candidate,
        "risk_score": ncu_risk_score,
        "risk": _risk_summary(
            base["risk"],
            memory_bound_signal,
            cache_reuse_signal,
            compute_signal,
            occupancy_risk,
            occupancy is not None,
        ),
        "feature_summary": (
            f"{base['feature_summary']}, "
            f"ncu_dram={_format_optional(dram)}, ncu_sm={_format_optional(sm)}, "
            f"ncu_l2={_format_optional(l2)}, ncu_occ={_format_optional(occupancy)}, "
            f"ncu_risk={ncu_risk_score:.2f}"
        ),
    }


def _has_ncu_metrics(row: pd.Series) -> bool:
    return any(pd.notna(row.get(column)) for column in ["ncu_sol_dram_pct", "ncu_memory_util_pct", "ncu_sm_util_pct"])


def _pct(row: pd.Series, column: str, fallback: str | None = None, default: float | None = None) -> float | None:
    value = row.get(column)
    if pd.isna(value) and fallback is not None:
        value = row.get(fallback)
    if pd.isna(value):
        return default
    return max(0.0, min(1.0, float(value) / 100.0))


def _risk_summary(
    base_risk: object,
    memory_bound_signal: float,
    cache_reuse_signal: float,
    compute_signal: float,
    occupancy_risk: float,
    occupancy_measured: bool,
) -> str:
    risks = [str(base_risk)] if str(base_risk) else []
    if memory_bound_signal >= 0.60:
        risks.append("NCU confirms DRAM-bound")
    if cache_reuse_signal >= 0.35:
        risks.append("NCU cache/reuse penalty")
    if compute_signal >= 0.45:
        risks.append("NCU compute pressure")
    if occupancy_measured and occupancy_risk >= 0.15:
        risks.append("low occupancy")
    return ", ".join(risks)


def _format_optional(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"
