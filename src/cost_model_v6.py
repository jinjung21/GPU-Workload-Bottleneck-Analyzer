from __future__ import annotations

import pandas as pd

from .config import HardwareConfig
from .cost_model_v5 import estimate_feature_cost_v5


V6_OPTIONAL_COUNTERS = (
    "ncu_l1_hit_rate_pct",
    "ncu_l2_hit_rate_pct",
    "ncu_global_load_efficiency_pct",
    "ncu_global_store_efficiency_pct",
    "ncu_warp_execution_efficiency_pct",
    "ncu_branch_efficiency_pct",
    "ncu_memory_stall_pct",
    "ncu_long_scoreboard_stall_pct",
    "ncu_short_scoreboard_stall_pct",
    "ncu_barrier_stall_pct",
    "ncu_eligible_warps_per_scheduler",
    "ncu_registers_per_thread",
)


def estimate_feature_cost_v6(
    profile_row: pd.Series,
    hardware: HardwareConfig,
    metadata_row: pd.Series | None = None,
) -> dict[str, float | str | bool]:
    """Cache/stall-aware PIM/NMP model.

    v6 extends v5 with optional Nsight Compute cache hit, memory efficiency,
    warp efficiency, stall, and transaction features. Missing counters are
    allowed so older Nsight Compute reports can still flow through the model.
    """

    base = estimate_feature_cost_v5(profile_row, hardware, metadata_row)
    if not _has_v6_metrics(profile_row):
        result = dict(base)
        result["risk"] = f"{base['risk']}; no cache/stall counters"
        result["feature_summary"] = f"{base['feature_summary']}, v6=partial"
        return result

    l1_hit = _pct(profile_row, "ncu_l1_hit_rate_pct", default=0.0)
    l2_hit = _pct(profile_row, "ncu_l2_hit_rate_pct", default=0.0)
    load_eff = _pct(profile_row, "ncu_global_load_efficiency_pct", default=1.0)
    store_eff = _pct(profile_row, "ncu_global_store_efficiency_pct", default=1.0)
    warp_eff = _pct(profile_row, "ncu_warp_execution_efficiency_pct", default=1.0)
    branch_eff = _pct(profile_row, "ncu_branch_efficiency_pct", default=1.0)
    memory_stall = _pct(profile_row, "ncu_memory_stall_pct", default=0.0)
    long_scoreboard = _pct(profile_row, "ncu_long_scoreboard_stall_pct", default=0.0)
    short_scoreboard = _pct(profile_row, "ncu_short_scoreboard_stall_pct", default=0.0)
    barrier_stall = _pct(profile_row, "ncu_barrier_stall_pct", default=0.0)
    eligible_warps = _number(profile_row, "ncu_eligible_warps_per_scheduler", default=4.0)
    registers = _number(profile_row, "ncu_registers_per_thread", default=0.0)
    collective_bonus = _collective_memory_bonus(profile_row, metadata_row)
    counter_coverage = sum(pd.notna(profile_row.get(column)) for column in V6_OPTIONAL_COUNTERS) / len(
        V6_OPTIONAL_COUNTERS
    )

    cache_hit_score = max(l1_hit, l2_hit)
    low_cache_reuse = 1.0 - cache_hit_score
    memory_efficiency_gap = 1.0 - min(load_eff, store_eff)
    latency_stall = max(memory_stall, long_scoreboard, short_scoreboard)
    divergence_risk = max(0.0, 1.0 - min(warp_eff, branch_eff))
    low_scheduler_supply = max(0.0, (2.0 - eligible_warps) / 2.0)
    register_pressure = min(1.0, registers / 128.0) if registers > 0 else 0.0

    pim_opportunity = min(
        1.0,
        0.30 * low_cache_reuse
        + 0.24 * latency_stall
        + 0.18 * memory_efficiency_gap
        + 0.16 * low_scheduler_supply
        + collective_bonus
        + 0.12 * _traffic_amplification(profile_row),
    )
    gpu_locality_advantage = min(
        1.0,
        0.42 * cache_hit_score
        + 0.22 * (1.0 - memory_efficiency_gap)
        + 0.18 * register_pressure
        + 0.18 * barrier_stall,
    )
    control_flow_risk = min(1.0, 0.65 * divergence_risk + 0.35 * barrier_stall)

    raw_calibration = 1.0 - 0.26 * pim_opportunity + 0.34 * gpu_locality_advantage + 0.22 * control_flow_risk
    calibration = 1.0 + counter_coverage * (raw_calibration - 1.0)
    calibrated_pim_time_ms = max(0.0, float(base["estimated_pim_time_ms"]) * calibration)
    gpu_time_ms = float(base["predicted_gpu_time_ms"])
    speedup = gpu_time_ms / calibrated_pim_time_ms if calibrated_pim_time_ms > 0 else 0.0
    risk_score = min(1.0, 0.42 * gpu_locality_advantage + 0.36 * control_flow_risk + 0.22 * register_pressure)
    standard_candidate = (
        bool(base["predicted_candidate"])
        and speedup >= 1.10
        and pim_opportunity >= 0.18
        and gpu_locality_advantage < 0.68
        and control_flow_risk < 0.55
        and risk_score < 0.68
    )
    collective_candidate = (
        collective_bonus > 0.0
        and speedup >= 1.10
        and pim_opportunity >= 0.50
        and gpu_locality_advantage < 0.68
        and control_flow_risk < 0.55
        and risk_score < 0.68
    )
    predicted_candidate = standard_candidate or collective_candidate

    return {
        "predicted_gpu_time_ms": gpu_time_ms,
        "estimated_pim_time_ms": calibrated_pim_time_ms,
        "estimated_speedup": speedup,
        "predicted_candidate": predicted_candidate,
        "risk_score": risk_score,
        "ncu_feature_coverage": counter_coverage,
        "risk": _risk_summary(
            base["risk"],
            pim_opportunity,
            gpu_locality_advantage,
            control_flow_risk,
            collective_bonus,
            counter_coverage,
        ),
        "feature_summary": (
            f"{base['feature_summary']}, "
            f"cache_hit={cache_hit_score:.2f}, mem_eff_gap={memory_efficiency_gap:.2f}, "
            f"latency_stall={latency_stall:.2f}, divergence={divergence_risk:.2f}, "
            f"collective_bonus={collective_bonus:.2f}, v6_opportunity={pim_opportunity:.2f}, "
            f"v6_risk={risk_score:.2f}, ncu_coverage={counter_coverage:.0%}"
        ),
    }


def _has_v6_metrics(row: pd.Series) -> bool:
    return any(
        pd.notna(row.get(column))
        for column in [
            "ncu_l1_hit_rate_pct",
            "ncu_l2_hit_rate_pct",
            "ncu_global_load_efficiency_pct",
            "ncu_global_store_efficiency_pct",
            "ncu_warp_execution_efficiency_pct",
            "ncu_branch_efficiency_pct",
            "ncu_memory_stall_pct",
            "ncu_long_scoreboard_stall_pct",
            "ncu_short_scoreboard_stall_pct",
        ]
    )


def _pct(row: pd.Series, column: str, default: float) -> float:
    value = row.get(column)
    if pd.isna(value):
        return default
    return max(0.0, min(1.0, float(value) / 100.0))


def _number(row: pd.Series, column: str, default: float) -> float:
    value = row.get(column)
    if pd.isna(value):
        return default
    return float(value)


def _traffic_amplification(row: pd.Series) -> float:
    l2_transactions = _number(row, "ncu_l2_read_transactions", 0.0) + _number(row, "ncu_l2_write_transactions", 0.0)
    if l2_transactions <= 0:
        return 0.0
    dram_bytes = max(1.0, float(row["dram_bytes"]))
    return min(1.0, l2_transactions * 32.0 / dram_bytes)


def _collective_memory_bonus(row: pd.Series, metadata_row: pd.Series | None) -> float:
    name = str(row.get("kernel_name", "")).strip().lower()
    notes = str(metadata_row.get("paper_notes", "") if metadata_row is not None else "").lower()
    if any(token in name for token in ["reduction", "scan"]):
        return 0.08
    if any(token in notes for token in ["reduction", "scan", "prefix", "synchronization"]):
        return 0.08
    return 0.0


def _risk_summary(
    base_risk: object,
    pim_opportunity: float,
    gpu_locality_advantage: float,
    control_flow_risk: float,
    collective_bonus: float,
    counter_coverage: float,
) -> str:
    risks = [str(base_risk)] if str(base_risk) else []
    if pim_opportunity >= 0.55:
        risks.append("cache/stall counters favor PIM")
    if collective_bonus > 0.0:
        risks.append("collective memory primitive")
    if gpu_locality_advantage >= 0.55:
        risks.append("GPU locality advantage")
    if control_flow_risk >= 0.35:
        risks.append("control-flow or barrier risk")
    if counter_coverage < 0.75:
        risks.append(f"partial NCU coverage ({counter_coverage:.0%})")
    return ", ".join(risks)
