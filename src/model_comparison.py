from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import HardwareConfig
from .cost_model_v3 import estimate_feature_cost_v3
from .cost_model_v4 import estimate_feature_cost_v4
from .cost_model_v5 import estimate_feature_cost_v5


@dataclass(frozen=True)
class AnalyticalPIMConfig:
    name: str = "bank_parallel_pim_opportunity"
    pim_memory_bandwidth: float = 4.0e12
    host_pim_bandwidth: float = 512e9
    low_complexity_throughput: float = 1.0e12
    medium_complexity_throughput: float = 8.0e12
    high_complexity_throughput: float = 0.02e12
    candidate_speedup_threshold: float = 1.10


def build_model_comparison(
    profile: pd.DataFrame,
    baseline: pd.DataFrame,
    hardware: HardwareConfig,
    pim_config: AnalyticalPIMConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = pim_config or AnalyticalPIMConfig()
    rows = []
    profile_by_name = {_normalize_name(row["kernel_name"]): row for _, row in profile.iterrows()}

    for _, expected in baseline.iterrows():
        matched = _find_profile_row(profile_by_name, expected["benchmark"], expected["aliases"])
        if matched is None:
            continue

        target = bool(expected["expected_pim_candidate"])
        predictions = {
            "ai_only": _predict_ai_only(matched, hardware),
            "traffic_only": _predict_traffic_only(matched),
            "heuristic_v1": bool(matched["pim_nmp_score"] >= 60),
        }
        analytical = estimate_pim_speedup(matched, expected, config)
        predictions["analytical_v2"] = _predict_analytical_candidate(analytical, expected, config)
        feature_cost = estimate_feature_cost_v3(matched, hardware, expected)
        predictions["feature_cost_v3"] = bool(feature_cost["predicted_candidate"])
        feature_cost_v4 = estimate_feature_cost_v4(matched, hardware, expected)
        predictions["feature_cost_v4"] = bool(feature_cost_v4["predicted_candidate"])
        feature_cost_v5 = None
        if _has_ncu_metrics(matched):
            feature_cost_v5 = estimate_feature_cost_v5(matched, hardware, expected)
            predictions["feature_cost_v5"] = bool(feature_cost_v5["predicted_candidate"])

        for model_name, predicted in predictions.items():
            estimate = _model_estimate(model_name, analytical, feature_cost, feature_cost_v4, feature_cost_v5)
            rows.append(
                {
                    "model": model_name,
                    "benchmark": expected["benchmark"],
                    "target_candidate": target,
                    "predicted_candidate": predicted,
                    "gpu_runtime_ms": float(matched["runtime_ms"]),
                    "estimated_speedup": estimate["estimated_speedup"],
                    "estimated_pim_time_ms": estimate["estimated_pim_time_ms"],
                    "risk": estimate["risk"],
                    "feature_summary": estimate["feature_summary"],
                }
            )

    comparison = pd.DataFrame(rows)
    metrics = evaluate_binary_models(comparison)
    return comparison, metrics


def estimate_pim_speedup(
    profile_row: pd.Series,
    metadata_row: pd.Series,
    config: AnalyticalPIMConfig | None = None,
) -> dict[str, float | str]:
    cfg = config or AnalyticalPIMConfig()
    runtime_s = float(profile_row["runtime_s"])
    flops = float(profile_row["flops"])
    dram_bytes = float(profile_row["dram_bytes"])

    complexity = str(metadata_row["operation_complexity"]).lower()
    communication = str(metadata_row["communication_intensity"]).lower()
    partitionability = str(metadata_row["partitionability"]).lower()
    host_sensitivity = str(metadata_row["host_transfer_sensitivity"]).lower()

    compute_time = flops / _pim_compute_throughput(complexity, cfg)
    memory_time = dram_bytes * _memory_residency_factor(host_sensitivity) / cfg.pim_memory_bandwidth
    host_time = dram_bytes * _host_transfer_factor(host_sensitivity) / cfg.host_pim_bandwidth
    sync_time = runtime_s * _sync_overhead_factor(communication, partitionability)
    estimated_time = compute_time + memory_time + host_time + sync_time
    speedup = runtime_s / estimated_time if estimated_time > 0 else 0.0

    return {
        "estimated_pim_time_ms": estimated_time * 1000.0,
        "estimated_speedup": speedup,
        "risk": _risk_summary(complexity, communication, partitionability, host_sensitivity),
    }


def evaluate_binary_models(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in comparison.groupby("model"):
        target = group["target_candidate"].astype(bool)
        predicted = group["predicted_candidate"].astype(bool)
        tp = int((target & predicted).sum())
        fp = int((~target & predicted).sum())
        tn = int((~target & ~predicted).sum())
        fn = int((target & ~predicted).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        rows.append(
            {
                "model": model_name,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": _safe_div(2 * precision * recall, precision + recall),
                "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
            }
        )
    return pd.DataFrame(rows).sort_values(["f1", "precision", "recall"], ascending=False)


def _model_estimate(
    model_name: str,
    analytical: dict[str, float | str],
    feature_cost: dict[str, float | str | bool],
    feature_cost_v4: dict[str, float | str | bool],
    feature_cost_v5: dict[str, float | str | bool] | None,
) -> dict[str, object]:
    if model_name == "analytical_v2":
        return {
            "estimated_speedup": analytical["estimated_speedup"],
            "estimated_pim_time_ms": analytical["estimated_pim_time_ms"],
            "risk": analytical["risk"],
            "feature_summary": "",
        }
    if model_name == "feature_cost_v3":
        return {
            "estimated_speedup": feature_cost["estimated_speedup"],
            "estimated_pim_time_ms": feature_cost["estimated_pim_time_ms"],
            "risk": feature_cost["risk"],
            "feature_summary": feature_cost["feature_summary"],
        }
    if model_name == "feature_cost_v4":
        return {
            "estimated_speedup": feature_cost_v4["estimated_speedup"],
            "estimated_pim_time_ms": feature_cost_v4["estimated_pim_time_ms"],
            "risk": feature_cost_v4["risk"],
            "feature_summary": feature_cost_v4["feature_summary"],
        }
    if model_name == "feature_cost_v5" and feature_cost_v5 is not None:
        return {
            "estimated_speedup": feature_cost_v5["estimated_speedup"],
            "estimated_pim_time_ms": feature_cost_v5["estimated_pim_time_ms"],
            "risk": feature_cost_v5["risk"],
            "feature_summary": feature_cost_v5["feature_summary"],
        }
    return {
        "estimated_speedup": "",
        "estimated_pim_time_ms": "",
        "risk": "",
        "feature_summary": "",
    }


def _predict_ai_only(row: pd.Series, hardware: HardwareConfig) -> bool:
    return float(row["arithmetic_intensity"]) < hardware.ridge_point


def _predict_traffic_only(row: pd.Series) -> bool:
    return float(row["dram_bytes"]) >= 1e9


def _predict_analytical_candidate(
    estimate: dict[str, float | str],
    metadata_row: pd.Series,
    config: AnalyticalPIMConfig,
) -> bool:
    if str(metadata_row["operation_complexity"]).lower() == "high":
        return False
    if (
        str(metadata_row["communication_intensity"]).lower() == "high"
        and str(metadata_row["host_transfer_sensitivity"]).lower() == "high"
    ):
        return False
    return float(estimate["estimated_speedup"]) >= config.candidate_speedup_threshold


def _pim_compute_throughput(complexity: str, config: AnalyticalPIMConfig) -> float:
    if complexity == "low":
        return config.low_complexity_throughput
    if complexity == "medium":
        return config.medium_complexity_throughput
    return config.high_complexity_throughput


def _memory_residency_factor(host_sensitivity: str) -> float:
    return {"low": 0.20, "medium": 0.35, "high": 0.70}.get(host_sensitivity, 0.45)


def _host_transfer_factor(host_sensitivity: str) -> float:
    return {"low": 0.005, "medium": 0.03, "high": 0.20}.get(host_sensitivity, 0.05)


def _sync_overhead_factor(communication: str, partitionability: str) -> float:
    communication_factor = {"low": 0.005, "medium": 0.03, "high": 0.10}.get(communication, 0.05)
    partition_penalty = {"high": 0.00, "medium": 0.02, "low": 0.08}.get(partitionability, 0.03)
    return communication_factor + partition_penalty


def _risk_summary(complexity: str, communication: str, partitionability: str, host_sensitivity: str) -> str:
    risks = []
    if complexity == "high":
        risks.append("compute-heavy")
    if communication == "high":
        risks.append("high communication")
    if partitionability == "low":
        risks.append("poor partitionability")
    if host_sensitivity == "high":
        risks.append("host transfer sensitive")
    return ", ".join(risks) if risks else "low model risk"


def _find_profile_row(profile_by_name: dict[str, pd.Series], benchmark: str, aliases: str) -> pd.Series | None:
    candidates = [benchmark, *str(aliases).split("|")]
    for candidate in candidates:
        normalized = _normalize_name(candidate)
        if normalized in profile_by_name:
            return profile_by_name[normalized]
    return None


def _normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _has_ncu_metrics(row: pd.Series) -> bool:
    return any(pd.notna(row.get(column)) for column in ["ncu_sol_dram_pct", "ncu_memory_util_pct", "ncu_sm_util_pct"])


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
