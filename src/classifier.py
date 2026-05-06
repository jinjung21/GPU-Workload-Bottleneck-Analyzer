import pandas as pd

from .config import HardwareConfig


def classify_kernel(row: pd.Series, hardware: HardwareConfig) -> tuple[str, str, int, str, str]:
    """Return bottleneck class, PIM/NMP suitability, score, rationale, and recommendation."""

    intensity = float(row["arithmetic_intensity"])
    utilization = float(row["roofline_utilization"])
    access_pattern = str(row["memory_access_pattern"]).lower()

    is_memory_bound = intensity < hardware.ridge_point
    is_underutilized = utilization < 0.35

    if is_underutilized:
        bottleneck = "underutilized / latency-bound"
    elif is_memory_bound and access_pattern == "irregular":
        bottleneck = "memory-bound / irregular"
    elif is_memory_bound:
        bottleneck = "memory-bound / bandwidth-bound"
    else:
        bottleneck = "compute-bound"

    pim_score, pim_reason = score_pim_nmp_candidate(row, hardware)
    if pim_score >= 85 or (pim_score >= 75 and access_pattern == "irregular"):
        pim = "strong NMP/PIM candidate"
    elif pim_score >= 60:
        pim = "PIM-friendly"
    elif pim_score >= 35:
        pim = "possible PIM candidate"
    else:
        pim = "low PIM priority"

    if is_underutilized:
        recommendation = "latency, divergence, occupancy 문제 가능성 확인"
    elif is_memory_bound and access_pattern == "irregular":
        recommendation = "데이터 locality가 낮고 PIM/NMP 오프로딩에 매우 적합"
    elif is_memory_bound:
        recommendation = "메모리 트래픽 감소, coalescing 개선, bulk 연산의 경우 PIM 고려"
    else:
        recommendation = "연산 최적화, tiling, occupancy 개선 필요"

    return bottleneck, pim, pim_score, pim_reason, recommendation


def score_pim_nmp_candidate(row: pd.Series, hardware: HardwareConfig) -> tuple[int, str]:
    """Score how attractive a kernel is for near/in-memory execution.

    The score intentionally uses normalized Roofline features instead of fixed
    kernel names so the same model can evaluate future Nsight Compute data.
    """

    intensity = float(row["arithmetic_intensity"])
    dram_bytes = float(row["dram_bytes"])
    achieved_bandwidth = float(row["achieved_bandwidth"])
    access_pattern = str(row["memory_access_pattern"]).lower()

    intensity_score = max(0.0, 1.0 - min(intensity / hardware.ridge_point, 1.0)) * 35.0
    traffic_score = min(dram_bytes / 1e9, 1.0) * 25.0
    bandwidth_score = min(achieved_bandwidth / hardware.peak_memory_bandwidth, 1.0) * 20.0
    irregular_score = 20.0 if access_pattern == "irregular" else 0.0
    total = round(intensity_score + traffic_score + bandwidth_score + irregular_score)

    reasons = []
    if intensity_score >= 20:
        reasons.append("low arithmetic intensity")
    if traffic_score >= 15:
        reasons.append("large DRAM traffic")
    if bandwidth_score >= 10:
        reasons.append("high bandwidth pressure")
    if irregular_score:
        reasons.append("irregular access")
    if not reasons:
        reasons.append("limited memory-side benefit")

    return int(total), ", ".join(reasons)


def add_classifications(profile: pd.DataFrame, hardware: HardwareConfig) -> pd.DataFrame:
    """Attach classification columns to a Roofline metrics dataframe."""

    result = profile.copy()
    classified = result.apply(lambda row: classify_kernel(row, hardware), axis=1)
    result["bottleneck_classification"] = [item[0] for item in classified]
    result["pim_nmp_suitability"] = [item[1] for item in classified]
    result["pim_nmp_score"] = [item[2] for item in classified]
    result["pim_nmp_score_reason"] = [item[3] for item in classified]
    result["recommendation"] = [item[4] for item in classified]
    return result
