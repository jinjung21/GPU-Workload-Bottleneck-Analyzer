import pandas as pd

from .config import HardwareConfig


def classify_kernel(row: pd.Series, hardware: HardwareConfig) -> tuple[str, str, str]:
    """Return bottleneck class, PIM/NMP suitability, and optimization recommendation."""

    intensity = float(row["arithmetic_intensity"])
    utilization = float(row["roofline_utilization"])
    dram_bytes = float(row["dram_bytes"])
    access_pattern = str(row["memory_access_pattern"]).lower()

    is_memory_bound = intensity < hardware.ridge_point
    is_low_intensity = intensity < hardware.ridge_point * 0.35
    is_large_memory = dram_bytes >= 1e9
    is_underutilized = utilization < 0.35

    if is_underutilized:
        bottleneck = "underutilized / latency-bound"
    elif is_memory_bound and access_pattern == "irregular":
        bottleneck = "memory-bound / irregular"
    elif is_memory_bound:
        bottleneck = "memory-bound / bandwidth-bound"
    else:
        bottleneck = "compute-bound"

    if is_low_intensity and access_pattern == "irregular" and is_large_memory:
        pim = "strong NMP/PIM candidate"
    elif is_low_intensity and is_large_memory:
        pim = "PIM-friendly"
    elif is_memory_bound:
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

    return bottleneck, pim, recommendation


def add_classifications(profile: pd.DataFrame, hardware: HardwareConfig) -> pd.DataFrame:
    """Attach classification columns to a Roofline metrics dataframe."""

    result = profile.copy()
    classified = result.apply(lambda row: classify_kernel(row, hardware), axis=1)
    result["bottleneck_classification"] = [item[0] for item in classified]
    result["pim_nmp_suitability"] = [item[1] for item in classified]
    result["recommendation"] = [item[2] for item in classified]
    return result
