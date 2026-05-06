import pandas as pd

from src.classifier import add_classifications
from src.config import HardwareConfig
from src.roofline import add_roofline_metrics


def test_roofline_metrics_preserve_raw_utilization_above_one() -> None:
    hardware = HardwareConfig(peak_flops=100.0, peak_memory_bandwidth=50.0)
    profile = pd.DataFrame(
        {
            "kernel_name": ["too_fast"],
            "runtime_ms": [100.0],
            "flops": [20.0],
            "dram_read_bytes": [10.0],
            "dram_write_bytes": [0.0],
            "memory_access_pattern": ["regular"],
            "notes": [""],
        }
    )

    result = add_roofline_metrics(profile, hardware)

    assert result.loc[0, "roofline_utilization_raw"] == 2.0
    assert result.loc[0, "roofline_utilization"] == 1.0
    assert result.loc[0, "exceeds_roofline"]


def test_classifier_marks_irregular_large_low_intensity_as_strong_pim_candidate() -> None:
    hardware = HardwareConfig(peak_flops=15e12, peak_memory_bandwidth=900e9)
    profile = pd.DataFrame(
        {
            "kernel_name": ["embedding_lookup"],
            "runtime_ms": [4.0],
            "flops": [60_000_000.0],
            "dram_read_bytes": [2_500_000_000.0],
            "dram_write_bytes": [120_000_000.0],
            "memory_access_pattern": ["irregular"],
            "notes": [""],
        }
    )

    result = add_roofline_metrics(profile, hardware)
    result = add_classifications(result, hardware)

    assert result.loc[0, "bottleneck_classification"] == "memory-bound / irregular"
    assert result.loc[0, "pim_nmp_suitability"] == "strong NMP/PIM candidate"
    assert result.loc[0, "pim_nmp_score"] >= 75
    assert "irregular access" in result.loc[0, "pim_nmp_score_reason"]


def test_classifier_keeps_compute_bound_kernel_low_priority() -> None:
    hardware = HardwareConfig(peak_flops=15e12, peak_memory_bandwidth=900e9)
    profile = pd.DataFrame(
        {
            "kernel_name": ["matrix_mul_tiled"],
            "runtime_ms": [12.0],
            "flops": [120_000_000_000.0],
            "dram_read_bytes": [750_000_000.0],
            "dram_write_bytes": [16_000_000.0],
            "memory_access_pattern": ["regular"],
            "notes": [""],
        }
    )

    result = add_roofline_metrics(profile, hardware)
    result = add_classifications(result, hardware)

    assert result.loc[0, "bottleneck_classification"] == "compute-bound"
    assert result.loc[0, "pim_nmp_suitability"] == "low PIM priority"
    assert result.loc[0, "pim_nmp_score"] < 35
