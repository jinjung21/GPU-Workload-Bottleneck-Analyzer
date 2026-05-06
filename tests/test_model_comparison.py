import pandas as pd

from src.config import HardwareConfig
from src.model_comparison import build_model_comparison, estimate_pim_speedup
from src.roofline import add_roofline_metrics
from src.classifier import add_classifications


def test_model_comparison_reports_false_positive_for_traffic_only() -> None:
    hardware = HardwareConfig(peak_flops=15e12, peak_memory_bandwidth=900e9)
    profile = pd.DataFrame(
        {
            "kernel_name": ["GEMM-TILED"],
            "runtime_ms": [12.0],
            "flops": [120_000_000_000.0],
            "dram_read_bytes": [750_000_000.0],
            "dram_write_bytes": [300_000_000.0],
            "memory_access_pattern": ["regular"],
            "notes": [""],
        }
    )
    baseline = pd.DataFrame(
        {
            "benchmark": ["GEMM-TILED"],
            "aliases": ["matrix_mul_tiled"],
            "domain": ["Dense linear algebra"],
            "paper_memory_bound": [False],
            "expected_pim_candidate": [False],
            "expected_pim_priority": ["low"],
            "expected_min_score": [0],
            "operation_complexity": ["high"],
            "communication_intensity": ["low"],
            "partitionability": ["high"],
            "host_transfer_sensitivity": ["medium"],
            "paper_notes": ["Negative control"],
        }
    )

    analyzed = add_classifications(add_roofline_metrics(profile, hardware), hardware)
    comparison, metrics = build_model_comparison(analyzed, baseline, hardware)

    traffic = metrics[metrics["model"] == "traffic_only"].iloc[0]
    analytical = metrics[metrics["model"] == "analytical_v2"].iloc[0]
    assert traffic["fp"] == 1
    assert analytical["tn"] == 1
    assert not bool(comparison[comparison["model"] == "analytical_v2"].iloc[0]["predicted_candidate"])


def test_estimate_pim_speedup_penalizes_high_complexity_workload() -> None:
    profile_row = pd.Series(
        {
            "runtime_s": 0.01,
            "flops": 100_000_000_000.0,
            "dram_bytes": 1_000_000_000.0,
        }
    )
    metadata_row = pd.Series(
        {
            "operation_complexity": "high",
            "communication_intensity": "high",
            "partitionability": "medium",
            "host_transfer_sensitivity": "high",
        }
    )

    estimate = estimate_pim_speedup(profile_row, metadata_row)

    assert estimate["estimated_speedup"] < 1.0
    assert "compute-heavy" in estimate["risk"]
