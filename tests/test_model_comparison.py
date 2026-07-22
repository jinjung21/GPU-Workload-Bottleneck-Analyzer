import pandas as pd

from src.config import HardwareConfig
from src.cost_model_v3 import estimate_feature_cost_v3
from src.cost_model_v4 import estimate_feature_cost_v4
from src.cost_model_v6 import estimate_feature_cost_v6
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
    feature_v3 = metrics[metrics["model"] == "feature_cost_v3"].iloc[0]
    feature_v4 = metrics[metrics["model"] == "feature_cost_v4"].iloc[0]
    assert traffic["fp"] == 1
    assert analytical["tn"] == 1
    assert feature_v3["tn"] == 1
    assert feature_v4["tn"] == 1
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


def test_feature_cost_v3_rejects_sort_like_high_transfer_workload() -> None:
    hardware = HardwareConfig(peak_flops=15e12, peak_memory_bandwidth=900e9)
    profile_row = pd.Series(
        {
            "runtime_s": 0.007,
            "flops": 3_000_000_000.0,
            "dram_bytes": 4_000_000_000.0,
            "arithmetic_intensity": 0.75,
            "achieved_flops": 428.57e9,
            "achieved_bandwidth": 571.43e9,
            "memory_access_pattern": "irregular",
        }
    )
    metadata_row = pd.Series(
        {
            "operation_complexity": "medium",
            "communication_intensity": "high",
            "partitionability": "medium",
            "host_transfer_sensitivity": "high",
        }
    )

    estimate = estimate_feature_cost_v3(profile_row, hardware, metadata_row)

    assert estimate["estimated_speedup"] > 1.0
    assert not estimate["predicted_candidate"]
    assert "host transfer sensitive" in estimate["risk"]


def test_feature_cost_v4_penalizes_high_reuse_dense_gemm() -> None:
    hardware = HardwareConfig(peak_flops=13.45e12, peak_memory_bandwidth=616e9)
    profile_row = pd.Series(
        {
            "runtime_s": 0.00113,
            "flops": 2_147_483_648.0,
            "dram_bytes": 541_065_216.0,
            "arithmetic_intensity": 3.969,
            "achieved_flops": 1.90e12,
            "achieved_bandwidth": 479e9,
            "memory_access_pattern": "regular",
        }
    )
    metadata_row = pd.Series(
        {
            "operation_complexity": "medium",
            "communication_intensity": "low",
            "partitionability": "high",
            "host_transfer_sensitivity": "medium",
            "data_reuse_potential": "high",
        }
    )

    estimate = estimate_feature_cost_v4(profile_row, hardware, metadata_row)

    assert estimate["estimated_speedup"] < 1.0
    assert not estimate["predicted_candidate"]
    assert "high data reuse" in estimate["risk"]


def test_feature_cost_v6_reports_partial_counter_coverage() -> None:
    hardware = HardwareConfig(peak_flops=13.45e12, peak_memory_bandwidth=616e9)
    profile_row = pd.Series(
        {
            "kernel_name": "vector_add",
            "runtime_s": 0.00036,
            "runtime_ms": 0.36,
            "flops": 16_777_216.0,
            "dram_bytes": 201_326_592.0,
            "arithmetic_intensity": 0.0833,
            "achieved_flops": 46e9,
            "achieved_bandwidth": 555e9,
            "memory_access_pattern": "regular",
            "ncu_sol_dram_pct": 85.0,
            "ncu_l1_hit_rate_pct": 0.0,
            "ncu_l2_hit_rate_pct": 33.0,
            "ncu_long_scoreboard_stall_pct": 91.0,
        }
    )
    metadata_row = pd.Series(
        {
            "operation_complexity": "low",
            "communication_intensity": "low",
            "partitionability": "high",
            "host_transfer_sensitivity": "low",
            "data_reuse_potential": "low",
        }
    )

    estimate = estimate_feature_cost_v6(profile_row, hardware, metadata_row)

    assert 0 < estimate["ncu_feature_coverage"] < 0.75
    assert "partial NCU coverage" in estimate["risk"]
