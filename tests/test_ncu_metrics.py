from pathlib import Path

import pandas as pd

from src.config import HardwareConfig
from src.cost_model_v5 import estimate_feature_cost_v5
from src.ncu_metrics import attach_ncu_metrics, parse_ncu_report_file
from src.roofline import add_roofline_metrics


def test_parse_ncu_text_report(tmp_path: Path) -> None:
    report = tmp_path / "vector_add_ncu.txt"
    report.write_text(
        """
        Section: GPU Speed Of Light
        Memory [%]                                                                           %                          85.34
        SOL DRAM                                                                             %                          85.34
        SOL L1/TEX Cache                                                                     %                          16.39
        SOL L2 Cache                                                                         %                          31.08
        SM [%]                                                                               %                          12.71
        Duration                                                                       usecond                         358.08
        Section: Occupancy
        Achieved Occupancy                                                                   %                          87.88
        Achieved Active Warps Per SM                                                      warp                          28.12
        """,
        encoding="utf-8",
    )

    row = parse_ncu_report_file(report)

    assert row["kernel_name"] == "vector_add"
    assert row["ncu_sol_dram_pct"] == 85.34
    assert row["ncu_sm_util_pct"] == 12.71
    assert row["ncu_achieved_occupancy_pct"] == 87.88


def test_attach_ncu_metrics_matches_kernel_name() -> None:
    profile = pd.DataFrame({"kernel_name": ["vector_add"], "runtime_ms": [1.0]})
    metrics = pd.DataFrame({"kernel_name": ["Vector-Add"], "ncu_sol_dram_pct": [80.0]})

    merged = attach_ncu_metrics(profile, metrics)

    assert merged.loc[0, "ncu_sol_dram_pct"] == 80.0


def test_feature_cost_v5_uses_ncu_memory_bound_signal() -> None:
    hardware = HardwareConfig("test", peak_flops=10e12, peak_memory_bandwidth=1e12)
    profile = pd.DataFrame(
        {
            "kernel_name": ["vector_add"],
            "runtime_ms": [1.0],
            "flops": [1e6],
            "dram_read_bytes": [512e6],
            "dram_write_bytes": [512e6],
            "memory_access_pattern": ["regular"],
            "notes": [""],
        }
    )
    profile = add_roofline_metrics(profile, hardware)
    row = profile.iloc[0].copy()
    row["ncu_sol_dram_pct"] = 85.0
    row["ncu_memory_util_pct"] = 85.0
    row["ncu_sm_util_pct"] = 12.0
    row["ncu_sol_l2_pct"] = 31.0
    row["ncu_sol_l1_tex_pct"] = 16.0
    row["ncu_achieved_occupancy_pct"] = 88.0

    estimate = estimate_feature_cost_v5(row, hardware)

    assert "ncu_dram=0.85" in str(estimate["feature_summary"])
    assert float(estimate["estimated_speedup"]) > 0.0
