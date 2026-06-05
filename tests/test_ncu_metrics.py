from pathlib import Path

import pandas as pd

from src.config import HardwareConfig
from src.cost_model_v5 import estimate_feature_cost_v5
from src.cost_model_v6 import estimate_feature_cost_v6
from src.ncu_metrics import attach_ncu_metrics, parse_ncu_report_file
from src.roofline import add_roofline_metrics


def test_parse_ncu_text_report(tmp_path: Path) -> None:
    report = tmp_path / "vector_add_ncu.txt"
    report.write_text(
        """
        Section: GPU Speed Of Light
        Memory [%]                                                                           %                          85.34
        Memory Throughput                                                         Gbyte/second                         554.92
        Mem Busy                                                                             %                          31.08
        Max Bandwidth                                                                        %                          85.41
        SOL DRAM                                                                             %                          85.34
        SOL L1/TEX Cache                                                                     %                          16.39
        SOL L2 Cache                                                                         %                          31.08
        SM [%]                                                                               %                          12.71
        Duration                                                                       usecond                         358.08
        Section: Occupancy
        Achieved Occupancy                                                                   %                          87.88
        Achieved Active Warps Per SM                                                      warp                          28.12
        L1/TEX Hit Rate                                                                      %                          22.50
        L2 Hit Rate                                                                          %                          35.00
        Warp Execution Efficiency                                                           %                          96.00
        Stall Long Scoreboard                                                                %                          31.00
        Eligible Warps Per Scheduler                                                      warp                           0.07
        WRN   On average each warp of this kernel spends 101.6 cycles being stalled waiting on
              a L1TEX operation. This represents about 91.2% of the total average.
        """,
        encoding="utf-8",
    )

    row = parse_ncu_report_file(report)

    assert row["kernel_name"] == "vector_add"
    assert row["ncu_sol_dram_pct"] == 85.34
    assert row["ncu_sm_util_pct"] == 12.71
    assert row["ncu_achieved_occupancy_pct"] == 87.88
    assert row["ncu_l1_hit_rate_pct"] == 22.50
    assert row["ncu_l2_hit_rate_pct"] == 35.00
    assert row["ncu_warp_execution_efficiency_pct"] == 96.00
    assert row["ncu_memory_throughput_gbps"] == 554.92
    assert row["ncu_mem_busy_pct"] == 31.08
    assert row["ncu_max_bandwidth_pct"] == 85.41
    assert row["ncu_eligible_warps_per_scheduler"] == 0.07
    assert row["ncu_long_scoreboard_stall_pct"] == 91.2


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


def test_feature_cost_v6_uses_cache_and_stall_signals() -> None:
    hardware = HardwareConfig("test", peak_flops=10e12, peak_memory_bandwidth=1e12)
    profile = pd.DataFrame(
        {
            "kernel_name": ["random_gather"],
            "runtime_ms": [2.0],
            "flops": [1e6],
            "dram_read_bytes": [1024e6],
            "dram_write_bytes": [64e6],
            "memory_access_pattern": ["irregular"],
            "notes": [""],
        }
    )
    profile = add_roofline_metrics(profile, hardware)
    row = profile.iloc[0].copy()
    row["ncu_sol_dram_pct"] = 80.0
    row["ncu_memory_util_pct"] = 80.0
    row["ncu_sm_util_pct"] = 10.0
    row["ncu_l1_hit_rate_pct"] = 10.0
    row["ncu_l2_hit_rate_pct"] = 18.0
    row["ncu_global_load_efficiency_pct"] = 52.0
    row["ncu_global_store_efficiency_pct"] = 70.0
    row["ncu_warp_execution_efficiency_pct"] = 90.0
    row["ncu_branch_efficiency_pct"] = 95.0
    row["ncu_long_scoreboard_stall_pct"] = 36.0

    estimate = estimate_feature_cost_v6(row, hardware)

    assert "cache_hit=0.18" in str(estimate["feature_summary"])
    assert "latency_stall=0.36" in str(estimate["feature_summary"])
    assert float(estimate["estimated_speedup"]) > 0.0


def test_feature_cost_v6_allows_collective_memory_primitive_override() -> None:
    hardware = HardwareConfig("test", peak_flops=10e12, peak_memory_bandwidth=1e12)
    profile = pd.DataFrame(
        {
            "kernel_name": ["reduction"],
            "runtime_ms": [1.0],
            "flops": [1e6],
            "dram_read_bytes": [512e6],
            "dram_write_bytes": [4e6],
            "memory_access_pattern": ["regular"],
            "notes": [""],
        }
    )
    profile = add_roofline_metrics(profile, hardware)
    row = profile.iloc[0].copy()
    row["ncu_sol_dram_pct"] = 0.5
    row["ncu_l1_hit_rate_pct"] = 0.0
    row["ncu_l2_hit_rate_pct"] = 16.0
    row["ncu_long_scoreboard_stall_pct"] = 88.0
    row["ncu_eligible_warps_per_scheduler"] = 0.04
    metadata = pd.Series(
        {
            "paper_notes": "Reduction is a bandwidth-sensitive parallel primitive with synchronization phases.",
        }
    )

    estimate = estimate_feature_cost_v6(row, hardware, metadata)

    assert estimate["predicted_candidate"] is True
    assert "collective memory primitive" in str(estimate["risk"])
