from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


NCU_METRIC_COLUMNS = [
    "ncu_duration_us",
    "ncu_memory_util_pct",
    "ncu_memory_throughput_gbps",
    "ncu_mem_busy_pct",
    "ncu_max_bandwidth_pct",
    "ncu_mem_pipes_busy_pct",
    "ncu_sol_dram_pct",
    "ncu_sol_l1_tex_pct",
    "ncu_sol_l2_pct",
    "ncu_sm_util_pct",
    "ncu_achieved_occupancy_pct",
    "ncu_active_warps_per_sm",
    "ncu_registers_per_thread",
    "ncu_l1_hit_rate_pct",
    "ncu_l2_hit_rate_pct",
    "ncu_global_load_efficiency_pct",
    "ncu_global_store_efficiency_pct",
    "ncu_warp_execution_efficiency_pct",
    "ncu_branch_efficiency_pct",
    "ncu_issue_slot_util_pct",
    "ncu_one_or_more_eligible_pct",
    "ncu_no_eligible_pct",
    "ncu_issued_warp_per_scheduler",
    "ncu_active_warps_per_scheduler",
    "ncu_eligible_warps_per_scheduler",
    "ncu_warp_cycles_per_issued_instruction",
    "ncu_warp_cycles_per_executed_instruction",
    "ncu_avg_active_threads_per_warp",
    "ncu_avg_not_predicated_threads_per_warp",
    "ncu_memory_stall_pct",
    "ncu_barrier_stall_pct",
    "ncu_long_scoreboard_stall_pct",
    "ncu_short_scoreboard_stall_pct",
    "ncu_dram_read_bytes",
    "ncu_dram_write_bytes",
    "ncu_l2_read_transactions",
    "ncu_l2_write_transactions",
]

_TEXT_METRICS = {
    "ncu_duration_us": ["Duration"],
    "ncu_memory_util_pct": ["Memory [%]", "Mem Busy"],
    "ncu_memory_throughput_gbps": ["Memory Throughput"],
    "ncu_mem_busy_pct": ["Mem Busy"],
    "ncu_max_bandwidth_pct": ["Max Bandwidth"],
    "ncu_mem_pipes_busy_pct": ["Mem Pipes Busy"],
    "ncu_sol_dram_pct": ["SOL DRAM", "Max Bandwidth"],
    "ncu_sol_l1_tex_pct": ["SOL L1/TEX Cache"],
    "ncu_sol_l2_pct": ["SOL L2 Cache"],
    "ncu_sm_util_pct": ["SM [%]"],
    "ncu_achieved_occupancy_pct": ["Achieved Occupancy"],
    "ncu_active_warps_per_sm": ["Achieved Active Warps Per SM"],
    "ncu_registers_per_thread": ["Registers Per Thread"],
    "ncu_l1_hit_rate_pct": ["L1/TEX Hit Rate", "L1 Hit Rate", "l1tex hit rate"],
    "ncu_l2_hit_rate_pct": ["L2 Hit Rate", "lts hit rate"],
    "ncu_global_load_efficiency_pct": ["Global Memory Load Efficiency", "Global Load Efficiency"],
    "ncu_global_store_efficiency_pct": ["Global Memory Store Efficiency", "Global Store Efficiency"],
    "ncu_warp_execution_efficiency_pct": ["Warp Execution Efficiency"],
    "ncu_branch_efficiency_pct": ["Branch Efficiency"],
    "ncu_issue_slot_util_pct": ["Issue Slots Busy", "Issue Slot Utilization"],
    "ncu_one_or_more_eligible_pct": ["One or More Eligible"],
    "ncu_no_eligible_pct": ["No Eligible"],
    "ncu_issued_warp_per_scheduler": ["Issued Warp Per Scheduler"],
    "ncu_active_warps_per_scheduler": ["Active Warps Per Scheduler"],
    "ncu_eligible_warps_per_scheduler": ["Eligible Warps Per Scheduler"],
    "ncu_warp_cycles_per_issued_instruction": ["Warp Cycles Per Issued Instruction"],
    "ncu_warp_cycles_per_executed_instruction": ["Warp Cycles Per Executed Instruction"],
    "ncu_avg_active_threads_per_warp": ["Avg. Active Threads Per Warp"],
    "ncu_avg_not_predicated_threads_per_warp": ["Avg. Not Predicated Off Threads Per Warp"],
    "ncu_memory_stall_pct": ["Stall Memory Dependency", "Stall Memory Throttle", "Memory Stall"],
    "ncu_barrier_stall_pct": ["Stall Barrier", "Barrier Stall"],
    "ncu_long_scoreboard_stall_pct": ["Stall Long Scoreboard", "Long Scoreboard"],
    "ncu_short_scoreboard_stall_pct": ["Stall Short Scoreboard", "Short Scoreboard"],
    "ncu_dram_read_bytes": ["DRAM Read Bytes", "dram bytes read"],
    "ncu_dram_write_bytes": ["DRAM Write Bytes", "dram bytes write", "dram bytes written"],
    "ncu_l2_read_transactions": ["L2 Read Transactions", "L2 Read Sectors"],
    "ncu_l2_write_transactions": ["L2 Write Transactions", "L2 Write Sectors"],
}

_CSV_METRIC_ALIASES = {
    "ncu_l1_hit_rate_pct": ["l1tex", "hit_rate"],
    "ncu_l2_hit_rate_pct": ["lts", "hit_rate"],
    "ncu_warp_execution_efficiency_pct": ["warp_execution_efficiency"],
    "ncu_branch_efficiency_pct": ["branch_efficiency"],
    "ncu_dram_read_bytes": ["dram", "bytes_read"],
    "ncu_dram_write_bytes": ["dram", "bytes_write"],
}


def load_ncu_metrics_csv(path: str | Path) -> pd.DataFrame:
    """Load normalized Nsight Compute metrics for analyzer merge."""

    data = pd.read_csv(path)
    if "kernel_name" not in data.columns:
        raise ValueError("NCU metrics CSV must contain kernel_name")

    for column in NCU_METRIC_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data[["kernel_name", *NCU_METRIC_COLUMNS]]


def attach_ncu_metrics(profile: pd.DataFrame, metrics: pd.DataFrame | None) -> pd.DataFrame:
    if metrics is None:
        return profile

    merged = profile.copy()
    merged["_normalized_kernel_name"] = merged["kernel_name"].map(_normalize_name)
    normalized_metrics = metrics.copy()
    normalized_metrics["_normalized_kernel_name"] = normalized_metrics["kernel_name"].map(_normalize_name)
    normalized_metrics = normalized_metrics.drop(columns=["kernel_name"])
    normalized_metrics = normalized_metrics.drop_duplicates("_normalized_kernel_name", keep="first")
    merged = merged.merge(normalized_metrics, on="_normalized_kernel_name", how="left")
    return merged.drop(columns=["_normalized_kernel_name"])


def parse_ncu_report_file(path: str | Path, kernel_name: str | None = None) -> dict[str, object]:
    """Parse either text or CSV output generated by Nsight Compute.

    Older Nsight Compute versions emit several CSV dialects depending on flags.
    This parser intentionally accepts both the human-readable text table and
    CSV rows that contain a known metric label plus a numeric value.
    """

    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8", errors="replace")
    row: dict[str, object] = {"kernel_name": kernel_name or _kernel_name_from_path(report_path)}
    row.update({column: pd.NA for column in NCU_METRIC_COLUMNS})

    _parse_text_metrics(text, row)
    _parse_warning_metrics(text, row)
    _parse_csv_like_metrics(text, row)
    return row


def _parse_text_metrics(text: str, row: dict[str, object]) -> None:
    for column, labels in _TEXT_METRICS.items():
        for label in labels:
            values = []
            for line in text.splitlines():
                stripped = line.strip()
                if not _starts_with_any_label(stripped, [label]):
                    continue
                numeric_values = _numeric_values(stripped)
                if numeric_values:
                    values.append(numeric_values[-1])
            if values:
                row[column] = sum(values) / len(values)
                break


def _parse_csv_like_metrics(text: str, row: dict[str, object]) -> None:
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 2:
            continue
        joined = " ".join(field.strip() for field in fields)
        normalized = _normalize_metric_text(joined)
        for column, labels in _TEXT_METRICS.items():
            if not _matches_any_label(joined, labels) and not _matches_csv_alias(normalized, column):
                continue
            numeric_values = _numeric_values(joined)
            if numeric_values:
                row[column] = numeric_values[-1]


def _parse_warning_metrics(text: str, row: dict[str, object]) -> None:
    scoreboard_match = re.search(r"represents about\s+(\d+(?:\.\d+)?)%\s+of the total", text)
    if scoreboard_match:
        row["ncu_long_scoreboard_stall_pct"] = float(scoreboard_match.group(1))


def _numeric_values(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", text)]


def _matches_any_label(text: str, labels: list[str]) -> bool:
    normalized = _normalize_metric_text(text)
    return any(_normalize_metric_text(label) in normalized for label in labels)


def _starts_with_any_label(text: str, labels: list[str]) -> bool:
    lowered = text.lower()
    return any(lowered.startswith(label.lower()) for label in labels)


def _matches_csv_alias(text: str, column: str) -> bool:
    aliases = _CSV_METRIC_ALIASES.get(column)
    if not aliases:
        return False
    return all(alias in text for alias in aliases)


def _normalize_metric_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _kernel_name_from_path(path: Path) -> str:
    name = path.name
    for suffix in ("_ncu.txt", "_ncu.csv", ".ncu.txt", ".ncu.csv", ".txt", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")
