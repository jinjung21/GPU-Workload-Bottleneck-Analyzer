#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import load_paper_baseline_csv
from src.classifier import add_classifications
from src.config import HardwareConfig
from src.model_comparison import build_model_comparison
from src.parser import load_profile_csv
from src.roofline import add_roofline_metrics


def main() -> None:
    args = parse_args()
    hardware = HardwareConfig(args.hardware_name, args.peak_flops, args.peak_memory_bandwidth)
    baseline = load_paper_baseline_csv(args.paper_baseline)
    benchmark_order = {str(name): index for index, name in enumerate(baseline["benchmark"])}
    rows = []

    for profile_path in sorted(args.input_dir.glob("gpu_profile_*.csv")):
        scale = _scale_from_path(profile_path)
        profile = load_profile_csv(profile_path)
        profile = add_roofline_metrics(profile, hardware)
        profile = add_classifications(profile, hardware)
        model_comparison, _ = build_model_comparison(profile, baseline, hardware)
        decision_model = _decision_model(model_comparison)
        decisions = model_comparison[model_comparison["model"] == decision_model].set_index("benchmark")

        for _, row in profile.iterrows():
            benchmark = str(row["kernel_name"])
            decision = decisions.loc[benchmark] if benchmark in decisions.index else None
            rows.append(
                {
                    "scale": scale,
                    "benchmark": benchmark,
                    "_benchmark_order": benchmark_order.get(benchmark, 999),
                    "runtime_ms": float(row["runtime_ms"]),
                    "arithmetic_intensity": float(row["arithmetic_intensity"]),
                    "achieved_bandwidth_gbps": float(row["achieved_bandwidth"]) / 1e9,
                    "roofline_utilization": float(row["roofline_utilization_raw"]),
                    "bottleneck": row["bottleneck_classification"],
                    "pim_score": int(row["pim_nmp_score"]),
                    "decision_model": decision_model,
                    "final_decision": _final_decision(decision),
                    "estimated_speedup": _decision_value(decision, "estimated_speedup"),
                    "notes": row["notes"],
                }
            )

    summary = pd.DataFrame(rows)
    summary["_scale_order"] = summary["scale"].map({"small": 0, "medium": 1, "large": 2}).fillna(99)
    summary = summary.sort_values(["_scale_order", "_benchmark_order"]).drop(columns=["_scale_order", "_benchmark_order"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    args.output_markdown.write_text(_markdown_summary(summary), encoding="utf-8")
    print(f"Wrote size sweep summary CSV: {args.output_csv}")
    print(f"Wrote size sweep summary Markdown: {args.output_markdown}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark size sweep outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("profiles/size_sweep"))
    parser.add_argument("--paper-baseline", type=Path, default=Path("paper_baselines/gpu_benchmark_metadata.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/size_sweep/size_sweep_summary.csv"))
    parser.add_argument("--output-markdown", type=Path, default=Path("outputs/size_sweep/size_sweep_summary.md"))
    parser.add_argument("--hardware-name", default="RTX 2080 Ti")
    parser.add_argument("--peak-flops", type=float, default=13.45e12)
    parser.add_argument("--peak-memory-bandwidth", type=float, default=616e9)
    return parser.parse_args()


def _scale_from_path(path: Path) -> str:
    match = re.match(r"gpu_profile_(.+)\.csv", path.name)
    return match.group(1) if match else path.stem


def _decision_model(model_comparison: pd.DataFrame) -> str:
    models = set(model_comparison["model"])
    for model in ["feature_cost_v6", "feature_cost_v5", "feature_cost_v4", "feature_cost_v3", "analytical_v2"]:
        if model in models:
            return model
    raise ValueError("No supported decision model in size sweep comparison")


def _final_decision(decision: pd.Series | None) -> str:
    if decision is None:
        return ""
    return "PIM/NMP" if bool(decision["predicted_candidate"]) else "GPU"


def _decision_value(decision: pd.Series | None, column: str) -> float | str:
    if decision is None:
        return ""
    return float(decision[column])


def _markdown_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    display["runtime_ms"] = display["runtime_ms"].map(lambda value: f"{value:.3f}")
    display["arithmetic_intensity"] = display["arithmetic_intensity"].map(lambda value: f"{value:.4f}")
    display["achieved_bandwidth_gbps"] = display["achieved_bandwidth_gbps"].map(lambda value: f"{value:.2f}")
    display["roofline_utilization"] = display["roofline_utilization"].map(lambda value: f"{value:.2%}")
    display["estimated_speedup"] = display["estimated_speedup"].map(
        lambda value: "" if value == "" else f"{float(value):.2f}x"
    )
    columns = [
        "scale",
        "benchmark",
        "runtime_ms",
        "arithmetic_intensity",
        "achieved_bandwidth_gbps",
        "bottleneck",
        "pim_score",
        "final_decision",
        "estimated_speedup",
    ]
    return "\n".join(
        [
            "# GPU Benchmark Size Sweep Summary",
            "",
            "This table compares benchmark behavior across small, medium, and large input sizes.",
            "",
            display[columns].to_markdown(index=False),
            "",
        ]
    )


if __name__ == "__main__":
    main()
