from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import HardwareConfig


def format_flops(value: float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f} TFLOP/s"
    return f"{value / 1e9:.2f} GFLOP/s"


def format_bandwidth(value: float) -> str:
    return f"{value / 1e9:.2f} GB/s"


def build_summary_table(profile: pd.DataFrame) -> pd.DataFrame:
    """Create a display-oriented dataframe for reports and terminal output."""

    return pd.DataFrame(
        {
            "kernel": profile["kernel_name"],
            "runtime_ms": profile["runtime_ms"].map(lambda value: f"{value:.2f}"),
            "AI_FLOP_per_Byte": profile["arithmetic_intensity"].map(lambda value: f"{value:.4f}"),
            "achieved_perf": profile["achieved_flops"].map(format_flops),
            "achieved_bw": profile["achieved_bandwidth"].map(format_bandwidth),
            "roofline_util": profile["roofline_utilization_raw"].map(lambda value: f"{value:.2%}"),
            "over_roofline": profile["exceeds_roofline"].map(lambda value: "yes" if value else "no"),
            "bottleneck": profile["bottleneck_classification"],
            "PIM/NMP": profile["pim_nmp_suitability"],
            "PIM_score": profile["pim_nmp_score"],
            "score_reason": profile["pim_nmp_score_reason"],
            "recommendation": profile["recommendation"],
        }
    )


def save_markdown_report(
    profile: pd.DataFrame,
    hardware: HardwareConfig,
    figure_path: str | Path,
    output_path: str | Path,
    baseline_comparison: pd.DataFrame | None = None,
    model_comparison: pd.DataFrame | None = None,
    model_metrics: pd.DataFrame | None = None,
    model_figure_path: str | Path | None = None,
    end_to_end: pd.DataFrame | None = None,
    end_to_end_figure_path: str | Path | None = None,
    simulation_summary: dict[str, int | float | str] | None = None,
) -> None:
    """Write a markdown analysis report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Path(figure_path)
    figure_link = _relative_link(figure, output.parent)
    summary = build_summary_table(profile)
    baseline_section = _build_baseline_section(baseline_comparison)
    model_section = _build_model_comparison_section(model_comparison, model_metrics, model_figure_path, output.parent)
    end_to_end_section = _build_end_to_end_section(end_to_end, end_to_end_figure_path, output.parent)
    simulation_section = _build_simulation_section(simulation_summary)
    exceeded = profile[profile["exceeds_roofline"]]
    warnings = []
    if not exceeded.empty:
        kernels = ", ".join(exceeded["kernel_name"].astype(str))
        warnings.extend(
            [
                "## Data Quality Warnings",
                "",
                f"- The following kernels exceed the modeled Roofline limit and may need unit/config validation: {kernels}",
                "",
            ]
        )

    report = [
        "# GPU Workload Bottleneck Analysis Report",
        "",
        "## Hardware Config",
        "",
        f"- Target: {hardware.name}",
        f"- Peak compute: {format_flops(hardware.peak_flops)}",
        f"- Peak memory bandwidth: {format_bandwidth(hardware.peak_memory_bandwidth)}",
        f"- Ridge point: {hardware.ridge_point:.4f} FLOPs/Byte",
        "",
        "## Roofline Plot",
        "",
        f"![Roofline Plot]({figure_link})",
        "",
        *warnings,
        "## Kernel Summary",
        "",
        summary.to_markdown(index=False),
        "",
        *baseline_section,
        *model_section,
        *simulation_section,
        *end_to_end_section,
        "## Notes",
        "",
        "- Reports can be generated from proxy CSVs or CUDA benchmark profile CSVs.",
        "- PIM/NMP suitability still includes heuristic components that require calibration.",
        "- Nsight Compute counter-based parsing can be added when performance counter permissions are available.",
        "- Paper baseline comparison checks workload-category alignment, not measured PIM speedup.",
        "- Analytical PIM and end-to-end speedup estimates are model outputs, not hardware measurements.",
        "",
    ]
    output.write_text("\n".join(report), encoding="utf-8")


def _build_baseline_section(baseline_comparison: pd.DataFrame | None) -> list[str]:
    if baseline_comparison is None:
        return []

    profiled = baseline_comparison[baseline_comparison["model_alignment"] != "not profiled"]
    matches = profiled[profiled["model_alignment"] == "match"]
    match_rate = 0.0 if len(profiled) == 0 else len(matches) / len(profiled)
    sources = ", ".join(sorted(set(baseline_comparison["paper"].astype(str))))
    table_columns = [
        "benchmark",
        "domain",
        "paper_expected",
        "expected_min_score",
        "our_kernel",
        "our_pim_label",
        "our_score",
        "model_alignment",
    ]

    return [
        "## Paper Baseline Comparison",
        "",
        f"- Baseline source: {sources}.",
        f"- Coverage: {len(profiled)}/{len(baseline_comparison)} paper workloads profiled in this run.",
        f"- Alignment: {len(matches)}/{len(profiled)} matched expected score floors ({match_rate:.1%}).",
        "",
        baseline_comparison[table_columns].to_markdown(index=False),
        "",
    ]


def _build_model_comparison_section(
    model_comparison: pd.DataFrame | None,
    model_metrics: pd.DataFrame | None,
    model_figure_path: str | Path | None,
    report_dir: Path,
) -> list[str]:
    if model_comparison is None or model_metrics is None:
        return []

    models = set(model_comparison["model"])
    speedup_model = "feature_cost_v4" if "feature_cost_v4" in models else "feature_cost_v3"
    if speedup_model not in models:
        speedup_model = "analytical_v2"
    speedup = model_comparison[model_comparison["model"] == speedup_model].copy()
    speedup = speedup.sort_values("estimated_speedup", ascending=False)
    speedup_columns = [
        "benchmark",
        "target_candidate",
        "predicted_candidate",
        "estimated_speedup",
        "estimated_pim_time_ms",
        "risk",
        "feature_summary",
    ]
    metric_table = model_metrics.copy()
    for column in ["precision", "recall", "f1", "accuracy"]:
        metric_table[column] = metric_table[column].map(lambda value: f"{value:.2f}")
    speedup["estimated_speedup"] = speedup["estimated_speedup"].map(lambda value: f"{value:.2f}x")
    speedup["estimated_pim_time_ms"] = speedup["estimated_pim_time_ms"].map(lambda value: f"{value:.3f}")

    figure_lines = []
    if model_figure_path is not None:
        figure_link = _relative_link(Path(model_figure_path), report_dir)
        figure_lines = ["", f"![Model Comparison]({figure_link})", ""]

    return [
        "## Model Comparison",
        "",
        "The table compares simple baselines against analytical and feature-cost PIM/NMP candidate models.",
        *figure_lines,
        "",
        metric_table.to_markdown(index=False),
        "",
        f"### {speedup_model} PIM/NMP Estimates",
        "",
        speedup[speedup_columns].to_markdown(index=False),
        "",
    ]


def _build_end_to_end_section(
    end_to_end: pd.DataFrame | None,
    figure_path: str | Path | None,
    report_dir: Path,
) -> list[str]:
    if end_to_end is None or end_to_end.empty:
        return []

    table = end_to_end.copy()
    table["total_runtime_ms"] = table["total_runtime_ms"].map(lambda value: f"{value:.3f}")
    table["speedup_vs_gpu"] = table["speedup_vs_gpu"].map(lambda value: f"{value:.2f}x")
    table["runtime_reduction_pct"] = table["runtime_reduction_pct"].map(lambda value: f"{value:.1f}%")
    columns = [
        "model",
        "total_runtime_ms",
        "speedup_vs_gpu",
        "runtime_reduction_pct",
        "offloaded_kernels",
        "false_offloads",
        "missed_candidates",
        "runtime_source",
    ]

    figure_lines = []
    if figure_path is not None:
        figure_link = _relative_link(Path(figure_path), report_dir)
        figure_lines = ["", f"![End-to-End Runtime](../{figure_link})" if not figure_link.startswith("..") else f"![End-to-End Runtime]({figure_link})", ""]

    return [
        "## End-to-End Policy Estimate",
        "",
        "This section estimates total workload runtime by applying each offload policy to the same kernels. Candidate kernels use a common PIM/NMP runtime source; non-candidates keep measured GPU runtime.",
        *figure_lines,
        "",
        table[columns].to_markdown(index=False),
        "",
    ]


def _build_simulation_section(summary: dict[str, int | float | str] | None) -> list[str]:
    if not summary or int(summary.get("simulated", 0)) == 0:
        return []
    return [
        "## PIM Simulation Input",
        "",
        f"- Simulator source: {summary.get('simulators', '')}.",
        f"- Coverage: {summary.get('simulated', 0)}/{summary.get('benchmarks', 0)} profiled benchmarks have simulated PIM runtime ({summary.get('coverage', 0.0):.1%}).",
        "- End-to-end policy estimates use simulated PIM runtime for offloaded kernels.",
        "",
    ]


def _relative_link(target: Path, base_dir: Path) -> str:
    try:
        return target.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        import os

        return os.path.relpath(target.resolve(), base_dir.resolve()).replace("\\", "/")
