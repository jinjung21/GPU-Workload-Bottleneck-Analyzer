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
) -> None:
    """Write a markdown analysis report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Path(figure_path)
    figure_link = _relative_link(figure, output.parent)
    summary = build_summary_table(profile)
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
        "## Notes",
        "",
        "- This report uses fake profiling data and does not require CUDA or NVIDIA tools.",
        "- PIM/NMP suitability is a rule-based heuristic for early-stage research prototyping.",
        "- The parser is intentionally separated so Nsight Compute CSV support can be added later.",
        "",
    ]
    output.write_text("\n".join(report), encoding="utf-8")


def _relative_link(target: Path, base_dir: Path) -> str:
    try:
        return target.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        import os

        return os.path.relpath(target.resolve(), base_dir.resolve()).replace("\\", "/")
