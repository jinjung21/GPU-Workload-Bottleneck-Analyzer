from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gpu-bottleneck-analyzer-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gpu-bottleneck-analyzer-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import pandas as pd

from .config import HardwareConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_roofline_plot(profile: pd.DataFrame, hardware: HardwareConfig, output_path: str | Path) -> None:
    """Generate a Roofline plot and save it as a PNG file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    min_intensity = max(profile["arithmetic_intensity"].min() / 5, 1e-3)
    max_intensity = max(profile["arithmetic_intensity"].max() * 5, hardware.ridge_point * 2)

    intensities = _logspace(min_intensity, max_intensity, points=200)
    memory_roof = [hardware.peak_memory_bandwidth * intensity for intensity in intensities]
    compute_roof = [hardware.peak_flops for _ in intensities]
    roofline = [min(mem, comp) for mem, comp in zip(memory_roof, compute_roof)]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(intensities, roofline, label="Roofline limit", color="#1f77b4", linewidth=2.5)
    ax.axvline(hardware.ridge_point, color="#666666", linestyle="--", linewidth=1.2, label="Ridge point")
    ax.scatter(
        profile["arithmetic_intensity"],
        profile["achieved_flops"],
        s=85,
        color="#d62728",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="Kernels",
    )

    for _, row in profile.iterrows():
        ax.annotate(
            row["kernel_name"],
            (row["arithmetic_intensity"], row["achieved_flops"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("GPU Workload Roofline Analysis")
    ax.set_xlabel("Arithmetic intensity (FLOPs / Byte)")
    ax.set_ylabel("Performance (FLOP/s)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_model_comparison_plot(
    model_metrics: pd.DataFrame,
    model_comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a compact plot for model quality and analytical speedup estimates."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics = model_metrics.copy().sort_values("f1", ascending=False)
    models = set(model_comparison["model"])
    speedup_model = "feature_cost_v4" if "feature_cost_v4" in models else "feature_cost_v3"
    if speedup_model not in models:
        speedup_model = "analytical_v2"
    speedups = model_comparison[model_comparison["model"] == speedup_model].copy()
    speedups["estimated_speedup"] = pd.to_numeric(speedups["estimated_speedup"], errors="coerce")
    speedups = speedups.sort_values("estimated_speedup", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = range(len(metrics))
    width = 0.24
    axes[0].bar([value - width for value in x], metrics["precision"], width=width, label="Precision", color="#2ca02c")
    axes[0].bar(x, metrics["recall"], width=width, label="Recall", color="#1f77b4")
    axes[0].bar([value + width for value in x], metrics["f1"], width=width, label="F1", color="#d62728")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(metrics["model"], rotation=20, ha="right")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Candidate Selection Quality")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.6)
    axes[0].legend()

    colors = ["#2ca02c" if candidate else "#7f7f7f" for candidate in speedups["predicted_candidate"]]
    axes[1].barh(speedups["benchmark"], speedups["estimated_speedup"], color=colors)
    axes[1].axvline(1.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"{speedup_model} PIM/NMP Speedup Estimate")
    axes[1].set_xlabel("Estimated speedup vs GPU proxy runtime")
    axes[1].grid(axis="x", linestyle=":", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _logspace(start: float, stop: float, points: int) -> list[float]:
    """Small helper to avoid adding NumPy as a direct dependency."""

    import math

    log_start = math.log10(start)
    log_stop = math.log10(stop)
    step = (log_stop - log_start) / (points - 1)
    return [10 ** (log_start + step * i) for i in range(points)]
