import argparse
from pathlib import Path

from src.baseline import (
    compare_to_paper_baseline,
    load_paper_baseline_csv,
    summarize_baseline_alignment,
)
from src.classifier import add_classifications
from src.config import DEFAULT_HARDWARE, HardwareConfig
from src.end_to_end import build_end_to_end_evaluation
from src.model_comparison import build_model_comparison
from src.ncu_metrics import attach_ncu_metrics, load_ncu_metrics_csv
from src.parser import load_profile_csv
from src.plot import save_end_to_end_plot, save_model_comparison_plot, save_roofline_plot
from src.report import build_summary_table, save_markdown_report
from src.roofline import add_roofline_metrics
from src.simulator import attach_simulation_results, load_pim_simulation_csv, summarize_simulation_coverage


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_profile.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PRIM_BASELINE_PATH = BASE_DIR / "paper_baselines" / "prim2022_workloads.csv"


def main() -> None:
    args = parse_args()
    if args.pim_simulation and not args.paper_baseline:
        raise ValueError("--pim-simulation requires --paper-baseline so simulator rows can be matched to benchmarks")
    hardware = HardwareConfig(
        name=args.hardware_name,
        peak_flops=args.peak_flops,
        peak_memory_bandwidth=args.peak_memory_bandwidth,
    )
    figure_path = args.output_dir / "figures" / "roofline.png"
    model_figure_path = args.output_dir / "figures" / "model_comparison.png"
    end_to_end_figure_path = args.output_dir / "figures" / "end_to_end.png"
    report_path = args.output_dir / "reports" / "analysis_report.md"

    profile = load_profile_csv(args.input)
    profile = add_roofline_metrics(profile, hardware)
    profile = add_classifications(profile, hardware)
    ncu_metrics = load_ncu_metrics_csv(args.ncu_metrics) if args.ncu_metrics else None
    profile = attach_ncu_metrics(profile, ncu_metrics)

    baseline_comparison = None
    model_comparison = None
    model_metrics = None
    end_to_end = None
    simulation_summary = None
    if args.paper_baseline:
        baseline = load_paper_baseline_csv(args.paper_baseline)
        baseline_comparison = compare_to_paper_baseline(profile, baseline)
        model_comparison, model_metrics = build_model_comparison(profile, baseline, hardware)
        simulation = load_pim_simulation_csv(args.pim_simulation) if args.pim_simulation else None
        model_comparison = attach_simulation_results(model_comparison, simulation)
        simulation_summary = summarize_simulation_coverage(model_comparison)
        runtime_source = "simulated" if simulation is not None else "estimated"
        end_to_end_cost_model = (
            "feature_cost_v6"
            if "feature_cost_v6" in set(model_comparison["model"])
            else "feature_cost_v5"
            if "feature_cost_v5" in set(model_comparison["model"])
            else "feature_cost_v4"
        )
        end_to_end = build_end_to_end_evaluation(
            model_comparison,
            cost_model=end_to_end_cost_model,
            runtime_source=runtime_source,
        )

    save_roofline_plot(profile, hardware, figure_path)
    if model_comparison is not None and model_metrics is not None:
        save_model_comparison_plot(model_metrics, model_comparison, model_figure_path)
    if end_to_end is not None:
        save_end_to_end_plot(end_to_end, end_to_end_figure_path)
    save_markdown_report(
        profile,
        hardware,
        figure_path,
        report_path,
        baseline_comparison,
        model_comparison,
        model_metrics,
        model_figure_path if model_comparison is not None else None,
        end_to_end,
        end_to_end_figure_path if end_to_end is not None else None,
        simulation_summary,
    )

    print("\nGPU Workload Bottleneck Analyzer")
    print("=" * 34)
    print(f"Hardware: {hardware.name}")
    print(f"Ridge point: {hardware.ridge_point:.4f} FLOPs/Byte")
    print(f"Input: {args.input}")
    print()
    print(build_summary_table(profile).to_string(index=False))
    if baseline_comparison is not None:
        alignment = summarize_baseline_alignment(baseline_comparison)
        print()
        print("Paper baseline alignment")
        print("-" * 24)
        print(
            f"Baseline profiled: {alignment['profiled']}/{alignment['benchmarks']}, "
            f"matches: {alignment['matches']}, "
            f"match rate: {alignment['match_rate']:.1%}"
        )
    if model_metrics is not None:
        print()
        print("Model comparison")
        print("-" * 16)
        print(model_metrics.to_string(index=False, formatters=_metric_formatters()))
    if end_to_end is not None:
        print()
        print("End-to-end policy estimate")
        print("-" * 29)
        print(end_to_end.to_string(index=False, formatters=_end_to_end_formatters()))
    if simulation_summary is not None and simulation_summary["simulated"]:
        print()
        print("PIM simulation coverage")
        print("-" * 23)
        print(
            f"Simulated: {simulation_summary['simulated']}/{simulation_summary['benchmarks']}, "
            f"coverage: {simulation_summary['coverage']:.1%}, "
            f"simulator: {simulation_summary['simulators']}"
        )
    print()
    print(f"Saved roofline plot: {figure_path}")
    print(f"Saved report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GPU workload bottlenecks with a Roofline model.")
    parser.add_argument("--input", type=Path, default=DATA_PATH, help="Profiling CSV path.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated reports and figures.")
    parser.add_argument("--hardware-name", default=DEFAULT_HARDWARE.name, help="Target hardware label.")
    parser.add_argument("--peak-flops", type=float, default=DEFAULT_HARDWARE.peak_flops, help="Peak compute in FLOP/s.")
    parser.add_argument(
        "--peak-memory-bandwidth",
        type=float,
        default=DEFAULT_HARDWARE.peak_memory_bandwidth,
        help="Peak memory bandwidth in bytes/s.",
    )
    parser.add_argument(
        "--paper-baseline",
        type=Path,
        default=None,
        help=f"Optional paper workload baseline CSV. Example: {PRIM_BASELINE_PATH}",
    )
    parser.add_argument(
        "--pim-simulation",
        type=Path,
        default=None,
        help="Optional PIM simulator output CSV with kernel_name, simulator, simulated_pim_time_ms.",
    )
    parser.add_argument(
        "--ncu-metrics",
        type=Path,
        default=None,
        help="Optional Nsight Compute metrics CSV generated by scripts/parse_ncu_reports.py.",
    )
    return parser.parse_args()


def _metric_formatters() -> dict[str, object]:
    return {
        "precision": lambda value: f"{value:.2f}",
        "recall": lambda value: f"{value:.2f}",
        "f1": lambda value: f"{value:.2f}",
        "accuracy": lambda value: f"{value:.2f}",
    }


def _end_to_end_formatters() -> dict[str, object]:
    return {
        "total_runtime_ms": lambda value: f"{value:.3f}",
        "speedup_vs_gpu": lambda value: f"{value:.2f}x",
        "runtime_reduction_pct": lambda value: f"{value:.1f}%",
    }


if __name__ == "__main__":
    main()
