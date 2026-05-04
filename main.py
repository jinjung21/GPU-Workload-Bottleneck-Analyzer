import argparse
from pathlib import Path

from src.classifier import add_classifications
from src.config import DEFAULT_HARDWARE, HardwareConfig
from src.parser import load_profile_csv
from src.plot import save_roofline_plot
from src.report import build_summary_table, save_markdown_report
from src.roofline import add_roofline_metrics


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_profile.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    args = parse_args()
    hardware = HardwareConfig(
        name=args.hardware_name,
        peak_flops=args.peak_flops,
        peak_memory_bandwidth=args.peak_memory_bandwidth,
    )
    figure_path = args.output_dir / "figures" / "roofline.png"
    report_path = args.output_dir / "reports" / "analysis_report.md"

    profile = load_profile_csv(args.input)
    profile = add_roofline_metrics(profile, hardware)
    profile = add_classifications(profile, hardware)

    save_roofline_plot(profile, hardware, figure_path)
    save_markdown_report(profile, hardware, figure_path, report_path)

    print("\nGPU Workload Bottleneck Analyzer")
    print("=" * 34)
    print(f"Hardware: {hardware.name}")
    print(f"Ridge point: {hardware.ridge_point:.4f} FLOPs/Byte")
    print(f"Input: {args.input}")
    print()
    print(build_summary_table(profile).to_string(index=False))
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
