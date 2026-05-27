#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_MAPPINGS = {
    "add": ["vector_add", "saxpy", "reduction"],
    "gemv": ["gemv"],
}


def main() -> None:
    args = parse_args()
    rows = []
    for primitive, kernels in DEFAULT_MAPPINGS.items():
        log_path = args.log_dir / f"pimbench_{primitive}.log"
        if not log_path.exists():
            continue
        parsed = parse_log(log_path)
        for kernel in kernels:
            rows.append(
                {
                    "kernel_name": kernel,
                    "simulator": "SAIT-PIMSimulator",
                    "simulated_pim_time_ms": parsed["pim_cycles"] * args.cycle_time_ns / 1e6,
                    "simulated_pim_cycles": parsed["pim_cycles"],
                    "simulated_baseline_cycles": parsed["baseline_cycles"],
                    "simulated_speedup": parsed["speedup"],
                    "cycle_time_ns": args.cycle_time_ns,
                    "notes": (
                        f"PIMBenchFixture.{primitive}; "
                        f"PIM enabled cycle={parsed['pim_cycles']}; "
                        f"PIM disabled cycle={parsed['baseline_cycles']}; "
                        f"{args.cycle_time_ns:g} ns/cycle conversion"
                    ),
                }
            )

    if not rows:
        raise SystemExit(f"No SAIT PIMSimulator logs found in {args.log_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} simulator rows to {args.output}")


def parse_log(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cycles = [int(value) for value in re.findall(r">\s*Cycle\s*:\s*(\d+)", text)]
    if len(cycles) < 2:
        raise ValueError(f"Expected at least two cycle lines in {path}")
    speedup_match = re.search(r">\s*Speed-up\s*:\s*([0-9.]+)", text)
    baseline_cycles, pim_cycles = cycles[0], cycles[1]
    speedup = float(speedup_match.group(1)) if speedup_match else baseline_cycles / pim_cycles
    return {
        "baseline_cycles": baseline_cycles,
        "pim_cycles": pim_cycles,
        "speedup": speedup,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SAIT PIMSimulator benchmark logs to analyzer CSV.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory containing pimbench_*.log files.")
    parser.add_argument("--output", type=Path, required=True, help="Output simulator CSV path.")
    parser.add_argument(
        "--cycle-time-ns",
        type=float,
        default=1.0,
        help="Temporary cycle-to-time conversion in ns/cycle. Keep explicit in reports.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
