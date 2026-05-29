#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ncu_metrics import NCU_METRIC_COLUMNS, parse_ncu_report_file


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*_ncu.*")):
        rows.append(parse_ncu_report_file(path))

    if not rows:
        raise SystemExit(f"No NCU reports found in {args.input_dir}")

    output = pd.DataFrame(rows)
    output = output[["kernel_name", *NCU_METRIC_COLUMNS]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote NCU metrics CSV: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Nsight Compute report files into analyzer metrics CSV.")
    parser.add_argument("--input-dir", type=Path, default=Path("profiles/ncu"), help="Directory with *_ncu.txt/csv files.")
    parser.add_argument("--output", type=Path, default=Path("profiles/ncu_metrics.csv"), help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
