from __future__ import annotations

from pathlib import Path

import pandas as pd


BASELINE_REQUIRED_COLUMNS = {
    "benchmark",
    "aliases",
    "domain",
    "paper_memory_bound",
    "expected_pim_priority",
    "expected_min_score",
    "paper_notes",
}


def load_paper_baseline_csv(path: str | Path) -> pd.DataFrame:
    """Load paper baseline workload metadata used for model alignment checks."""

    baseline_path = Path(path)
    baseline = pd.read_csv(baseline_path)
    if baseline.empty:
        raise ValueError(f"Paper baseline CSV is empty: {baseline_path}")

    missing = BASELINE_REQUIRED_COLUMNS - set(baseline.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {baseline_path}: {missing_text}")

    baseline["expected_min_score"] = pd.to_numeric(baseline["expected_min_score"], errors="raise")
    baseline["paper_memory_bound"] = baseline["paper_memory_bound"].map(_parse_bool)
    return baseline


def compare_to_paper_baseline(profile: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Compare analyzed kernels with a paper workload baseline.

    A match means the analyzed kernel reaches the expected PIM/NMP score floor
    for the corresponding paper workload. Missing rows are reported explicitly
    so the report shows which benchmark coverage is still incomplete.
    """

    profile_by_name = {_normalize_name(row["kernel_name"]): row for _, row in profile.iterrows()}
    rows = []
    for _, expected in baseline.iterrows():
        matched = _find_profile_row(profile_by_name, expected["benchmark"], expected["aliases"])
        if matched is None:
            rows.append(_missing_row(expected))
            continue

        score = int(matched["pim_nmp_score"])
        expected_min = int(expected["expected_min_score"])
        model_alignment = "match" if score >= expected_min else "miss"
        rows.append(
            {
                "paper": "PrIM 2022",
                "benchmark": expected["benchmark"],
                "domain": expected["domain"],
                "paper_expected": expected["expected_pim_priority"],
                "expected_min_score": expected_min,
                "our_kernel": matched["kernel_name"],
                "our_bottleneck": matched["bottleneck_classification"],
                "our_pim_label": matched["pim_nmp_suitability"],
                "our_score": score,
                "model_alignment": model_alignment,
                "paper_notes": expected["paper_notes"],
            }
        )

    return pd.DataFrame(rows)


def summarize_baseline_alignment(comparison: pd.DataFrame) -> dict[str, int | float]:
    profiled = comparison[comparison["model_alignment"] != "not profiled"]
    matches = profiled[profiled["model_alignment"] == "match"]
    total_profiled = len(profiled)
    return {
        "benchmarks": len(comparison),
        "profiled": total_profiled,
        "matches": len(matches),
        "match_rate": 0.0 if total_profiled == 0 else len(matches) / total_profiled,
    }


def _find_profile_row(profile_by_name: dict[str, pd.Series], benchmark: str, aliases: str) -> pd.Series | None:
    candidates = [benchmark, *str(aliases).split("|")]
    for candidate in candidates:
        normalized = _normalize_name(candidate)
        if normalized in profile_by_name:
            return profile_by_name[normalized]
    return None


def _missing_row(expected: pd.Series) -> dict[str, object]:
    return {
        "paper": "PrIM 2022",
        "benchmark": expected["benchmark"],
        "domain": expected["domain"],
        "paper_expected": expected["expected_pim_priority"],
        "expected_min_score": int(expected["expected_min_score"]),
        "our_kernel": "",
        "our_bottleneck": "",
        "our_pim_label": "",
        "our_score": "",
        "model_alignment": "not profiled",
        "paper_notes": expected["paper_notes"],
    }


def _normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value in paper baseline: {value}")
