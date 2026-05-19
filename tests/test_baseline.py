from pathlib import Path

import pandas as pd

from src.baseline import compare_to_paper_baseline, load_paper_baseline_csv, summarize_baseline_alignment


def test_compare_to_paper_baseline_matches_alias() -> None:
    profile = pd.DataFrame(
        {
            "kernel_name": ["vector_add"],
            "bottleneck_classification": ["memory-bound / bandwidth-bound"],
            "pim_nmp_suitability": ["PIM-friendly"],
            "pim_nmp_score": [62],
        }
    )
    baseline = pd.DataFrame(
        {
            "source": ["unit test"],
            "benchmark": ["VA"],
            "aliases": ["vector_add|vector-add"],
            "domain": ["Dense linear algebra"],
            "paper_memory_bound": [True],
            "expected_pim_candidate": [True],
            "expected_pim_priority": ["medium"],
            "expected_min_score": [45],
            "operation_complexity": ["low"],
            "communication_intensity": ["low"],
            "partitionability": ["high"],
            "host_transfer_sensitivity": ["low"],
            "paper_notes": ["Vector addition baseline"],
        }
    )

    comparison = compare_to_paper_baseline(profile, baseline)

    assert comparison.loc[0, "benchmark"] == "VA"
    assert comparison.loc[0, "paper"] == "unit test"
    assert comparison.loc[0, "our_kernel"] == "vector_add"
    assert comparison.loc[0, "model_alignment"] == "match"


def test_compare_to_paper_baseline_reports_missing_workload() -> None:
    profile = pd.DataFrame(
        {
            "kernel_name": ["unrelated"],
            "bottleneck_classification": ["compute-bound"],
            "pim_nmp_suitability": ["low PIM priority"],
            "pim_nmp_score": [10],
        }
    )
    baseline = pd.DataFrame(
        {
            "benchmark": ["BFS"],
            "aliases": ["breadth_first_search"],
            "domain": ["Graph processing"],
            "paper_memory_bound": [True],
            "expected_pim_candidate": [True],
            "expected_pim_priority": ["high"],
            "expected_min_score": [70],
            "operation_complexity": ["low"],
            "communication_intensity": ["high"],
            "partitionability": ["medium"],
            "host_transfer_sensitivity": ["medium"],
            "paper_notes": ["Graph traversal baseline"],
        }
    )

    comparison = compare_to_paper_baseline(profile, baseline)

    assert comparison.loc[0, "model_alignment"] == "not profiled"


def test_summarize_baseline_alignment_ignores_missing_rows() -> None:
    comparison = pd.DataFrame(
        {
            "model_alignment": ["match", "miss", "not profiled"],
        }
    )

    summary = summarize_baseline_alignment(comparison)

    assert summary["benchmarks"] == 3
    assert summary["profiled"] == 2
    assert summary["matches"] == 1
    assert summary["match_rate"] == 0.5


def test_load_paper_baseline_csv(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    baseline_path.write_text(
        "\n".join(
            [
                "benchmark,aliases,domain,paper_memory_bound,expected_pim_candidate,expected_pim_priority,expected_min_score,operation_complexity,communication_intensity,partitionability,host_transfer_sensitivity,paper_notes",
                "VA,vector_add,Dense linear algebra,True,True,medium,45,low,low,high,low,Vector addition baseline",
            ]
        ),
        encoding="utf-8",
    )

    baseline = load_paper_baseline_csv(baseline_path)

    assert baseline.loc[0, "benchmark"] == "VA"
    assert baseline.loc[0, "source"] == "PrIM 2022"
    assert baseline.loc[0, "expected_min_score"] == 45
