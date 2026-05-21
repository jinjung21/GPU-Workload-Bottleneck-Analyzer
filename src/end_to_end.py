from __future__ import annotations

import pandas as pd


def build_end_to_end_evaluation(
    model_comparison: pd.DataFrame,
    cost_model: str = "feature_cost_v4",
    runtime_source: str = "estimated",
) -> pd.DataFrame:
    """Estimate total workload runtime for each offload decision policy.

    Each model decides which kernels to offload. Runtime is evaluated with a
    common cost model so the comparison measures decision quality, not different
    timing formulas.
    """

    if model_comparison.empty:
        return pd.DataFrame()
    if cost_model not in set(model_comparison["model"]):
        raise ValueError(f"Missing cost model rows for end-to-end evaluation: {cost_model}")
    if runtime_source not in {"estimated", "simulated"}:
        raise ValueError(f"Unsupported runtime source: {runtime_source}")

    cost_rows = model_comparison[model_comparison["model"] == cost_model].copy()
    cost_column = _cost_column(cost_rows, runtime_source)
    cost_rows[cost_column] = pd.to_numeric(cost_rows[cost_column], errors="raise")
    cost_by_benchmark = cost_rows.set_index("benchmark")[cost_column].to_dict()
    gpu_by_benchmark = cost_rows.set_index("benchmark")["gpu_runtime_ms"].to_dict()
    target_by_benchmark = cost_rows.set_index("benchmark")["target_candidate"].astype(bool).to_dict()
    gpu_only_ms = float(sum(gpu_by_benchmark.values()))

    rows = [
        {
            "model": "gpu_only",
            "total_runtime_ms": gpu_only_ms,
            "speedup_vs_gpu": 1.0,
            "runtime_reduction_pct": 0.0,
            "offloaded_kernels": 0,
            "false_offloads": 0,
            "missed_candidates": int(sum(target_by_benchmark.values())),
            "cost_model": cost_model,
            "runtime_source": runtime_source,
        }
    ]

    for model_name, group in model_comparison.groupby("model"):
        selected = group.set_index("benchmark")["predicted_candidate"].astype(bool).to_dict()
        total_runtime = 0.0
        false_offloads = 0
        missed_candidates = 0
        offloaded = 0

        for benchmark, gpu_runtime in gpu_by_benchmark.items():
            predicted = selected.get(benchmark, False)
            target = target_by_benchmark[benchmark]
            if predicted:
                total_runtime += cost_by_benchmark[benchmark]
                offloaded += 1
                false_offloads += int(not target)
            else:
                total_runtime += gpu_runtime
                missed_candidates += int(target)

        rows.append(
            {
                "model": model_name,
                "total_runtime_ms": total_runtime,
                "speedup_vs_gpu": gpu_only_ms / total_runtime if total_runtime > 0 else 0.0,
                "runtime_reduction_pct": 100.0 * (1.0 - total_runtime / gpu_only_ms) if gpu_only_ms > 0 else 0.0,
                "offloaded_kernels": offloaded,
                "false_offloads": false_offloads,
                "missed_candidates": missed_candidates,
                "cost_model": cost_model,
                "runtime_source": runtime_source,
            }
        )

    rows.append(_oracle_row(gpu_only_ms, gpu_by_benchmark, cost_by_benchmark, target_by_benchmark, cost_model))
    rows[-1]["runtime_source"] = runtime_source
    return pd.DataFrame(rows).sort_values(
        ["speedup_vs_gpu", "false_offloads", "missed_candidates"],
        ascending=[False, True, True],
    )


def _oracle_row(
    gpu_only_ms: float,
    gpu_by_benchmark: dict[str, float],
    cost_by_benchmark: dict[str, float],
    target_by_benchmark: dict[str, bool],
    cost_model: str,
) -> dict[str, float | int | str]:
    total_runtime = 0.0
    offloaded = 0
    for benchmark, gpu_runtime in gpu_by_benchmark.items():
        if target_by_benchmark[benchmark]:
            total_runtime += cost_by_benchmark[benchmark]
            offloaded += 1
        else:
            total_runtime += gpu_runtime

    return {
        "model": "oracle_labels",
        "total_runtime_ms": total_runtime,
        "speedup_vs_gpu": gpu_only_ms / total_runtime if total_runtime > 0 else 0.0,
        "runtime_reduction_pct": 100.0 * (1.0 - total_runtime / gpu_only_ms) if gpu_only_ms > 0 else 0.0,
        "offloaded_kernels": offloaded,
        "false_offloads": 0,
        "missed_candidates": 0,
        "cost_model": cost_model,
    }


def _cost_column(cost_rows: pd.DataFrame, runtime_source: str) -> str:
    if runtime_source == "estimated":
        return "estimated_pim_time_ms"
    if "simulated_pim_time_ms" not in cost_rows.columns:
        raise ValueError("simulated runtime source requested, but simulated_pim_time_ms is missing")
    if cost_rows["simulated_pim_time_ms"].isna().any():
        missing = ", ".join(cost_rows.loc[cost_rows["simulated_pim_time_ms"].isna(), "benchmark"].astype(str))
        raise ValueError(f"Missing simulated PIM time for benchmarks: {missing}")
    return "simulated_pim_time_ms"
