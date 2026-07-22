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
    if runtime_source not in {"estimated", "simulated", "simulator_scaled"}:
        raise ValueError(f"Unsupported runtime source: {runtime_source}")

    cost_rows = model_comparison[model_comparison["model"] == cost_model].copy()
    cost_rows["selected_pim_time_ms"] = _select_pim_runtime(cost_rows, runtime_source)
    cost_rows["selected_pim_time_basis"] = _select_pim_runtime_basis(cost_rows, runtime_source)
    cost_by_benchmark = cost_rows.set_index("benchmark")["selected_pim_time_ms"].to_dict()
    basis_by_benchmark = cost_rows.set_index("benchmark")["selected_pim_time_basis"].to_dict()
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
            "simulator_backed_offloads": 0,
            "analytical_fallback_offloads": 0,
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
        simulator_backed = 0
        analytical_fallback = 0

        for benchmark, gpu_runtime in gpu_by_benchmark.items():
            predicted = selected.get(benchmark, False)
            target = target_by_benchmark[benchmark]
            if predicted:
                total_runtime += cost_by_benchmark[benchmark]
                offloaded += 1
                simulator_backed += int(str(basis_by_benchmark[benchmark]).startswith("simulator"))
                analytical_fallback += int(basis_by_benchmark[benchmark] == "analytical estimate")
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
                "simulator_backed_offloads": simulator_backed,
                "analytical_fallback_offloads": analytical_fallback,
                "cost_model": cost_model,
                "runtime_source": runtime_source,
            }
        )

    rows.append(
        _oracle_row(
            gpu_only_ms,
            gpu_by_benchmark,
            cost_by_benchmark,
            target_by_benchmark,
            basis_by_benchmark,
            cost_model,
        )
    )
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
    basis_by_benchmark: dict[str, str],
    cost_model: str,
) -> dict[str, float | int | str]:
    total_runtime = 0.0
    offloaded = 0
    simulator_backed = 0
    analytical_fallback = 0
    for benchmark, gpu_runtime in gpu_by_benchmark.items():
        if target_by_benchmark[benchmark]:
            total_runtime += cost_by_benchmark[benchmark]
            offloaded += 1
            simulator_backed += int(str(basis_by_benchmark[benchmark]).startswith("simulator"))
            analytical_fallback += int(basis_by_benchmark[benchmark] == "analytical estimate")
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
        "simulator_backed_offloads": simulator_backed,
        "analytical_fallback_offloads": analytical_fallback,
        "cost_model": cost_model,
    }


def _select_pim_runtime(cost_rows: pd.DataFrame, runtime_source: str) -> pd.Series:
    if runtime_source == "estimated":
        return pd.to_numeric(cost_rows["estimated_pim_time_ms"], errors="raise")
    if runtime_source == "simulator_scaled":
        estimated = pd.to_numeric(cost_rows["estimated_pim_time_ms"], errors="raise")
        scaled = _numeric_column(cost_rows, "simulated_scaled_pim_time_ms")
        direct = _numeric_column(cost_rows, "simulated_pim_time_ms")
        return scaled.fillna(direct).fillna(estimated)
    if "simulated_pim_time_ms" not in cost_rows.columns:
        raise ValueError("simulated runtime source requested, but simulated_pim_time_ms is missing")
    estimated = pd.to_numeric(cost_rows["estimated_pim_time_ms"], errors="raise")
    simulated = pd.to_numeric(cost_rows["simulated_pim_time_ms"], errors="coerce")
    return simulated.fillna(estimated)


def _select_pim_runtime_basis(cost_rows: pd.DataFrame, runtime_source: str) -> pd.Series:
    if runtime_source == "estimated":
        return pd.Series("analytical estimate", index=cost_rows.index)
    if runtime_source == "simulated":
        direct = _numeric_column(cost_rows, "simulated_pim_time_ms")
        return direct.map(lambda value: "simulator direct time" if pd.notna(value) else "analytical estimate")

    scaled = _numeric_column(cost_rows, "simulated_scaled_pim_time_ms")
    direct = _numeric_column(cost_rows, "simulated_pim_time_ms")
    basis = pd.Series("analytical estimate", index=cost_rows.index)
    basis.loc[direct.notna()] = "simulator direct time"
    basis.loc[scaled.notna()] = "simulator speedup scaled"
    return basis


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
