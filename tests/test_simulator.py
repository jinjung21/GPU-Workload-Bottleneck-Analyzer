from pathlib import Path

import pandas as pd
import pytest

from src.simulator import attach_simulation_results, load_pim_simulation_csv, summarize_simulation_coverage


def test_load_pim_simulation_csv_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "sim.csv"
    path.write_text(
        "\n".join(
            [
                "kernel_name,simulator,simulated_pim_time_ms,notes",
                "vector_add,test_sim,0.05,unit test",
            ]
        ),
        encoding="utf-8",
    )

    simulation = load_pim_simulation_csv(path)

    assert simulation.loc[0, "kernel_name"] == "vector_add"
    assert simulation.loc[0, "simulated_pim_time_ms"] == 0.05
    assert simulation.loc[0, "normalized_kernel"] == "vector_add"


def test_load_pim_simulation_csv_rejects_nonpositive_time(tmp_path: Path) -> None:
    path = tmp_path / "sim.csv"
    path.write_text(
        "\n".join(
            [
                "kernel_name,simulator,simulated_pim_time_ms",
                "vector_add,test_sim,0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="simulated_pim_time_ms"):
        load_pim_simulation_csv(path)


def test_attach_simulation_results_and_summary() -> None:
    model_comparison = pd.DataFrame(
        {
            "model": ["feature_cost_v4", "feature_cost_v4"],
            "benchmark": ["vector_add", "cublas_sgemm"],
            "predicted_candidate": [True, False],
        }
    )
    simulation = pd.DataFrame(
        {
            "kernel_name": ["vector_add"],
            "simulator": ["test_sim"],
            "simulated_pim_time_ms": [0.05],
            "notes": ["unit test"],
            "normalized_kernel": ["vector_add"],
        }
    )

    attached = attach_simulation_results(model_comparison, simulation)
    summary = summarize_simulation_coverage(attached)

    assert attached.loc[0, "simulated_pim_time_ms"] == 0.05
    assert pd.isna(attached.loc[1, "simulated_pim_time_ms"])
    assert summary["simulated"] == 1
    assert summary["benchmarks"] == 2
    assert summary["simulators"] == "test_sim"
