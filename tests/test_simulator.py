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


def test_load_pim_simulation_csv_can_compute_time_from_cycles(tmp_path: Path) -> None:
    path = tmp_path / "sim.csv"
    path.write_text(
        "\n".join(
            [
                "kernel_name,simulator,simulated_pim_cycles,simulated_baseline_cycles,cycle_time_ns",
                "gemv,test_sim,13166,36082,1.0",
            ]
        ),
        encoding="utf-8",
    )

    simulation = load_pim_simulation_csv(path)

    assert simulation.loc[0, "simulated_pim_time_ms"] == 0.013166
    assert round(simulation.loc[0, "simulated_speedup"], 5) == 2.74054


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
            "gpu_runtime_ms": [1.0, 4.0],
        }
    )
    simulation = pd.DataFrame(
        {
            "kernel_name": ["vector_add"],
            "simulator": ["test_sim"],
            "simulated_pim_time_ms": [0.05],
            "simulated_pim_cycles": [3349],
            "simulated_baseline_cycles": [6651],
            "simulated_speedup": [1.98597],
            "cycle_time_ns": [1.0],
            "notes": ["unit test"],
            "normalized_kernel": ["vector_add"],
        }
    )

    attached = attach_simulation_results(model_comparison, simulation)
    summary = summarize_simulation_coverage(attached)

    assert attached.loc[0, "simulated_pim_time_ms"] == 0.05
    assert round(attached.loc[0, "simulated_scaled_pim_time_ms"], 6) == round(1.0 / 1.98597, 6)
    assert attached.loc[0, "simulated_pim_cycles"] == 3349
    assert attached.loc[0, "simulated_speedup"] == 1.98597
    assert attached.loc[0, "simulation_time_basis"] == "simulator speedup scaled to measured GPU runtime"
    assert pd.isna(attached.loc[1, "simulated_pim_time_ms"])
    assert summary["simulated"] == 1
    assert summary["benchmarks"] == 2
    assert summary["simulators"] == "test_sim"
