from __future__ import annotations

from pathlib import Path

import pandas as pd


SIMULATION_REQUIRED_COLUMNS = {
    "kernel_name",
    "simulator",
    "simulated_pim_time_ms",
}


def load_pim_simulation_csv(path: str | Path) -> pd.DataFrame:
    """Load optional PIM simulator output.

    The simulator CSV is intentionally small and simulator-agnostic. External
    adapters can convert SAIT PIMSimulator, Ramulator-PIM, UPMEM simulator, or
    another backend into this schema.
    """

    simulation_path = Path(path)
    simulation = pd.read_csv(simulation_path)
    if simulation.empty:
        raise ValueError(f"PIM simulation CSV is empty: {simulation_path}")

    missing = SIMULATION_REQUIRED_COLUMNS - set(simulation.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {simulation_path}: {missing_text}")

    simulation = simulation.copy()
    simulation["simulated_pim_time_ms"] = pd.to_numeric(
        simulation["simulated_pim_time_ms"],
        errors="raise",
    )
    if (simulation["simulated_pim_time_ms"] <= 0).any():
        raise ValueError("simulated_pim_time_ms must be positive for all simulator rows")
    if "notes" not in simulation.columns:
        simulation["notes"] = ""
    simulation["normalized_kernel"] = simulation["kernel_name"].map(_normalize_name)
    return simulation


def attach_simulation_results(model_comparison: pd.DataFrame, simulation: pd.DataFrame | None) -> pd.DataFrame:
    """Attach simulator timing rows to model comparison output by benchmark name."""

    result = model_comparison.copy()
    result["simulated_pim_time_ms"] = pd.NA
    result["simulator"] = ""
    result["simulation_notes"] = ""
    if simulation is None:
        return result

    sim_by_name = simulation.set_index("normalized_kernel").to_dict(orient="index")
    simulated_times = []
    simulator_names = []
    simulation_notes = []
    for benchmark in result["benchmark"]:
        matched = sim_by_name.get(_normalize_name(benchmark))
        if matched is None:
            simulated_times.append(pd.NA)
            simulator_names.append("")
            simulation_notes.append("")
            continue
        simulated_times.append(float(matched["simulated_pim_time_ms"]))
        simulator_names.append(str(matched["simulator"]))
        simulation_notes.append(str(matched.get("notes", "")))

    result["simulated_pim_time_ms"] = simulated_times
    result["simulator"] = simulator_names
    result["simulation_notes"] = simulation_notes
    return result


def summarize_simulation_coverage(model_comparison: pd.DataFrame) -> dict[str, int | float | str]:
    if model_comparison.empty or "simulated_pim_time_ms" not in model_comparison.columns:
        return {"benchmarks": 0, "simulated": 0, "coverage": 0.0, "simulators": ""}

    unique = model_comparison.drop_duplicates("benchmark")
    simulated = unique["simulated_pim_time_ms"].notna()
    simulators = sorted(set(unique.loc[simulated, "simulator"].astype(str)) - {""})
    return {
        "benchmarks": len(unique),
        "simulated": int(simulated.sum()),
        "coverage": 0.0 if len(unique) == 0 else float(simulated.mean()),
        "simulators": ", ".join(simulators),
    }


def _normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")
