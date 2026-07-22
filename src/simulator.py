from __future__ import annotations

from pathlib import Path

import pandas as pd


SIMULATION_REQUIRED_COLUMNS = {
    "kernel_name",
    "simulator",
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
    _normalize_optional_numeric(simulation, "simulated_pim_cycles")
    _normalize_optional_numeric(simulation, "simulated_baseline_cycles")
    _normalize_optional_numeric(simulation, "simulated_speedup")
    _normalize_optional_numeric(simulation, "cycle_time_ns")
    if "cycle_time_ns" not in simulation.columns:
        simulation["cycle_time_ns"] = pd.NA

    if "simulated_pim_time_ms" in simulation.columns:
        simulation["simulated_pim_time_ms"] = pd.to_numeric(
            simulation["simulated_pim_time_ms"],
            errors="raise",
        )
    elif {"simulated_pim_cycles", "cycle_time_ns"} <= set(simulation.columns):
        simulation["simulated_pim_time_ms"] = simulation["simulated_pim_cycles"] * simulation["cycle_time_ns"] / 1e6
    else:
        raise ValueError(
            "PIM simulation CSV requires simulated_pim_time_ms, or simulated_pim_cycles plus cycle_time_ns"
        )

    if simulation["simulated_pim_time_ms"].isna().any() or (simulation["simulated_pim_time_ms"] <= 0).any():
        raise ValueError("simulated_pim_time_ms must be positive for all simulator rows")
    if "simulated_speedup" not in simulation.columns:
        simulation["simulated_speedup"] = pd.NA
    if simulation["simulated_speedup"].isna().any() and {"simulated_baseline_cycles", "simulated_pim_cycles"} <= set(simulation.columns):
        missing_speedup = simulation["simulated_speedup"].isna()
        simulation.loc[missing_speedup, "simulated_speedup"] = (
            simulation.loc[missing_speedup, "simulated_baseline_cycles"]
            / simulation.loc[missing_speedup, "simulated_pim_cycles"]
        )
    if "notes" not in simulation.columns:
        simulation["notes"] = ""
    simulation["normalized_kernel"] = simulation["kernel_name"].map(_normalize_name)
    return simulation


def attach_simulation_results(model_comparison: pd.DataFrame, simulation: pd.DataFrame | None) -> pd.DataFrame:
    """Attach raw and GPU-normalized simulator results by benchmark name.

    Simulator cycles belong to the simulator's own timing domain.  When the
    simulator also reports a baseline-to-PIM speedup, that ratio is applied to
    the measured GPU runtime so end-to-end totals do not directly mix unrelated
    absolute clocks.
    """

    result = model_comparison.copy()
    result["simulated_pim_time_ms"] = pd.NA
    result["simulated_scaled_pim_time_ms"] = pd.NA
    result["simulated_pim_cycles"] = pd.NA
    result["simulated_baseline_cycles"] = pd.NA
    result["simulated_speedup"] = pd.NA
    result["cycle_time_ns"] = pd.NA
    result["simulator"] = ""
    result["simulation_notes"] = ""
    result["simulation_time_basis"] = ""
    if simulation is None:
        return result

    sim_by_name = simulation.set_index("normalized_kernel").to_dict(orient="index")
    simulated_times = []
    simulated_scaled_times = []
    simulated_pim_cycles = []
    simulated_baseline_cycles = []
    simulated_speedups = []
    cycle_times = []
    simulator_names = []
    simulation_notes = []
    simulation_time_bases = []
    for _, row in result.iterrows():
        benchmark = row["benchmark"]
        matched = sim_by_name.get(_normalize_name(benchmark))
        if matched is None:
            simulated_times.append(pd.NA)
            simulated_scaled_times.append(pd.NA)
            simulated_pim_cycles.append(pd.NA)
            simulated_baseline_cycles.append(pd.NA)
            simulated_speedups.append(pd.NA)
            cycle_times.append(pd.NA)
            simulator_names.append("")
            simulation_notes.append("")
            simulation_time_bases.append("")
            continue
        raw_time_ms = float(matched["simulated_pim_time_ms"])
        speedup = matched.get("simulated_speedup", pd.NA)
        gpu_runtime_ms = pd.to_numeric(row.get("gpu_runtime_ms"), errors="coerce")
        if pd.notna(speedup) and float(speedup) > 0 and pd.notna(gpu_runtime_ms) and float(gpu_runtime_ms) > 0:
            scaled_time_ms = float(gpu_runtime_ms) / float(speedup)
            time_basis = "simulator speedup scaled to measured GPU runtime"
        else:
            scaled_time_ms = pd.NA
            time_basis = "raw simulator time only"
        simulated_times.append(raw_time_ms)
        simulated_scaled_times.append(scaled_time_ms)
        simulated_pim_cycles.append(matched.get("simulated_pim_cycles", pd.NA))
        simulated_baseline_cycles.append(matched.get("simulated_baseline_cycles", pd.NA))
        simulated_speedups.append(speedup)
        cycle_times.append(matched.get("cycle_time_ns", pd.NA))
        simulator_names.append(str(matched["simulator"]))
        simulation_notes.append(str(matched.get("notes", "")))
        simulation_time_bases.append(time_basis)

    result["simulated_pim_time_ms"] = simulated_times
    result["simulated_scaled_pim_time_ms"] = simulated_scaled_times
    result["simulated_pim_cycles"] = simulated_pim_cycles
    result["simulated_baseline_cycles"] = simulated_baseline_cycles
    result["simulated_speedup"] = simulated_speedups
    result["cycle_time_ns"] = cycle_times
    result["simulator"] = simulator_names
    result["simulation_notes"] = simulation_notes
    result["simulation_time_basis"] = simulation_time_bases
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


def _normalize_optional_numeric(frame: pd.DataFrame, column: str) -> None:
    if column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
