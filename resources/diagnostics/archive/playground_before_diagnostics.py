"""Deterministic smoke/regression simulations for the CONSCIENCE engine.

The four scenarios progressively enable:

1. a wave vector,
2. wave vector + collisionless baryons,
3. wave vector + collisionless baryons + gas,
4. wave vector + collisionless baryons + gas + a central sink.

Baseline arrays are written below resources/data/. That directory is ignored by
Git, while this harness remains versioned with the source code.

Create baselines with ``python chats_playground.py baseline``, compare after a
code change with ``python chats_playground.py compare``, and verify exact
midpoint continuation with ``python chats_playground.py restart-check``.
Use ``python chats_playground.py segment-check`` to verify that ending and
restarting an unchanged simulation composes like one uninterrupted run.
"""

from __future__ import annotations

# Historical snapshot/migration: retained for review, not for execution.
if __name__ == "__main__":
    raise SystemExit("Archived file; use the documented active diagnostics instead.")


import argparse
import hashlib
import json
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "resources" / "data" / "chat_regression_baselines"
SCENARIOS = (
    "wave",
    "wave_baryons",
    "wave_baryons_gas",
    "wave_baryons_gas_sink",
)
SEED = 20261002
RTOL = 1.0e-11
ATOL = 1.0e-12


def _configure_local_runtime_directories() -> None:
    """Keep compiler scratch/cache files inside the writable ignored data tree."""
    runtime_root = PROJECT_ROOT / "resources" / "data" / "chat_runtime"
    temp_dir = runtime_root / "tmp"
    cache_dir = runtime_root / "cupy_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["CUPY_CACHE_DIR"] = str(cache_dir)


def _configure_cuda_on_windows() -> str | None:
    """Make an installed CUDA toolkit discoverable before importing CuPy."""
    if os.name != "nt":
        return os.environ.get("CUDA_PATH")

    configured = os.environ.get("CUDA_PATH")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))

    toolkit_root = Path(
        os.environ.get("ProgramFiles", r"C:\Program Files")
    ) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if toolkit_root.is_dir():
        candidates.extend(
            sorted(
                (path for path in toolkit_root.glob("v*") if path.is_dir()),
                reverse=True,
            )
        )

    for candidate in candidates:
        bin_dir = candidate / "bin"
        if not bin_dir.is_dir() or not any(bin_dir.glob("nvrtc64_*.dll")):
            continue

        os.environ["CUDA_PATH"] = str(candidate)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(bin_dir) not in path_entries:
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return str(candidate)
    return None


_configure_local_runtime_directories()
CUDA_PATH = _configure_cuda_on_windows()

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons  # noqa: E402
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas  # noqa: E402
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody  # noqa: E402
from resources.Classes.Simulation_Class import Simulation_Class  # noqa: E402
from resources.Classes.Wave_vector_class import Wave_vector_class  # noqa: E402


def _reset_random_state(seed: int) -> None:
    """Reset every random generator currently used by the project."""
    np.random.seed(seed)
    cp.random.seed(seed)


def _base_simulation(total_time: float = 0.003) -> Simulation_Class:
    return Simulation_Class(
        dim=3,
        boundaries=[(-10.0, 10.0)] * 3,
        N=16,
        total_time=total_time,
        h=0.001,
        order_of_evolution=2,
        use_gravity=True,
        static_potential=None,
        save_max_vals=False,
        m_s=2.5e-22,
        use_sponge=False,
        self_int=False,
    )


def _add_wave(simulation: Simulation_Class) -> Wave_vector_class:
    wave_vector = Wave_vector_class(
        packet_type="resources/solitons/GroundState(1).dat",
        means=[0.0, 0.0, 0.0],
        st_deviations=[0.5, 0.5, 0.5],
        simulation=simulation,
        mass=1,
        omega=1,
        momenta=[0.0, 0.0, 0.0],
        spin=0,
        desired_soliton_mass=5.3090064e7,
        random_seed=SEED,
    )
    simulation.add_wave_vector(wave_vector)
    return wave_vector


def _add_baryons(simulation: Simulation_Class) -> Baryons:
    particle_count = 64
    total_mass = 1.0e6
    rng = np.random.default_rng(SEED)
    theta = np.linspace(0.0, 2.0 * np.pi, particle_count, endpoint=False)
    positions = np.column_stack(
        (2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros(particle_count))
    )
    velocities = np.column_stack(
        (-0.05 * np.sin(theta), 0.05 * np.cos(theta), np.zeros(particle_count))
    )
    positions += rng.normal(0.0, 1.0e-3, size=positions.shape)
    velocities += rng.normal(0.0, 1.0e-4, size=velocities.shape)

    # Bypass only the GPU-random profile initializer, which takes minutes to
    # JIT-compile on the local GTX 1050 Ti. All evolution methods are inherited
    # normally from Baryons/NBody and are exercised by the scenarios below.
    baryons = Baryons.__new__(Baryons)
    baryons.simulation = simulation
    baryons.N = particle_count
    baryons.m_particle = total_mass / particle_count
    baryons.positions = cp.asarray(positions, dtype=cp.float64)
    baryons.velocities = cp.asarray(velocities, dtype=cp.float64)
    baryons.name = "regression_baryons"
    baryons._density_cache = None
    baryons._density_cache_valid = False
    simulation.add_baryons(baryons)
    return baryons


def _add_gas(simulation: Simulation_Class) -> NBodyGas:
    x, y, z = (np.asarray(grid) for grid in simulation.grids)
    radius_squared = x * x + y * y + z * z
    rho = np.exp(-0.5 * radius_squared / 2.5**2) + 1.0e-12
    rho *= 2.0e5 / (rho.sum() * simulation.dV)

    zeros = np.zeros_like(rho)
    gas = NBodyGas(
        simulation=simulation,
        rho=rho,
        vx=zeros,
        vy=zeros,
        vz=zeros,
        cs=0.05,
        gamma=5.0 / 3.0,
        tcool=None,
        e_floor=1.0e-10,
        rho_floor=1.0e-12,
        cfl=0.4,
        max_substeps=20,
        name="regression_gas",
    )
    simulation.add_baryons(gas)
    return gas


def _add_sink(simulation: Simulation_Class) -> SinkNBody:
    sink = SinkNBody(
        simulation=simulation,
        N_sinks=1,
        initial_masses=[1.0e5],
        initial_positions=[[0.0, 0.0, 0.0]],
        initial_velocities=[[0.0, 0.0, 0.0]],
        capture_radius=2.5 * min(simulation.dx),
        softening_length=1.5 * min(simulation.dx),
        reservoir_tau=0.01,
    )
    simulation.add_baryons(sink)
    return sink


def build_scenario(
    name: str,
    total_time: float = 0.003,
) -> tuple[Simulation_Class, dict[str, Any]]:
    """Build one scenario and retain named references to stateful components."""
    _reset_random_state(SEED)
    simulation = _base_simulation(total_time=total_time)
    components: dict[str, Any] = {}

    wave_vector = _add_wave(simulation)
    components["wave"] = wave_vector.wave_vector

    if name != "wave":
        components["baryons"] = _add_baryons(simulation)
    if name in {"wave_baryons_gas", "wave_baryons_gas_sink"}:
        components["gas"] = _add_gas(simulation)
    if name == "wave_baryons_gas_sink":
        components["sink"] = _add_sink(simulation)

    return simulation, components


def _cpu(value: Any) -> np.ndarray:
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def _capture_component_state(
    state: dict[str, np.ndarray],
    stage: str,
    components: dict[str, Any],
) -> None:
    for index, wave in enumerate(components["wave"]):
        state[f"{stage}_wave_{index}_psi"] = _cpu(wave.psi)
        state[f"{stage}_wave_{index}_multiplicity"] = np.asarray(wave.multiplicity)

    baryons = components.get("baryons")
    if baryons is not None:
        state[f"{stage}_baryons_positions"] = _cpu(baryons.positions)
        state[f"{stage}_baryons_velocities"] = _cpu(baryons.velocities)
        state[f"{stage}_baryons_particle_mass"] = np.asarray(baryons.m_particle)

    gas = components.get("gas")
    if gas is not None:
        for attribute in ("rho", "vx", "vy", "vz", "E"):
            state[f"{stage}_gas_{attribute}"] = _cpu(getattr(gas, attribute))
        for attribute in (
            "cs",
            "cfl",
            "rho_floor",
            "max_substeps",
            "gamma",
            "e_floor",
            "E_radiated",
        ):
            state[f"{stage}_gas_{attribute}"] = np.asarray(getattr(gas, attribute))

    sink = components.get("sink")
    if sink is not None:
        for attribute in (
            "positions",
            "velocities",
            "mass_bh",
            "mass_res",
            "masses",
        ):
            state[f"{stage}_sink_{attribute}"] = _cpu(getattr(sink, attribute))
        for attribute in (
            "capture_radius",
            "softening_length",
            "softening_bh",
            "softening_cusp",
            "reservoir_tau",
            "E_diss_kin_total",
            "E_diss_formation_total",
            "total_accreted_mass",
        ):
            state[f"{stage}_sink_{attribute}"] = np.asarray(getattr(sink, attribute))


def _capture_evolution_state(
    state: dict[str, np.ndarray],
    simulation: Simulation_Class,
) -> None:
    evolution = simulation.evolution
    if evolution is None:
        raise RuntimeError("Evolution state is unavailable")

    total_density = evolution._compute_total_density(simulation.wave_functions)
    state["final_total_density_without_sinks"] = _cpu(total_density)

    density_with_sinks = total_density.copy()
    for component in simulation.baryonic_matter:
        if evolution._is_sink_system(component):
            density_with_sinks += component.deposit_to_grid()
    state["final_total_density_with_sinks"] = _cpu(density_with_sinks)

    state["accessible_times"] = np.asarray(evolution.scribe.accessible_times)
    if evolution.scribe.energy_log:
        columns = (
            "time",
            "K_total",
            "W",
            "E_total",
            "K_flow",
            "U_quantum",
            "K_baryons",
            "W_self",
            "W_static",
            "W/|E|",
            "E_diss",
            "E_tot_cons",
        )
        state["energy_columns"] = np.asarray(columns)
        state["energy_values"] = np.asarray(
            [
                [record.get(column, np.nan) for column in columns]
                for record in evolution.scribe.energy_log
            ],
            dtype=np.float64,
        )


def run_scenario(name: str) -> dict[str, np.ndarray]:
    print(f"\n=== Running deterministic scenario: {name} ===")
    simulation, components = build_scenario(name)
    state: dict[str, np.ndarray] = {}
    _capture_component_state(state, "initial", components)

    simulation.evolve(
        save_every=simulation.num_steps,
        diagnostics_every=simulation.num_steps,
    )

    _capture_component_state(state, "final", components)
    _capture_evolution_state(state, simulation)
    state["seed"] = np.asarray(SEED)
    state["num_steps"] = np.asarray(simulation.num_steps)
    state["time_step"] = np.asarray(simulation.h)
    return state


def _components_from_simulation(simulation: Simulation_Class) -> dict[str, Any]:
    components: dict[str, Any] = {"wave": simulation.wave_functions}
    for component in simulation.baryonic_matter:
        if isinstance(component, SinkNBody):
            components["sink"] = component
        elif isinstance(component, NBodyGas):
            components["gas"] = component
        elif isinstance(component, Baryons):
            components["baryons"] = component
    return components


def _capture_finished_simulation(
    simulation: Simulation_Class,
) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    _capture_component_state(
        state,
        "final",
        _components_from_simulation(simulation),
    )
    _capture_evolution_state(state, simulation)
    state["num_steps"] = np.asarray(simulation.num_steps)
    state["time_step"] = np.asarray(simulation.h)
    return state


def _capture_physical_state(
    simulation: Simulation_Class,
) -> dict[str, np.ndarray]:
    """Capture evolved fields without segment-local output diagnostics."""
    state: dict[str, np.ndarray] = {}
    _capture_component_state(
        state,
        "final",
        _components_from_simulation(simulation),
    )
    evolution = simulation.evolution
    if evolution is None:
        raise RuntimeError("Evolution state is unavailable")
    total_density = evolution._compute_total_density(simulation.wave_functions)
    state["final_total_density_without_sinks"] = _cpu(total_density)
    for component in simulation.baryonic_matter:
        if evolution._is_sink_system(component):
            total_density = total_density + component.deposit_to_grid()
    state["final_total_density_with_sinks"] = _cpu(total_density)
    return state


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _state_digest(state: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(_array_digest(state[key]).encode("ascii"))
    return digest.hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    source_files = sorted((PROJECT_ROOT / "resources").rglob("*.py"))
    source_files.append(Path(__file__))
    for path in source_files:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _environment_metadata() -> dict[str, Any]:
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_path": CUDA_PATH,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "gpu": device_name,
        "git_commit": _git_commit(),
        "source_digest": _source_digest(),
    }


def create_baselines(
    baseline_dir: Path,
    scenario_names: tuple[str, ...],
    overwrite: bool,
) -> bool:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    existing = [baseline_dir / f"{name}.npz" for name in scenario_names]
    if not overwrite and any(path.exists() for path in existing):
        found = ", ".join(str(path) for path in existing if path.exists())
        raise FileExistsError(
            f"Baseline files already exist: {found}. "
            "Use --overwrite only when intentionally accepting new reference behavior."
        )

    scenario_metadata: dict[str, Any] = {}
    for name in scenario_names:
        state = run_scenario(name)
        output_path = baseline_dir / f"{name}.npz"
        np.savez_compressed(output_path, **state)
        scenario_metadata[name] = {
            "file": output_path.name,
            "state_digest": _state_digest(state),
            "array_digests": {
                key: _array_digest(value)
                for key, value in sorted(state.items())
            },
        }
        del state

    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "rtol": RTOL,
        "atol": ATOL,
        "environment": _environment_metadata(),
        "scenarios": scenario_metadata,
    }
    (baseline_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"\nBaselines saved to ignored directory: {baseline_dir}")
    return True


def _compare_array(
    key: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {
        "key": key,
        "reference_shape": reference.shape,
        "candidate_shape": candidate.shape,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        details["reason"] = "shape or dtype changed"
        return False, details

    if reference.dtype.kind in "biufc" and candidate.dtype.kind in "biufc":
        exact = bool(np.array_equal(reference, candidate, equal_nan=True))
        close = bool(
            np.allclose(
                reference,
                candidate,
                rtol=RTOL,
                atol=ATOL,
                equal_nan=True,
            )
        )
        difference = np.abs(candidate - reference)
        finite_difference = difference[np.isfinite(difference)]
        max_absolute = (
            float(np.max(finite_difference))
            if finite_difference.size
            else 0.0
        )
        scale = float(np.nanmax(np.abs(reference))) if reference.size else 0.0
        details.update(
            {
                "exact": exact,
                "close": close,
                "max_absolute_difference": max_absolute,
                "relative_to_reference_max": max_absolute / max(scale, 1.0),
            }
        )
        return close, details

    exact = bool(np.array_equal(reference, candidate))
    details["exact"] = exact
    return exact, details


def compare_with_baselines(
    baseline_dir: Path,
    scenario_names: tuple[str, ...],
) -> bool:
    manifest_path = baseline_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Run baseline mode first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_passed = True
    report: dict[str, Any] = {
        "baseline_environment": manifest.get("environment"),
        "current_environment": _environment_metadata(),
        "scenarios": {},
    }

    for name in scenario_names:
        baseline_path = baseline_dir / f"{name}.npz"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline: {baseline_path}")

        candidate = run_scenario(name)
        scenario_passed = True
        comparisons: list[dict[str, Any]] = []
        with np.load(baseline_path, allow_pickle=False) as stored:
            reference_keys = set(stored.files)
            candidate_keys = set(candidate)
            if reference_keys != candidate_keys:
                scenario_passed = False
                comparisons.append(
                    {
                        "reason": "state keys changed",
                        "missing": sorted(reference_keys - candidate_keys),
                        "added": sorted(candidate_keys - reference_keys),
                    }
                )

            for key in sorted(reference_keys & candidate_keys):
                passed, details = _compare_array(key, stored[key], candidate[key])
                scenario_passed &= passed
                if not passed or not details.get("exact", False):
                    comparisons.append(details)

        candidate_digest = _state_digest(candidate)
        expected_digest = (
            manifest.get("scenarios", {})
            .get(name, {})
            .get("state_digest")
        )
        report["scenarios"][name] = {
            "passed": scenario_passed,
            "bitwise_state_match": candidate_digest == expected_digest,
            "differences": comparisons,
        }
        all_passed &= scenario_passed
        status = "PASS" if scenario_passed else "FAIL"
        exact_note = (
            "bitwise identical"
            if candidate_digest == expected_digest
            else "within tolerance or changed"
        )
        print(f"{status}: {name} ({exact_note})")

        del candidate

    report_path = baseline_dir / "latest_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nComparison report: {report_path}")
    return all_passed


def check_midpoint_restarts(
    baseline_dir: Path,
    scenario_names: tuple[str, ...],
) -> bool:
    """Compare a six-step run with a reload from its step-three checkpoint."""
    report: dict[str, Any] = {
        "total_time": 0.006,
        "midpoint_step": 3,
        "environment": _environment_metadata(),
        "scenarios": {},
    }
    all_passed = True
    resume_root = baseline_dir / "restart_runs"
    resume_root.mkdir(parents=True, exist_ok=True)

    for name in scenario_names:
        print(f"\n=== Restart equivalence scenario: {name} ===")
        continuous, _ = build_scenario(name, total_time=0.006)
        continuous.evolve(save_every=3, diagnostics_every=3)
        continuous_state = _capture_finished_simulation(continuous)

        midpoint_manifest = (
            Path(continuous.snapshot_directory)
            / "checkpoints"
            / "step_00000003"
            / "manifest.json"
        )
        resumed_output = resume_root / f"{name}_{uuid.uuid4().hex[:8]}"
        resumed = Simulation_Class.from_checkpoint(
            midpoint_manifest,
            output_directory=resumed_output,
        )
        resumed.resume()
        resumed_state = _capture_finished_simulation(resumed)

        reference_keys = set(continuous_state)
        resumed_keys = set(resumed_state)
        scenario_passed = reference_keys == resumed_keys
        differences: list[dict[str, Any]] = []
        if reference_keys != resumed_keys:
            differences.append(
                {
                    "reason": "state keys changed",
                    "missing": sorted(reference_keys - resumed_keys),
                    "added": sorted(resumed_keys - reference_keys),
                }
            )

        for key in sorted(reference_keys & resumed_keys):
            passed, details = _compare_array(
                key,
                continuous_state[key],
                resumed_state[key],
            )
            scenario_passed &= passed
            if not passed or not details.get("exact", False):
                differences.append(details)

        exact = _state_digest(continuous_state) == _state_digest(resumed_state)
        report["scenarios"][name] = {
            "passed": scenario_passed,
            "bitwise_state_match": exact,
            "midpoint_manifest": str(midpoint_manifest.resolve()),
            "resumed_output": str(resumed_output.resolve()),
            "differences": differences,
        }
        all_passed &= scenario_passed
        status = "PASS" if scenario_passed else "FAIL"
        exact_note = "bitwise identical" if exact else "within tolerance or changed"
        print(f"{status}: {name} ({exact_note})")

    report_path = baseline_dir / "latest_restart_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRestart comparison report: {report_path}")
    return all_passed


def check_endpoint_segments(
    baseline_dir: Path,
    scenario_names: tuple[str, ...],
) -> bool:
    """Compare six uninterrupted steps with two completed three-step segments."""
    report: dict[str, Any] = {
        "continuous_time": 0.006,
        "segment_time": 0.003,
        "environment": _environment_metadata(),
        "scenarios": {},
    }
    all_passed = True

    for name in scenario_names:
        print(f"\n=== Endpoint segment composition scenario: {name} ===")
        continuous, _ = build_scenario(name, total_time=0.006)
        continuous.evolve(save_every=3, diagnostics_every=3)
        continuous_state = _capture_physical_state(continuous)

        first_segment, _ = build_scenario(name, total_time=0.003)
        first_segment.evolve(save_every=3, diagnostics_every=3)
        second_segment = Simulation_Class.from_checkpoint(
            first_segment.snapshot_directory
        )
        second_segment.start_new_segment(total_time=0.003)
        second_segment.evolve(save_every=3, diagnostics_every=3)
        segmented_state = _capture_physical_state(second_segment)

        reference_keys = set(continuous_state)
        segmented_keys = set(segmented_state)
        scenario_passed = reference_keys == segmented_keys
        differences: list[dict[str, Any]] = []
        if reference_keys != segmented_keys:
            differences.append(
                {
                    "reason": "state keys changed",
                    "missing": sorted(reference_keys - segmented_keys),
                    "added": sorted(segmented_keys - reference_keys),
                }
            )

        for key in sorted(reference_keys & segmented_keys):
            passed, details = _compare_array(
                key,
                continuous_state[key],
                segmented_state[key],
            )
            scenario_passed &= passed
            if not passed or not details.get("exact", False):
                differences.append(details)

        exact = _state_digest(continuous_state) == _state_digest(segmented_state)
        report["scenarios"][name] = {
            "passed": scenario_passed,
            "bitwise_state_match": exact,
            "differences": differences,
        }
        all_passed &= scenario_passed
        status = "PASS" if scenario_passed else "FAIL"
        exact_note = "bitwise identical" if exact else "within tolerance"
        print(f"{status}: {name} ({exact_note})")

    report_path = baseline_dir / "latest_segment_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nEndpoint segment report: {report_path}")
    return all_passed


def check_solver_compatibility() -> bool:
    """Reject unsupported gas steps before output or physical state changes."""
    simulation, components = build_scenario("wave_baryons_gas")
    simulation.initialize_simulation()
    state = {}
    _capture_component_state(state, "initial", components)
    before = _state_digest(state)
    for order in (4, 6):
        simulation.evolution.order = order
        try:
            simulation.evolution.evolve(simulation.wave_functions)
        except ValueError as error:
            assert "order_of_evolution=2" in str(error)
        else:
            raise AssertionError(f"Gas accepted evolution order {order}")
    gas = components["gas"]
    for dt in (-0.001, float("nan"), float("inf")):
        try:
            gas.drift(dt, None)
        except ValueError as error:
            assert "non-negative" in str(error)
        else:
            raise AssertionError(f"Gas accepted dt={dt}")
    gas.drift(0, None)
    _capture_component_state(state, "initial", components)
    assert before == _state_digest(state), "Rejected steps changed physical state"
    assert simulation.snapshot_directory is None, "Rejected run created output"
    print("PASS: solver compatibility (state unchanged)")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("baseline", "compare", "restart-check", "segment-check", "compatibility-check"),
        help=(
            "Create references, compare them, verify checkpoint continuation, "
            "or verify endpoint composition."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("all",) + SCENARIOS,
        default="all",
        help="Run all scenarios or one named scenario.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Ignored directory holding reference arrays and reports.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow baseline mode to replace an existing reference.",
    )
    return parser.parse_args()


def main() -> int:
    os.chdir(PROJECT_ROOT)
    args = _parse_args()
    baseline_dir = args.baseline_dir.resolve()
    scenario_names = SCENARIOS if args.scenario == "all" else (args.scenario,)

    if args.mode == "baseline":
        passed = create_baselines(
            baseline_dir=baseline_dir,
            scenario_names=scenario_names,
            overwrite=args.overwrite,
        )
    elif args.mode == "compare":
        passed = compare_with_baselines(
            baseline_dir=baseline_dir,
            scenario_names=scenario_names,
        )
    elif args.mode == "restart-check":
        passed = check_midpoint_restarts(
            baseline_dir=baseline_dir,
            scenario_names=scenario_names,
        )
    elif args.mode == "compatibility-check":
        passed = check_solver_compatibility()
    else:
        passed = check_endpoint_segments(
            baseline_dir=baseline_dir,
            scenario_names=scenario_names,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
