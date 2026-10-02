"""Versioned, atomic checkpoint persistence for simulations."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np


SCHEMA_VERSION = 1


def _replace_with_retry(source: str | os.PathLike[str], target: Path) -> None:
    """Atomically publish a file/directory despite brief Windows file locks."""
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _json_value(value: Any) -> Any:
    """Convert NumPy/CuPy scalars and containers into JSON-safe values."""
    if isinstance(value, cp.ndarray):
        value = cp.asnumpy(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_value(data), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _save_npy(path: Path, value: Any) -> None:
    array = cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def _save_npz(path: Path, **arrays: Any) -> None:
    cpu_arrays = {
        key: cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)
        for key, value in arrays.items()
    }
    with path.open("wb") as handle:
        np.savez(handle, **cpu_arrays)


def _simulation_config(simulation: Any) -> dict[str, Any]:
    return {
        "dim": simulation.dim,
        "boundaries": simulation.boundaries,
        "N": simulation.N,
        "total_time": simulation.total_time,
        "h": simulation.h,
        "order_of_evolution": simulation.order_of_evolution,
        "m_s": simulation.m_s,
        "sponge_V0": simulation.sponge_V0,
        "use_sponge": simulation.use_sponge,
        "use_gravity": simulation.use_gravity,
        "save_max_vals": simulation.save_max_vals,
        "sink_formation": simulation._sink_cfg,
        "sim_units": {
            "dUnits": simulation.dUnits,
            "tUnits": simulation.tUnits,
            "mUnits": simulation.mUnits,
            "eUnits": simulation.eUnits,
        },
        "use_units": simulation.use_units,
        "self_int": simulation.use_self_int,
        "a_s": simulation.a_s,
        "has_static_potential": simulation.static_potential is not None,
        "has_external_density": simulation.external_density is not None,
        "overwrite_density": simulation.overwrite_density,
        "spin": getattr(simulation, "spin", None),
    }


def _wave_metadata(wave: Any, filename: str) -> dict[str, Any]:
    optional_names = (
        "desired_soliton_mass",
        "soliton_mass",
        "scaling_lambda",
        "soliton_radius",
        "soliton_mass_radius",
        "soliton_dynamic_mass",
        "conversion_factor",
    )
    metadata = {
        "file": filename,
        "multiplicity": getattr(wave, "multiplicity", 1),
        "means": getattr(wave, "means", None),
        "momenta": getattr(wave, "momenta", None),
        "st_deviations": getattr(wave, "st_deviations", None),
        "omega": getattr(wave, "omega", 1),
        "packet_type": str(getattr(wave, "packet_type", "checkpoint")),
    }
    metadata["optional"] = {
        name: getattr(wave, name)
        for name in optional_names
        if hasattr(wave, name)
    }
    return metadata


def _save_component(directory: Path, index: int, component: Any) -> dict[str, Any]:
    component_type = component.__class__.__name__
    filename = f"component_{index}_{component_type}.npz"
    path = directory / filename

    if component_type == "Baryons":
        _save_npz(
            path,
            positions=component.positions,
            velocities=component.velocities,
        )
        metadata = {
            "N": component.N,
            "m_particle": component.m_particle,
            "name": getattr(component, "name", "baryons"),
            "scale_radius": getattr(component, "scale_radius", None),
            "truncation_radius": getattr(component, "truncation_radius", None),
        }
    elif component_type == "NBodyGas":
        _save_npz(
            path,
            rho=component.rho,
            vx=component.vx,
            vy=component.vy,
            vz=component.vz,
            E=component.E,
        )
        metadata = {
            "name": component.name,
            "cfl": component.cfl,
            "rho_floor": component.rho_floor,
            "max_substeps": component.max_substeps,
            "gamma": component.gamma,
            "cs": component.cs,
            "tcool": component.tcool,
            "e_floor": component.e_floor,
            "E_radiated": component.E_radiated,
            "rho_ref": component.rho_ref,
        }
    elif component_type == "SinkNBody":
        _save_npz(
            path,
            positions=component.positions,
            velocities=component.velocities,
            mass_bh=component.mass_bh,
            mass_res=component.mass_res,
            masses=component.masses,
        )
        metadata = {
            "N": component.N,
            "capture_radius": component.capture_radius,
            "softening_length": component.softening_length,
            "softening_bh": component.softening_bh,
            "softening_cusp": component.softening_cusp,
            "reservoir_tau": component.reservoir_tau,
            "E_diss_formation_total": component.E_diss_formation_total,
            "E_diss_kin_total": component.E_diss_kin_total,
            "E_diss_kin_last": component.E_diss_kin_last,
            "E_diss_merge_total": getattr(component, "E_diss_merge_total", 0.0),
            "total_accreted_mass": component.total_accreted_mass,
            "accretion_history": component.accretion_history,
        }
    else:
        raise TypeError(f"Unsupported checkpoint component: {component_type}")

    return {
        "type": component_type,
        "file": filename,
        "metadata": _json_value(metadata),
    }


def save_checkpoint(
    simulation: Any,
    wave_functions: list[Any],
    scribe: Any,
    current_step: int,
    current_time: float,
    save_every: int,
    diagnostics_every: int,
) -> Path:
    """Write a complete step checkpoint and atomically publish its manifest."""
    if scribe.snapshot_directory is None:
        raise ValueError("Cannot checkpoint before the output directory exists")

    checkpoint_root = Path(scribe.snapshot_directory) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final_name = f"step_{int(current_step):08d}"
    final_directory = checkpoint_root / final_name
    if final_directory.exists():
        final_directory = checkpoint_root / f"{final_name}_{uuid.uuid4().hex[:8]}"

    temporary_directory = checkpoint_root / f".{final_directory.name}.{uuid.uuid4().hex}.tmp"
    temporary_directory.mkdir(parents=False)

    try:
        waves = []
        for index, wave in enumerate(wave_functions):
            filename = f"wave_{index}.npy"
            _save_npy(temporary_directory / filename, wave.psi)
            waves.append(_wave_metadata(wave, filename))

        components = [
            _save_component(temporary_directory, index, component)
            for index, component in enumerate(simulation.baryonic_matter)
        ]

        external_density_file = None
        if simulation.external_density is not None:
            external_density_file = "external_density.npy"
            _save_npy(
                temporary_directory / external_density_file,
                simulation.external_density,
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "current_step": int(current_step),
            "current_time": float(current_time),
            "save_every": int(save_every),
            "diagnostics_every": int(diagnostics_every),
            "simulation_config": _simulation_config(simulation),
            "waves": waves,
            "components": components,
            "external_density_file": external_density_file,
            "scribe": {
                "accessible_times": scribe.accessible_times,
                "wave_values": scribe.wave_values,
                "energy_log": scribe.energy_log,
                "max_wave_values": scribe.max_wave_vals_during_evolution,
            },
        }
        _atomic_json(temporary_directory / "manifest.json", manifest)
        _replace_with_retry(temporary_directory, final_directory)
    except Exception:
        for child in temporary_directory.glob("*"):
            try:
                child.unlink()
            except OSError:
                pass
        try:
            temporary_directory.rmdir()
        except OSError:
            pass
        raise

    manifest_path = final_directory / "manifest.json"
    _atomic_json(
        checkpoint_root / "latest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(manifest_path.relative_to(checkpoint_root)),
        },
    )
    return manifest_path


def _resolve_manifest(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).resolve()
    if candidate.is_file():
        if candidate.name == "latest.json":
            pointer = json.loads(candidate.read_text(encoding="utf-8"))
            return (candidate.parent / pointer["checkpoint"]).resolve()
        return candidate

    direct_manifest = candidate / "manifest.json"
    if direct_manifest.exists():
        return direct_manifest

    latest = candidate / "latest.json"
    if not latest.exists():
        latest = candidate / "checkpoints" / "latest.json"
    if latest.exists():
        pointer = json.loads(latest.read_text(encoding="utf-8"))
        return (latest.parent / pointer["checkpoint"]).resolve()

    raise FileNotFoundError(f"No checkpoint manifest found under {candidate}")


def _restore_wave(simulation: Any, directory: Path, data: dict[str, Any]) -> Any:
    from resources.Classes.Wave_function_class import Wave_function

    wave = Wave_function.__new__(Wave_function)
    wave.simulation = simulation
    wave.dim = simulation.dim
    wave.boundaries = simulation.boundaries
    wave.N = simulation.N
    wave.total_time = simulation.total_time
    wave.h = simulation.h
    wave.num_steps = simulation.num_steps
    wave.dx = simulation.dx
    wave.grids = simulation.grids
    wave.mass = simulation.mass_s
    wave.h_bar_tilde = simulation.h_bar_tilde
    wave.multiplicity = data.get("multiplicity", 1)
    wave.means = data.get("means") or [0.0] * simulation.dim
    wave.momenta = data.get("momenta") or [0.0] * simulation.dim
    wave.st_deviations = data.get("st_deviations") or [1.0] * simulation.dim
    wave.omega = data.get("omega", 1)
    wave.packet_type = data.get("packet_type", "checkpoint")
    wave.packet_creator = None
    wave.psi = cp.asarray(np.load(directory / data["file"], allow_pickle=False))
    for name, value in data.get("optional", {}).items():
        setattr(wave, name, value)
    return wave


def _restore_component(
    simulation: Any,
    directory: Path,
    data: dict[str, Any],
) -> Any:
    from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
    from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
    from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody

    component_type = data["type"]
    metadata = data["metadata"]
    with np.load(directory / data["file"], allow_pickle=False) as arrays:
        if component_type == "Baryons":
            component = Baryons.__new__(Baryons)
            component.simulation = simulation
            component.N = int(metadata["N"])
            component.m_particle = float(metadata["m_particle"])
            component.positions = cp.asarray(arrays["positions"])
            component.velocities = cp.asarray(arrays["velocities"])
            component.name = metadata.get("name", "baryons")
            component.scale_radius = metadata.get("scale_radius")
            component.truncation_radius = metadata.get("truncation_radius")
            component._density_cache = None
            component._density_cache_valid = False
            return component

        if component_type == "NBodyGas":
            component = NBodyGas.__new__(NBodyGas)
            component.simulation = simulation
            component.name = metadata["name"]
            component.dim = int(simulation.dim)
            component.dx_list = [float(value) for value in simulation.dx]
            component.dx = min(component.dx_list)
            component.cell_volume = float(simulation.dV)
            component.cfl = float(metadata["cfl"])
            component.rho_floor = float(metadata["rho_floor"])
            component.max_substeps = int(metadata["max_substeps"])
            component.gamma = float(metadata["gamma"])
            component.cs = float(metadata["cs"])
            component.tcool = metadata["tcool"]
            component.e_floor = float(metadata["e_floor"])
            component.E_radiated = float(metadata["E_radiated"])
            component.rho_ref = float(metadata["rho_ref"])
            component.rho = cp.asarray(arrays["rho"])
            component.vx = cp.asarray(arrays["vx"])
            component.vy = cp.asarray(arrays["vy"])
            component.vz = cp.asarray(arrays["vz"])
            component.vel = [component.vx, component.vy, component.vz][: component.dim]
            component.E = cp.asarray(arrays["E"])
            return component

        if component_type == "SinkNBody":
            component = SinkNBody.__new__(SinkNBody)
            component.simulation = simulation
            component.N = int(metadata["N"])
            component.positions = cp.asarray(arrays["positions"])
            component.velocities = cp.asarray(arrays["velocities"])
            component.mass_bh = cp.asarray(arrays["mass_bh"])
            component.mass_res = cp.asarray(arrays["mass_res"])
            component.masses = cp.asarray(arrays["masses"])
            component.m_particle = 1.0
            component.name = "sinks"
            component._density_cache = None
            component._density_cache_valid = False
            for name in (
                "capture_radius",
                "softening_length",
                "softening_bh",
                "softening_cusp",
                "reservoir_tau",
                "E_diss_formation_total",
                "E_diss_kin_total",
                "E_diss_kin_last",
                "E_diss_merge_total",
                "total_accreted_mass",
                "accretion_history",
            ):
                setattr(component, name, metadata[name])
            return component

    raise ValueError(f"Unsupported checkpoint component: {component_type}")


def load_checkpoint(
    simulation_class: type,
    path: str | os.PathLike[str],
    static_potential: Any = None,
    output_directory: str | os.PathLike[str] | None = None,
) -> Any:
    """Reconstruct a Simulation_Class instance from a published checkpoint."""
    manifest_path = _resolve_manifest(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {manifest.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )

    config = manifest["simulation_config"]
    if config["has_static_potential"] and static_potential is None:
        raise ValueError(
            "This checkpoint used a static potential. Pass static_potential=... "
            "to Simulation_Class.from_checkpoint()."
        )

    simulation = simulation_class(
        dim=int(config["dim"]),
        boundaries=[tuple(pair) for pair in config["boundaries"]],
        N=int(config["N"]),
        total_time=float(config["total_time"]),
        h=float(config["h"]),
        order_of_evolution=int(config["order_of_evolution"]),
        m_s=float(config["m_s"]),
        sponge_V0=float(config["sponge_V0"]),
        use_sponge=bool(config["use_sponge"]),
        use_gravity=bool(config["use_gravity"]),
        static_potential=static_potential,
        baryonic_model=None,
        save_max_vals=bool(config["save_max_vals"]),
        sink_formation=config["sink_formation"],
        sim_units=config["sim_units"],
        use_units=bool(config["use_units"]),
        self_int=bool(config["self_int"]),
        a_s=float(config["a_s"]),
    )
    directory = manifest_path.parent
    simulation.wave_functions = [
        _restore_wave(simulation, directory, wave_data)
        for wave_data in manifest["waves"]
    ]
    simulation.num_of_w_vects_in_sim = len(simulation.wave_functions)
    simulation.baryonic_matter = [
        _restore_component(simulation, directory, component_data)
        for component_data in manifest["components"]
    ]

    external_file = manifest.get("external_density_file")
    if external_file is not None:
        simulation.external_density = cp.asarray(
            np.load(directory / external_file, allow_pickle=False)
        )
    simulation.overwrite_density = bool(config["overwrite_density"])
    if config.get("spin") is not None:
        simulation.spin = config["spin"]

    simulation.is_restart = True
    simulation.current_step = int(manifest["current_step"])
    simulation.current_time = float(manifest["current_time"])
    if output_directory is None:
        simulation.snapshot_directory = str(manifest_path.parents[2])
    else:
        resumed_output = Path(output_directory).resolve()
        resumed_output.mkdir(parents=True, exist_ok=True)
        simulation.snapshot_directory = str(resumed_output)
    simulation._restart_manifest = manifest
    simulation._restart_scribe_state = manifest.get("scribe", {})
    simulation._restart_save_every = int(manifest["save_every"])
    simulation._restart_diagnostics_every = int(manifest["diagnostics_every"])
    return simulation
