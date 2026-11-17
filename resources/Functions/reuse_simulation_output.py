import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import cupy as cp

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Wave_function_class import Wave_function

def load_wave_functions(sim, savedir: str, wave_files: List[str]) -> List[Wave_function]:
    wave_paths = _resolve_paths(savedir, wave_files)
    wave_functions = []

    for path in wave_paths:
        psi = np.load(path)

        wf = Wave_function(
            simulation=sim,
            packet_type="blank",
            momenta=[0.0, 0.0, 0.0],
            means=[0.0, 0.0, 0.0],
            st_deviations=[1.0, 1.0, 1.0],
            mass=1,
            omega=1,
        )

        wf.psi = cp.asarray(psi)

        # best-effort refresh of derived quantities
        try:
            wf.soliton_radius = wf.calculate_soliton_radius()
        except Exception:
            pass

        try:
            wf.soliton_mass_radius = wf.calculate_soliton_mass_radius()
        except Exception:
            pass

        try:
            wf.soliton_dynamic_mass = wf.calculate_dynamic_core_mass()
        except Exception:
            pass

        wave_functions.append(wf)

    return wave_functions

def load_restart_manifest(savedir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(savedir, "restart_state.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing restart manifest: {manifest_path}")

    with open(manifest_path, "r") as f:
        return json.load(f)


def _resolve_paths(savedir: str, file_list: List[str]) -> List[str]:
    return [os.path.join(savedir, f) for f in file_list]




def _restore_baryons_manual(sim, data) -> Baryons:
    obj = Baryons.__new__(Baryons)
    obj.simulation = sim
    obj.positions = cp.asarray(data["positions"])
    obj.velocities = cp.asarray(data["velocities"])
    obj.m_particle = float(data["m_particle"])
    obj.N = int(data["N"])
    return obj


def _restore_gas_manual(sim, data) -> NBodyGas:
    obj = NBodyGas.__new__(NBodyGas)
    obj.simulation = sim

    obj.rho = cp.asarray(data["rho"])
    obj.vx = cp.asarray(data["vx"])
    obj.vy = cp.asarray(data["vy"])
    obj.vz = cp.asarray(data["vz"])

    E = data["E"]
    obj.E = None if E is None else cp.asarray(E)

    obj.vel = [obj.vx, obj.vy, obj.vz]
    obj.cell_volume = sim.dV

    obj.E_radiated = float(data["E_radiated"]) if "E_radiated" in data else 0.0
    obj.name = str(data["name"]) if "name" in data else "gas"

    cs_val = data["cs"] if "cs" in data else None
    obj.cs = None if cs_val is None else float(cs_val)

    obj.gamma = float(data["gamma"]) if "gamma" in data and data["gamma"] is not None else 5/3
    obj.tcool = None if data.get("tcool") in (None, "None", "none", "") else float(data.get("tcool"))
    obj.e_floor = float(data["e_floor"]) if "e_floor" in data and data["e_floor"] is not None else 0.0
    obj.rho_floor = float(data["rho_floor"]) if "rho_floor" in data and data["rho_floor"] is not None else 0.0
    obj.cfl = float(data["cfl"]) if "cfl" in data and data["cfl"] is not None else 0.4
    obj.dx = float(data["dx"]) if "dx" in data and data["dx"] is not None else float(min(sim.dx))
    obj.max_substeps = int(data["max_substeps"]) if "max_substeps" in data and data["max_substeps"] is not None else 100
    obj.dim = int(data["dim"]) if "dim" in data and data["dim"] is not None else sim.dim

    if "dx_list" in data and data["dx_list"] is not None:
        obj.dx_list = list(data["dx_list"])
    else:
        obj.dx_list = list(sim.dx)

    return obj


def _restore_sink_manual(sim, data) -> SinkNBody:
    obj = SinkNBody.__new__(SinkNBody)

    obj.simulation = sim

    # Basic phase-space state
    obj.positions = cp.asarray(data["positions"], dtype=cp.float64)
    obj.velocities = cp.asarray(data["velocities"], dtype=cp.float64)

    # Mass state
    obj.mass_bh = cp.asarray(data.get("mass_bh", data.get("masses")), dtype=cp.float64)
    obj.mass_res = cp.asarray(
        data.get("mass_res", cp.zeros_like(obj.mass_bh)),
        dtype=cp.float64
    )

    if "masses" in data:
        obj.masses = cp.asarray(data["masses"], dtype=cp.float64)
    else:
        obj.masses = obj.mass_bh + obj.mass_res

    obj.N = int(data.get("N", obj.masses.size))

    # Spatial/accretion parameters
    min_dx = float(min(sim.dx))

    obj.capture_radius = float(data.get("capture_radius", 2.5 * min_dx))
    obj.softening_length = float(data.get("softening_length", 1.5 * min_dx))
    obj.softening_bh = float(data.get("softening_bh", obj.softening_length))
    obj.softening_cusp = float(data.get("softening_cusp", obj.capture_radius))

    # Reservoir/accretion bookkeeping
    obj.reservoir_tau = float(data.get("reservoir_tau", 10.0 * sim.h))

    obj.E_diss_formation_total = float(data.get("E_diss_formation_total", 0.0))
    obj.E_diss_kin_total = float(data.get("E_diss_kin_total", 0.0))
    obj.E_diss_kin_last = float(data.get("E_diss_kin_last", 0.0))

    obj.total_accreted_mass = float(data.get("total_accreted_mass", 0.0))
    obj.accretion_history = data.get("accretion_history", [])

    # Optional merge bookkeeping, used only if sinks merge later
    obj.E_diss_merge_total = float(data.get("E_diss_merge_total", 0.0))

    return obj

def load_baryonic_components(sim, savedir: str, component_files: List[str]) -> List[Any]:
    component_paths = _resolve_paths(savedir, component_files)
    restored = []

    for path in component_paths:
        data = np.load(path, allow_pickle=True)
        cls_name = str(data["class_name"])

        if cls_name == "Baryons":
            comp = _restore_baryons_manual(sim, data)
        elif cls_name == "NBodyGas":
            comp = _restore_gas_manual(sim, data)
        elif cls_name == "SinkNBody":
            comp = _restore_sink_manual(sim, data)
        else:
            raise ValueError(f"Unsupported baryonic component class: {cls_name}")

        restored.append(comp)

    return restored


def load_saved_state_into_sim(sim, savedir: str):
    """
    Load the latest saved checkpoint from `savedir` into an already created Simulation_Class instance.

    Returns
    -------
    start_step, current_time
    """
    manifest = load_restart_manifest(savedir)
    cfg = manifest["simulation_config"]

    # sanity checks
    if int(cfg["dim"]) != sim.dim:
        raise ValueError(f"dim mismatch: saved={cfg['dim']} current={sim.dim}")
    if int(cfg["N"]) != sim.N:
        raise ValueError(f"N mismatch: saved={cfg['N']} current={sim.N}")
    if [list(b) for b in cfg["boundaries"]] != [list(b) for b in sim.boundaries]:
        raise ValueError("boundaries mismatch")

    wave_files = cfg.get("wave_files", [])
    component_files = cfg.get("baryonic_component_files", [])

    sim.wave_functions = load_wave_functions(sim, savedir, wave_files)
    sim.baryonic_matter = load_baryonic_components(sim, savedir, component_files)

    sim.is_restart = True
    sim.snapshot_directory = savedir

    sim.initialize_simulation()

    sim.snapshot_directory = savedir
    sim.evolution.scribe.snapshot_directory = savedir

    start_step = int(manifest["current_step"])
    current_time = float(manifest["current_time"])

    return start_step, current_time

def _embedded_slice(old_N, new_N, shift, dx):
    """
    Return slice where an old_N block should be inserted into a new_N grid,
    shifted by physical distance `shift`.

    Works best when shift is close to an integer multiple of dx.
    """
    base_start = (new_N - old_N) // 2
    shift_cells = int(round(shift / dx))

    start = base_start + shift_cells
    end = start + old_N

    if start < 0 or end > new_N:
        raise ValueError(
            f"Shift {shift} is too large: block would occupy [{start}:{end}] "
            f"inside grid of size {new_N}."
        )

    actual_shift = shift_cells * dx
    if abs(actual_shift - shift) > 1e-8:
        print(
            f"Warning: requested shift {shift:.6g}, "
            f"using nearest grid-aligned shift {actual_shift:.6g}"
        )

    return slice(start, end)


def _embed_array(old_arr, new_shape, slices, fill_value=0.0):
    """
    Embed old_arr into a larger array with shape new_shape.
    """
    old_arr = cp.asarray(old_arr)
    new_arr = cp.full(new_shape, fill_value, dtype=old_arr.dtype)
    new_arr[slices] = old_arr
    return new_arr


def _wrap_positions_to_box(positions, boundaries):
    """
    Wrap particle positions periodically into simulation boundaries.
    """
    lows = cp.asarray([b[0] for b in boundaries])
    highs = cp.asarray([b[1] for b in boundaries])
    box = highs - lows

    return (positions - lows) % box + lows

def load_saved_state_into_larger_sim(
    sim,
    savedir: str,
    shift=(0.0, 0.0, 0.0),
    gas_rho_fill=1e-12,
    gas_E_fill=0.0,
):
    """
    Load a saved smaller simulation into a larger simulation box.

    This assumes that the physical grid spacing is unchanged, e.g.
    old [-25,25], N=128 -> new [-50,50], N=256.

    Parameters
    ----------
    sim:
        Already created larger Simulation_Class instance.
    savedir:
        Directory with restart_state.json and saved component files.
    shift:
        Physical shift applied to the loaded system, e.g. (20, 20, 0).
    gas_rho_fill:
        Density used outside the inserted gas block.
    gas_E_fill:
        Energy density used outside the inserted gas block.
    """

    manifest = load_restart_manifest(savedir)
    cfg = manifest["simulation_config"]

    old_N = int(cfg["N"])
    old_boundaries = cfg["boundaries"]

    if int(cfg["dim"]) != sim.dim:
        raise ValueError(f"dim mismatch: saved={cfg['dim']} current={sim.dim}")

    old_dx = [
        (old_boundaries[i][1] - old_boundaries[i][0]) / old_N
        for i in range(sim.dim)
    ]

    new_dx = list(sim.dx)

    for i in range(sim.dim):
        if not np.isclose(old_dx[i], new_dx[i], rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"dx mismatch in axis {i}: saved dx={old_dx[i]}, new dx={new_dx[i]}. "
                "This loader only embeds without interpolation, so dx must match."
            )

    if old_N > sim.N:
        raise ValueError(f"Cannot embed old_N={old_N} into new N={sim.N}")

    shift = tuple(float(s) for s in shift)

    slices = tuple(
        _embedded_slice(
            old_N=old_N,
            new_N=sim.N,
            shift=shift[i],
            dx=new_dx[i],
        )
        for i in range(sim.dim)
    )

    new_shape = tuple([sim.N] * sim.dim)

    wave_files = cfg.get("wave_files", [])
    component_files = cfg.get("baryonic_component_files", [])

    # Load old components first
    sim.wave_functions = load_wave_functions(sim, savedir, wave_files)
    sim.baryonic_matter = load_baryonic_components(sim, savedir, component_files)

    # Embed wave functions
    for wf in sim.wave_functions:
        old_psi = wf.psi
        wf.psi = _embed_array(
            old_psi,
            new_shape=new_shape,
            slices=slices,
            fill_value=0.0,
        )

        try:
            wf.soliton_radius = wf.calculate_soliton_radius()
        except Exception:
            pass

        try:
            wf.soliton_mass_radius = wf.calculate_soliton_mass_radius()
        except Exception:
            pass

        try:
            wf.soliton_dynamic_mass = wf.calculate_dynamic_core_mass()
        except Exception:
            pass

    # Shift particle components and embed gas
    for comp in sim.baryonic_matter:

        # Collisionless particles / sink particles
        if hasattr(comp, "positions"):
            comp.positions = cp.asarray(comp.positions) + cp.asarray(shift)
            comp.positions = _wrap_positions_to_box(comp.positions, sim.boundaries)

        # Grid gas
        if isinstance(comp, NBodyGas):
            comp.rho = _embed_array(
                comp.rho,
                new_shape=new_shape,
                slices=slices,
                fill_value=gas_rho_fill,
            )

            comp.vx = _embed_array(
                comp.vx,
                new_shape=new_shape,
                slices=slices,
                fill_value=0.0,
            )
            comp.vy = _embed_array(
                comp.vy,
                new_shape=new_shape,
                slices=slices,
                fill_value=0.0,
            )
            comp.vz = _embed_array(
                comp.vz,
                new_shape=new_shape,
                slices=slices,
                fill_value=0.0,
            )

            if comp.E is not None:
                comp.E = _embed_array(
                    comp.E,
                    new_shape=new_shape,
                    slices=slices,
                    fill_value=gas_E_fill,
                )

            comp.vel = [comp.vx, comp.vy, comp.vz]
            comp.cell_volume = sim.dV
            comp.dx = float(min(sim.dx))
            comp.dx_list = list(sim.dx)
            comp.dim = sim.dim

    sim.is_restart = True
    sim.snapshot_directory = savedir

    sim.initialize_simulation()

    sim.snapshot_directory = savedir
    sim.evolution.scribe.snapshot_directory = savedir

    start_step = int(manifest["current_step"])
    current_time = float(manifest["current_time"])

    return start_step, current_time