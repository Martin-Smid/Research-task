from resources.Classes.Wave_function_class import Wave_function
import os
import json
import cupy as cp
import numpy as np
from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class

def _load_restart_metadata(restart_dir):
    restart_path = os.path.join(restart_dir, "restart_state.json")
    if not os.path.exists(restart_path):
        raise FileNotFoundError(f"Restart metadata not found: {restart_path}")

    with open(restart_path, "r") as f:
        return json.load(f)


def _load_wave_functions(restart_dir, wave_files):
    wave_arrays = []

    for fname in wave_files:
        path = os.path.join(restart_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Wave restart file not found: {path}")

        wave_arrays.append(cp.asarray(np.load(path)))

    return wave_arrays


def _load_baryonic_component_states(restart_dir, baryonic_component_files):
    states = []

    for fname in baryonic_component_files:
        path = os.path.join(restart_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baryonic restart file not found: {path}")

        loaded = np.load(path, allow_pickle=True)
        state = {key: loaded[key] for key in loaded.files}
        states.append(state)

    return states
def _restore_baryonic_components(sim, baryon_states):
    restored = []

    for state in baryon_states:
        cls_name = str(state["class_name"])

        if cls_name == "Baryons":
            comp = Baryons(
                sim,
                N_particles=int(state["N"]),
                total_mass=float(state["N"]) * float(state["m_particle"]),
            )
            comp.positions = cp.asarray(state["positions"])
            comp.velocities = cp.asarray(state["velocities"])
            comp.m_particle = float(state["m_particle"])
            comp.N = int(state["N"])
            restored.append(comp)

        elif cls_name == "NBodyGas":
            kwargs = {
                "rho": cp.asarray(state["rho"]),
                "vx": cp.asarray(state["vx"]),
                "vy": cp.asarray(state["vy"]),
                "vz": cp.asarray(state["vz"]),
            }
        

        
            comp = NBodyGas(sim, **kwargs)
            comp.E_radiated = float(state.get("E_radiated", 0.0))
            restored.append(comp)

        elif cls_name == "SinkNBody":
            comp = SinkNBody(sim)
            comp.positions = cp.asarray(state["positions"])
            comp.velocities = cp.asarray(state["velocities"])
            comp.mass_bh = cp.asarray(state["mass_bh"])
            comp.mass_res = cp.asarray(state["mass_res"])
            comp.masses = cp.asarray(state["masses"])
            comp.N = int(state["N"])
            comp.E_diss_kin_total = float(state.get("E_diss_kin_total", 0.0))
            comp.E_diss_formation_total = float(state.get("E_diss_formation_total", 0.0))
            restored.append(comp)

        else:
            raise ValueError(f"Unsupported baryonic component type in restart: {cls_name}")

    return restored

def _restore_wave_functions(sim, wave_arrays):
    restored = []

    for psi_arr in wave_arrays:
        wf = Wave_function(
            sim,
            packet_type="blank",
            momenta=[0.0] * sim.dim,
            means=[0.0] * sim.dim,
            st_deviations=[1.0] * sim.dim,
        )
        wf.psi = cp.asarray(psi_arr)
        restored.append(wf)
    return restored

def restart_simulation(restart_dir):
    from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
    from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
    from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
    from resources.Classes.Wave_function_class import Wave_function
    restart_data = _load_restart_metadata(restart_dir)
    sim_cfg = restart_data["simulation_config"]

    wave_arrays = _load_wave_functions(
        restart_dir,
        sim_cfg.get("wave_files", [])
    )
    
    baryon_states = _load_baryonic_component_states(
        restart_dir,
        sim_cfg.get("baryonic_component_files", [])
    )
    sim = Simulation_Class(
        dim=sim_cfg["dim"],
        boundaries=[tuple(b) for b in sim_cfg["boundaries"]],
        N=sim_cfg["N"],
        total_time=sim_cfg["total_time"],
        h=sim_cfg["h"],
        order_of_evolution=restart_data["order"],
        m_s=sim_cfg["m_s"],
        sponge_V0=sim_cfg["sponge_V0"],
        use_sponge=sim_cfg["use_sponge"],
        use_gravity=sim_cfg["use_gravity"],
        static_potential=None,
        baryonic_model=None,
        save_max_vals=sim_cfg["save_max_vals"],
        sink_formation=sim_cfg["sink_formation"],
        sim_units=sim_cfg["sim_units"],
        use_units=sim_cfg["use_units"],
        self_int=sim_cfg["self_int"],
        a_s=sim_cfg["a_s"],
    )
    restored_baryons = _restore_baryonic_components(sim, baryon_states)
    sim.add_baryons(restored_baryons)
    restored_wave_functions = _restore_wave_functions(sim, wave_arrays)
    sim.wave_functions = restored_wave_functions
    sim.num_of_w_vects_in_sim = len(restored_wave_functions)
    sim.is_restart = True
    sim.snapshot_directory = restart_dir
    return sim, restart_data["current_step"], restart_data