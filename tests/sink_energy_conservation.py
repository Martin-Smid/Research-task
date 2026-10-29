from pathlib import Path
import os
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np
import pandas as pd
from scipy.special import erf

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody


# =====================================================================================================
# TEST A: ENERGY CONSERVATION OF SINKS WITH AND WITHOUT SinkNBody(smooth_force) (units: kpc, Msun, Gyr)
#
# Part 1, binary: two equal sinks on a circular orbit in the periodic box, no psi. The energy is computed twice,
#   (a) from the trajectory with the analytic Gaussian pair energy (s = eps for the point receiver, s = sqrt(2) eps with
#       smooth_force) plus the periodic mean-density term -2 pi G M^2 r^2 / (3 V), independent of the code,
#   (b) from the E_total column logged by Diagnostics (grid model).
#   Reported: the largest deviation from the initial value relative to the kinetic energy, and the same for 2x more steps
#   (a symplectic scheme: the deviation should fall like h^2, and (a) must not depend on the kernel).
# Part 2, sink through uniform psi (set-up of sink_friction.py, N = 128, M_Q = 17.4): E_total from Diagnostics
#   (K of psi + W + K_sink, including the sink-psi interaction) should stay constant; its change is compared with the
#   energy the sink gave to psi, M v0 (v0 - v). Cases: softening 0.5 and 1.5 cells, smooth_force off/on, H and H / 2.
# The outputs of the single runs are deleted.
# =====================================================================================================

ANALYTIC_SINK_ENERGY = True       # Diagnostics: analytic sink-sink energy instead of the grid integral (always on with smooth_force)

# ---------------------------------------------------- part 1
BOX_B = 10.0
N_B = 64
M_B = 5.0e10                      # mass of each sink [Msun]
R0 = 2.0                          # separation [kpc]
EPS_B = (0.3, 0.6, 1.0)           # softening_bh [kpc]
N_ORBITS = 3
STEPS_PER_ORBIT = (200, 400)

# ---------------------------------------------------- part 2
V_SINK = 30.0
BOX = 9.6
N = 128                           # cell 0.15 kpc
M_S = 2.5e-22
MACH_Q = 17.4
RHO = 3.0e5
TRAVEL_FRACTION = 0.5
SOFTENING_CELLS = (0.5, 1.5)
H_VALUES = (4.0e-4, 2.0e-4)       # the stability limit for this grid is 6.1e-4


def make_sim(**kwargs):
    return Simulation_Class(dim=3, order_of_evolution=2, use_gravity=True, use_units=True, use_sponge=False,
                            self_int=False, **kwargs)


def read_energy(sim):
    directory = sim.evolution.scribe.snapshot_directory
    energy = pd.read_csv(os.path.join(directory, "energy.csv"))
    shutil.rmtree(directory, ignore_errors=True)
    return energy


def run_binary(eps, smooth, steps_per_orbit):
    sim = make_sim(boundaries=[(-BOX_B, BOX_B)] * 3, N=N_B, total_time=1.0, h=1.0e-3)
    G = float(sim.G)
    volume = (2.0 * BOX_B) ** 3
    s = eps * (np.sqrt(2.0) if smooth else 1.0)
    x = R0 / s
    enclosed = erf(x / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * x * x)
    a = G * M_B * enclosed / R0 ** 2 - 4.0 * np.pi / 3.0 * G * M_B * R0 / volume
    v = np.sqrt(a * R0 / 2.0)
    period = 2.0 * np.pi * (R0 / 2.0) / v
    h = period / steps_per_orbit
    n_steps = N_ORBITS * steps_per_orbit
    sink = SinkNBody(simulation=sim, N_sinks=2, initial_masses=[M_B, M_B],
                     initial_positions=[[-R0 / 2, 0.0, 0.0], [R0 / 2, 0.0, 0.0]],
                     initial_velocities=[[0.0, -v, 0.0], [0.0, v, 0.0]], softening_bh=eps, smooth_force=smooth)
    sim.h, sim.total_time, sim.num_steps = h, n_steps * h, n_steps
    sim.add_baryons(sink)
    sim.initialize_simulation()
    sim.evolution.diagnostics.analytic_sink_energy = ANALYTIC_SINK_ENERGY
    pos, vel = [cp.asnumpy(sink.positions).copy()], [cp.asnumpy(sink.velocities).copy()]
    original_step = sim.evolution._perform_evolution_step

    def recorded_step(wave_functions, total_density, step, save_step):
        result = original_step(wave_functions, total_density, step, save_step)
        pos.append(cp.asnumpy(sink.positions).copy())
        vel.append(cp.asnumpy(sink.velocities).copy())
        return result

    sim.evolution._perform_evolution_step = recorded_step
    sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=steps_per_orbit // 4)
    pos, vel = np.array(pos), np.array(vel)
    energy = read_energy(sim)

    r = np.linalg.norm(pos[:, 1] - pos[:, 0], axis=1)
    kinetic = 0.5 * M_B * np.sum(vel ** 2, axis=(1, 2))
    potential = M_B ** 2 * (-G * erf(r / (np.sqrt(2.0) * s)) / r) - 2.0 * np.pi * G * M_B ** 2 * r ** 2 / (3.0 * volume)
    total = kinetic + potential
    dev_analytic = np.abs(total - total[0]).max() / kinetic[0]
    e_log = energy["E_total"].to_numpy()
    dev_logged = np.abs(e_log - e_log[0]).max() / kinetic[0]
    momentum = np.abs(vel.sum(axis=1)).max() / (M_B * v)
    return dict(dev_analytic=dev_analytic, dev_logged=dev_logged, momentum=momentum, potential_over_kinetic=-potential[0] / kinetic[0])


def run_psi(softening_cells, smooth, h):
    sim0_dx = 2.0 * BOX / N
    eps = softening_cells * sim0_dx
    total_time = TRAVEL_FRACTION * 2.0 * BOX / V_SINK
    sim = make_sim(boundaries=[(-BOX, BOX)] * 3, N=N, total_time=total_time, h=h, m_s=M_S)
    G, hbar = float(sim.G), float(sim.h_bar_tilde)
    m_sink = hbar * V_SINK / (G * MACH_Q)
    wave_vector = Wave_vector_class(
        packet_type="resources/solitons/GroundState(1).dat", means=[0, 0, 0], st_deviations=[0.5, 0.5, 0.5], simulation=sim,
        mass=1, omega=1, momenta=[0, 0.0, 0], spin=0, desired_soliton_mass=5.0e7)
    sim.add_wave_vector(wave_vector)
    sink = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[m_sink],
                     initial_positions=[[-TRAVEL_FRACTION * BOX, 0.0, 0.0]], initial_velocities=[[V_SINK, 0.0, 0.0]],
                     softening_bh=eps, smooth_force=smooth)
    sim.add_baryons(sink)
    sim.initialize_simulation()
    sim.evolution.diagnostics.analytic_sink_energy = ANALYTIC_SINK_ENERGY
    psi = sim.wave_functions[0].psi
    sim.wave_functions[0].psi = cp.full(psi.shape, np.sqrt(RHO), dtype=cp.complex128)
    n_steps = sim.num_steps
    sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=max(1, n_steps // 20))
    v_end = float(cp.asnumpy(sink.velocities)[0, 0])
    energy = read_energy(sim)
    e = energy["E_total"].to_numpy()
    k_sink = 0.5 * m_sink * (V_SINK ** 2 - v_end ** 2)           # kinetic energy the sink lost
    out = dict(slowdown=1.0 - v_end / V_SINK, k_sink_lost=k_sink, dE_end=e[-1] - e[0], dE_max=np.abs(e - e[0]).max(),
               E0=e[0], n_samples=len(e))
    if "E_tot_cons" in energy:
        ec = energy["E_tot_cons"].to_numpy()
        out["dEcons_max"] = np.abs(ec - ec[0]).max()
    return out


def main():
    print("PART 1: binary, deviation of the total energy / initial kinetic energy over", N_ORBITS, "orbits")
    print(f"{'eps':>5} {'smooth':>6} {'steps/orbit':>11} {'|U|/K':>6} {'analytic':>10} {'logged':>10} {'momentum':>9}")
    for eps in EPS_B:
        for smooth in (False, True):
            for steps in STEPS_PER_ORBIT:
                r = run_binary(eps, smooth, steps)
                print(f"{eps:5.2f} {str(smooth):>6} {steps:11d} {r['potential_over_kinetic']:6.2f} {r['dev_analytic']:10.2e} "
                      f"{r['dev_logged']:10.2e} {r['momentum']:9.1e}", flush=True)

    print("\nPART 2: sink in uniform psi (M_Q = 17.4), E_total from Diagnostics")
    print(f"{'soft[cells]':>11} {'smooth':>6} {'H':>8} {'slowdown':>9} {'K_lost':>10} {'dE_end':>10} {'dE_max':>10} "
          f"{'dE_max/K_lost':>13} {'dEcons_max':>10}")
    for cells in SOFTENING_CELLS:
        for smooth in (False, True):
            for h in H_VALUES:
                r = run_psi(cells, smooth, h)
                print(f"{cells:11.2f} {str(smooth):>6} {h:8.1e} {r['slowdown']:9.4f} {r['k_sink_lost']:10.3e} {r['dE_end']:10.2e} "
                      f"{r['dE_max']:10.2e} {r['dE_max'] / r['k_sink_lost']:13.2e} {r.get('dEcons_max', float('nan')):10.2e}",
                      flush=True)


if __name__ == "__main__":
    main()
