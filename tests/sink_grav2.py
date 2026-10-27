from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody


# =====================================================================================================
# TWO SINKS (BLACK HOLES) ORBIT EACH OTHER (units: kpc, Msun, Gyr)
#
# Same set-up as sink_grav1.py (two sinks, no soliton or gas), but the sinks get perpendicular velocities (each one
# moves perpendicular to the line joining them, in opposite directions). V_FACTOR = 1 is a circular orbit; a smaller
# number gives an elongated ellipse, a larger one (up to ~1.4) an ellipse as well. Merging is switched on as in
# sink_grav1.py - it should not happen on an orbit; if it does, the orbit is too elongated.
# =====================================================================================================

# ---------------------------------------------------- sinks
M_SINK = 1.0e7                    # mass of each sink [Msun]
SEPARATION = 10.0                 # initial separation [kpc] (above ~7.5 kpc = L/4 the force in the periodic box is only approximate, so the orbit will not be an exact circle)
V_FACTOR = 1.0                    # speed as a multiple of the circular speed (1.0 = circle)

# ---------------------------------------------------- simulation
BOX = 15.0                        # the box is [-BOX, BOX]^3, i.e. 30 kpc
N = 64                            # grid resolution (the sink-sink force does not depend on it)
H = 5.0e-3                        # time step [Gyr] (no wave function, so no step limit; the sinks choose their own substeps)
TOTAL_TIME = 35.0                 # how long to simulate [Gyr] (see the printed period below; 35 Gyr is about 1.5 orbits)
MERGE_CELLS = 2                   # sinks merge when they are closer than this many cells (and are bound)

dx = 2.0 * BOX / N
MERGE_RADIUS = MERGE_CELLS * dx   # [kpc]
volume = (2.0 * BOX) ** 3

sim = Simulation_Class(

    dim=3, # 3D simulation
    boundaries=[(-BOX, BOX)] * 3, # box, the centre is at (0,0,0)
    N=N, # grid resolution
    total_time=TOTAL_TIME, # total time [Gyr]
    h=H, # time step [Gyr]
    order_of_evolution=2,
    use_gravity=True, # gravity on
    use_units=True, # kpc, Msun, Gyr
    use_sponge=False,
    self_int=False,

)
G = float(sim.G)
softening = 1.5 * dx              # same as the default softening_bh of a sink


def acceleration(r):
    """Acceleration of one sink towards the other at distance r (Gaussian-smoothed source + periodic box)."""
    x = r / softening
    enclosed = erf(x / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * x * x)
    return G * M_SINK * enclosed / r ** 2 - 4.0 * np.pi / 3.0 * G * M_SINK * r / volume


v_circ = np.sqrt(acceleration(SEPARATION) * SEPARATION / 2.0)     # circle of radius SEPARATION / 2 about the centre of mass
period = 2.0 * np.pi * (SEPARATION / 2.0) / v_circ
print(f"sinks {M_SINK:.1e} Msun, separation {SEPARATION} kpc, circular speed of each {v_circ:.3f} kpc/Gyr "
      f"({v_circ / 1.0227:.2f} km/s), period {period:.2f} Gyr, simulating {TOTAL_TIME:.1f} Gyr "
      f"({TOTAL_TIME / period:.2f} orbits, {round(TOTAL_TIME / H)} steps)")

v = V_FACTOR * v_circ
sink = SinkNBody(

    simulation=sim,
    N_sinks=2, # two sinks
    initial_masses=[M_SINK, M_SINK], # their masses
    initial_positions=[[-SEPARATION / 2, 0.0, 0.0], [SEPARATION / 2, 0.0, 0.0]], # on the x axis, symmetric about the centre
    initial_velocities=[[0.0, -v, 0.0], [0.0, v, 0.0]], # perpendicular to the line joining them, opposite directions (the centre of mass stays put)

)
sim.add_baryons(sink)
sim.sink_merging = True # bound sinks closer than the merge radius merge (done in Evolution before every step)
sim.sink_merge_radius = MERGE_RADIUS
sim.initialize_simulation()


# =====================================================================================================
# RUN (after every step the positions are stored and a merge is attempted)
# =====================================================================================================
n_steps = sim.num_steps
trajectory = np.full((n_steps + 1, 2, 3), np.nan)       # [step, sink, xyz]; after a merge only sink 0 is left
trajectory[0] = cp.asnumpy(sink.positions)
merges = []                                             # (time of the merge, energy lost in it)
original_step = sim.evolution._perform_evolution_step
sinks_before = [sink.N]


def step_and_record(wave_functions, total_density, step, save_step):
    if sink.N < sinks_before[0]:                        # the merge happened in Evolution just before this step
        merges.append((step * H, float(sink.E_diss_merge_total)))
        print(f"\nMERGE at t = {step * H:.3f} Gyr, mass {float(sink.masses.sum()):.4e} Msun")
    sinks_before[0] = sink.N
    result = original_step(wave_functions, total_density, step, save_step)
    positions = cp.asnumpy(sink.positions)
    trajectory[step + 1, :len(positions)] = positions
    return result


sim.evolution._perform_evolution_step = step_and_record
sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)

time = np.arange(n_steps + 1) * H
relative = trajectory[:, 1] - trajectory[:, 0]
separation = np.linalg.norm(relative, axis=1)
angle = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))      # NaN after the merge

print(f"\nsink separation: smallest {np.nanmin(separation):.3f}, largest {np.nanmax(separation):.3f} kpc "
      f"(initially {SEPARATION})")
valid = ~np.isnan(angle)
print(f"completed {abs(angle[valid][-1] - angle[valid][0]) / (2 * np.pi):.2f} orbits out of {TOTAL_TIME / period:.2f} planned")
com = np.nanmean(trajectory, axis=1)
print(f"centre-of-mass drift: {np.nanmax(np.linalg.norm(com - com[0], axis=1)):.2e} kpc (should be ~0)")
print("The sinks merged." if merges else f"The sinks did NOT merge (merge below {MERGE_RADIUS:.2f} kpc).")


# =====================================================================================================
# PLOTS (saved into the simulation directory)
# =====================================================================================================
output_dir = sim.evolution.scribe.snapshot_directory
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(trajectory[:, 0, 0], trajectory[:, 0, 1], label="sink 1")
axes[0].plot(trajectory[:, 1, 0], trajectory[:, 1, 1], label="sink 2")
axes[0].plot(trajectory[0, :, 0], trajectory[0, :, 1], "ko", label="start")
axes[0].set_aspect("equal")
axes[0].set_xlabel("x [kpc]")
axes[0].set_ylabel("y [kpc]")
axes[0].set_title("sink orbits")
axes[0].legend()

axes[1].plot(time, separation)
axes[1].axhline(SEPARATION, color="gray", linestyle="--", label="initial separation")
axes[1].axhline(MERGE_RADIUS, color="red", linestyle=":", label=f"merge radius {MERGE_RADIUS:.2f} kpc")
axes[1].set_xlabel("time [Gyr]")
axes[1].set_ylabel("separation [kpc]")
axes[1].set_title("sink separation")
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "sink_grav2.png"), dpi=150)
print(f"Plot saved in: {output_dir}")
