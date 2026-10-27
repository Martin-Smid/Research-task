from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody


# =====================================================================================================
# TWO SINKS (BLACK HOLES) FALL TOWARDS EACH OTHER FROM REST (units: kpc, Msun, Gyr)
#
# No soliton, gas or stars - just two sinks attracting each other in an empty box. The sinks start at rest, so
# they move along a straight line, and once they get closer than MERGE_RADIUS they merge (mass and momentum are
# conserved). The script plots x(t) of both sinks and their separation against the expectation (Newton + periodic box).
# =====================================================================================================

# ---------------------------------------------------- sinks
M_SINK = 1.0e7                    # mass of each sink [Msun]
SEPARATION = 10.0                 # initial separation [kpc]

# ---------------------------------------------------- simulation
BOX = 15.0                        # the box is [-BOX, BOX]^3, i.e. 30 kpc
N = 128                            # grid resolution (the sink-sink force does not depend on it; cell = 0.47 kpc)
H = 1.0e-3                        # time step [Gyr]
TOTAL_TIME = 4.5                  # how long to simulate [Gyr] (the fall takes ~3.9 Gyr, see the printout below)
MERGE_CELLS = 2                   # sinks merge when they are closer than this many cells (and are bound)

dx = 2.0 * BOX / N
MERGE_RADIUS = MERGE_CELLS * dx   # [kpc]

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

sink = SinkNBody(

    simulation=sim,
    N_sinks=2, # two sinks
    initial_masses=[M_SINK, M_SINK], # their masses
    initial_positions=[[-SEPARATION / 2, 0.0, 0.0], [SEPARATION / 2, 0.0, 0.0]], # on the x axis, symmetric about the centre
    initial_velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # both at rest

)
sim.add_baryons(sink)
sim.initialize_simulation()


# =====================================================================================================
# EXPECTATION: two equal masses falling from rest, same force as in the code (softened + periodic-box term)
# =====================================================================================================
G = float(sim.G)
softening = float(sink.softening_bh)
volume = (2.0 * BOX) ** 3


def acceleration(r):
    """Acceleration of one sink towards the other at distance r (Gaussian-smoothed mass + periodic box)."""
    s = np.sqrt(2.0) * softening
    x = r / s
    enclosed = erf(x / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * x * x)
    return G * M_SINK * enclosed / r ** 2 - 4.0 * np.pi / 3.0 * G * M_SINK * r / volume


def reached_merge_radius(t, y):
    return y[0] - MERGE_RADIUS


reached_merge_radius.terminal = True
expected = solve_ivp(lambda t, y: [y[1], -2.0 * acceleration(y[0])], [0.0, 100.0], [SEPARATION, 0.0],
                     events=reached_merge_radius, dense_output=True, rtol=1e-10, atol=1e-12)
t_expected = float(expected.t_events[0][0])
t_newton = np.pi / 2.0 * np.sqrt(SEPARATION ** 3 / (2.0 * G * 2.0 * M_SINK))   # no box, no softening, fall down to r = 0
print(f"sinks {M_SINK:.1e} Msun, separation {SEPARATION} kpc, G = {G:.4e}")
print(f"expected fall time to {MERGE_RADIUS:.2f} kpc: {t_expected:.2f} Gyr "
      f"(plain Newton in infinite space gives {t_newton:.2f} Gyr; the box weakens the force, softening weakens it up close)")
if t_expected > TOTAL_TIME:
    print(f"WARNING: TOTAL_TIME = {TOTAL_TIME} Gyr is shorter than the expected fall time, the sinks will not merge")


# =====================================================================================================
# RUN (after every step the positions are stored and a merge is attempted)
# =====================================================================================================
n_steps = sim.num_steps
trajectory = np.full((n_steps + 1, 2, 3), np.nan)       # [step, sink, xyz]; after a merge only sink 0 is left
trajectory[0] = cp.asnumpy(sink.positions)
merges = []                                             # (time, released kinetic energy)
original_step = sim.evolution._perform_evolution_step


def step_with_merging(wave_functions, total_density, step, save_step):
    result = original_step(wave_functions, total_density, step, save_step)
    if sink.N > 1:
        n_merged, dE = sink.merge_close_sinks(r_merge=MERGE_RADIUS, bound_check=True)
        if n_merged:
            merges.append(((step + 1) * H, dE))
            print(f"\nMERGE at t = {(step + 1) * H:.3f} Gyr, mass {float(sink.masses.sum()):.4e} Msun, "
                  f"released kinetic energy {dE:.3e}")
    positions = cp.asnumpy(sink.positions)
    trajectory[step + 1, :len(positions)] = positions
    return result


sim.evolution._perform_evolution_step = step_with_merging
sim.evolution.evolve(sim.wave_functions, save_every=500, diagnostics_every=500)

time = np.arange(n_steps + 1) * H
separation = np.linalg.norm(trajectory[:, 1] - trajectory[:, 0], axis=1)    # NaN after the merge

if merges:
    print(f"\nThe sinks merged at t = {merges[0][0]:.3f} Gyr (expected {t_expected:.2f} Gyr), "
          f"final mass {float(sink.masses.sum()):.4e} Msun (the sum was {2 * M_SINK:.4e})")
else:
    print(f"\nThe sinks did NOT merge within {TOTAL_TIME} Gyr, smallest separation {np.nanmin(separation):.3f} kpc "
          f"(merge below {MERGE_RADIUS:.2f} kpc)")


# =====================================================================================================
# PLOTS (saved into the simulation directory)
# =====================================================================================================
output_dir = sim.evolution.scribe.snapshot_directory
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(time, trajectory[:, 0, 0], label="sink 1")
axes[0].plot(time, trajectory[:, 1, 0], label="sink 2")
axes[0].set_xlabel("time [Gyr]")
axes[0].set_ylabel("x [kpc]")
axes[0].set_title("sink positions (motion along x only)")
axes[0].legend()

t_plot = time[time <= t_expected]
axes[1].plot(time, separation, label="simulation")
axes[1].plot(t_plot, expected.sol(t_plot)[0], "--", color="gray", label="expected (Newton + periodic box)")
axes[1].axhline(MERGE_RADIUS, color="red", linestyle=":", label=f"merge radius {MERGE_RADIUS:.2f} kpc")
if merges:
    axes[1].axvline(merges[0][0], color="green", linestyle=":", label="merge")
axes[1].set_xlabel("time [Gyr]")
axes[1].set_ylabel("separation [kpc]")
axes[1].set_title("sink separation")
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "sink_grav1.png"), dpi=150)
print(f"Plot saved in: {output_dir}")
