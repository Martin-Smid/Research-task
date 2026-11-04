"""Phase 2 of the sink-infall study: drop a sink into a relaxed (soliton) field and watch it fall to the soliton centre.

Starts a new simulation segment from the end of a finished run of spin_gravity_comparison.py (any spin, equal or unequal
fractions; the sink feels the total density of all components), finds the density peak (soliton), puts a sink of mass
M_BH at distance DIST from it along axis AXIS, at rest, and evolves for TOTAL_TIME. Output: a new simulation_* folder
(particle_trajectory.csv = sink track, max_locations.txt = peak track, snapshots).

Usage (repository root, GPU):
    SOURCE=resources/data/simulation_<phase 1 run> python tests/spin/sink_infall.py
Environment variables (defaults): SOURCE (required), M_BH (2.5e7 Msun), DIST (4.0 kpc, the sink field is usable up to L/4),
AXIS (0), NOSINK (0; 1 = control run without the sink, only the peak track is written), TOTAL_TIME (3.0 Gyr), H (5e-4), SAVE_EVERY (200 steps), SMOOTH (1 = smooth_force on), SOFT (1.5, softening length in cells), BOX (9.6, half box length).
The sink does not accrete (sim.sink_accretion stays False), so this is a pure gravitational infall + friction test."""

import os

import cupy as cp
import numpy as np

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody

SOURCE = os.environ["SOURCE"]
M_BH = float(os.environ.get("M_BH", 2.5e7))
DIST = float(os.environ.get("DIST", 4.0))
AXIS = int(os.environ.get("AXIS", 0))
TOTAL_TIME = float(os.environ.get("TOTAL_TIME", 3.0))
H = float(os.environ.get("H", 5e-4))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 200))
SMOOTH = bool(int(os.environ.get("SMOOTH", 1)))
BOX = float(os.environ.get("BOX", 9.6))
SOFT = float(os.environ.get("SOFT", 1.5))
NOSINK = bool(int(os.environ.get("NOSINK", 0)))

sim = Simulation_Class.from_checkpoint(SOURCE)
sim.start_new_segment(total_time=TOTAL_TIME, h=H, save_max_vals=False)
dx = min(sim.dx)

rho = sum(cp.abs(cp.asarray(wave.psi)) ** 2 for wave in sim.wave_functions)
peak = np.unravel_index(int(cp.argmax(rho)), rho.shape)
peak_pos = np.array([float(cp.asnumpy(g)[peak]) for g in sim.grids])
position = peak_pos.copy()
position[AXIS] += DIST
position = (position + BOX) % (2 * BOX) - BOX   # periodic wrap into [-BOX, BOX)
print(f"soliton peak at {peak_pos} (rho_max = {float(rho.max()):.3e}), sink at {position}, M_BH = {M_BH:.2e}")

if not NOSINK:
    sink = SinkNBody(
        simulation=sim,
        N_sinks=1,
        initial_masses=[M_BH],
        initial_positions=[position.tolist()],
        initial_velocities=[[0.0, 0.0, 0.0]],
        capture_radius=2.5 * dx,
        softening_length=SOFT * dx,
        smooth_force=SMOOTH,
    )
    sim.add_baryons(sink)
sim.initialize_simulation()
for wave in sim.wave_functions:
    wave.psi = cp.asarray(wave.psi)
sim.evolution.evolve(sim.wave_functions, save_every=SAVE_EVERY, diagnostics_every=SAVE_EVERY)
print("output:", sim.evolution.scribe.snapshot_directory)
