"""Minimal circular-orbit test: a sink in the middle of the box and one massless test particle orbiting it.

The Kepler velocity is set analytically, the simulation is run and two plots (trajectory and radial deviation
from the starting radius) are saved into the data directory of this run."""

import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# VALUES TO CHOOSE (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # box is [-BOX, BOX]^3
N = 64                              # grid resolution
TOTAL_TIME = 1.5                    # simulated time [Gyr]
H = 1.0e-3                          # time step [Gyr]
SINK_MASS = 5.0e10                  # mass of the central sink [Msun]
SINK_POSITION = (0.0, 0.0, 0.0)     # position of the sink
PARTICLE_POSITION = (2.0, 0.0, 0.0) # starting position of the particle (keep it well inside the box)
SAVE_EVERY = 100                    # how often (in steps) density snapshots are written


def kepler_circular_velocity(sim, sink_mass, particle_position, sink_position=(0.0, 0.0, 0.0)):
    """Circular-orbit velocity vector of a particle around a point mass in the periodic box.

    v^2 = G M / r - (4 pi / 3) G M r^2 / L^3: the periodic Poisson solver removes the mean density, which
    weakens the force by (4 pi / 3) G M r / L^3 (6.5 % at r = L/4). The 1-3 % axis anisotropy of the lattice remains.
    The direction is tangential (counter-clockwise in the xy plane, z component is zero)."""

    offset = np.array(particle_position, dtype=float) - np.array(sink_position, dtype=float)
    r = np.linalg.norm(offset)
    box_length = sim.boundaries[0][1] - sim.boundaries[0][0]
    speed = np.sqrt(float(sim.G) * sink_mass * (1.0 / r - 4.0 * np.pi / 3.0 * r ** 2 / box_length ** 3))
    tangent = np.array([-offset[1], offset[0], 0.0]) / np.hypot(offset[0], offset[1])
    return speed * tangent


# ---------------------------------------------------------------------------------------------------------
# SIMULATION, SINK, PARTICLE
# ---------------------------------------------------------------------------------------------------------
sim = Simulation_Class(

    dim=3,                              # 3D simulation
    boundaries=[(-BOX, BOX)] * 3,       # spatial boundaries
    N=N,                                # grid resolution
    total_time=TOTAL_TIME,              # total simulation time [Gyr]
    h=H,                                # time step [Gyr]
    order_of_evolution=2,               # order of the evolution scheme
    use_gravity=True,                   # gravity on
    use_sponge=False,                   # no absorbing boundary
    self_int=False,                     # no self-interaction

)

sink = SinkNBody(

    simulation=sim,                             # simulation the sink belongs to
    N_sinks=1,                                  # one sink
    initial_masses=[SINK_MASS],                 # its mass [Msun]
    initial_positions=[list(SINK_POSITION)],    # its position
    initial_velocities=[[0.0, 0.0, 0.0]],       # at rest

)

particle = Baryons(

    simulation=sim,                 # simulation the particle belongs to
    N_particles=1,                  # one test particle
    total_mass=1.0,                 # negligible mass [Msun], so it does not pull on the sink
    init_profile="uniform",         # irrelevant, position and velocity are overwritten below

)
velocity = kepler_circular_velocity(sim, SINK_MASS, PARTICLE_POSITION, SINK_POSITION)
particle.positions = cp.asarray([PARTICLE_POSITION], dtype=cp.float64)
particle.velocities = cp.asarray([velocity], dtype=cp.float64)
particle.invalidate_density_cache()

sim.add_baryons(sink)
sim.add_baryons(particle)
sim.initialize_simulation()

r0 = np.linalg.norm(np.array(PARTICLE_POSITION) - np.array(SINK_POSITION))
period = 2.0 * np.pi * r0 / np.linalg.norm(velocity)
print(f"Kepler speed {np.linalg.norm(velocity):.2f} kpc/Gyr, orbital period {period:.4f} Gyr, "
      f"{TOTAL_TIME / period:.2f} orbits in {TOTAL_TIME} Gyr")


# ---------------------------------------------------------------------------------------------------------
# EVOLUTION (the particle position is stored after every step)
# ---------------------------------------------------------------------------------------------------------
trajectory = [cp.asnumpy(particle.positions)[0].copy()]
original_step = sim.evolution._perform_evolution_step


def recorded_step(wave_functions, total_density, step, save_step):
    result = original_step(wave_functions, total_density, step, save_step)
    trajectory.append(cp.asnumpy(particle.positions)[0].copy())
    return result


sim.evolution._perform_evolution_step = recorded_step
sim.evolution.evolve(sim.wave_functions, save_every=SAVE_EVERY)

trajectory = np.array(trajectory)
time = np.arange(len(trajectory)) * H
radius = np.linalg.norm(trajectory - np.array(SINK_POSITION), axis=1)
output_dir = sim.evolution.scribe.snapshot_directory


# ---------------------------------------------------------------------------------------------------------
# PLOTS (saved into the data directory of this run)
# ---------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(trajectory[:, 0], trajectory[:, 1], label="particle trajectory")
ax.plot(*SINK_POSITION[:2], "g+", markersize=12, label="sink")
ax.plot(trajectory[0, 0], trajectory[0, 1], "bo", label="start")
ax.plot(trajectory[-1, 0], trajectory[-1, 1], "o", color="orange", label="end")
ax.set_aspect("equal")
ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.legend()
fig.savefig(os.path.join(output_dir, "orbit_trajectory.png"), dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(time, radius - r0, label="r - r0")
ax.axhline(0.0, color="gray", linestyle="--", label="reference: r = r0")
ax.set_xlabel("time [Gyr]")
ax.set_ylabel("r - r0 [kpc]")
ax.legend()
fig.savefig(os.path.join(output_dir, "orbit_radial_deviation.png"), dpi=150)

print(f"Plots saved in {output_dir}")
