"""Test: two sinks attract each other (a circular binary in the periodic box, no psi, gas or particles).

Before sink-sink gravity was added, the separation of two sinks stayed exactly constant (a sink moved only in the
field of the environment).

Set-up
    Two equal sinks of mass M at distance R0, each orbiting the centre of mass with speed v (a circle). The
    acceleration is computed INDEPENDENTLY of the code: force of a Gaussian source of width eps (point receiver) plus the
    mean-density term of the periodic Poisson solver, -(4 pi / 3) G M r / V (the same term as in the Kepler test in the periodic box).

Checks after N_ORBITS revolutions
    1) the centre of mass does not move and the total momentum is conserved (the force is symmetric),
    2) the angular momentum is conserved (the force is central),
    3) the separation stays at R0 (circular orbit) within MAX_SEPARATION_ERROR,
    4) the angular velocity matches the analytic value within MAX_OMEGA_ERROR."""

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np
from scipy.special import erf

from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # the box is [-BOX, BOX]^3 [kpc]
N_GRID = 64                         # grid resolution (the sink-sink force does not depend on it)
M_SINK = 5.0e10                     # mass of each sink [Msun]
R0 = 2.0                            # initial separation [kpc] (well below L/4, above the softening)
N_ORBITS = 3                        # how many orbits to integrate
STEPS_PER_ORBIT = 200               # steps h per orbit
MAX_MOMENTUM_ERROR = 1.0e-9         # total momentum / (M v)
MAX_ANGULAR_MOMENTUM_ERROR = 1.0e-6 # relative change of the angular momentum
MAX_SEPARATION_ERROR = 5.0e-3       # relative deviation of the separation from R0
MAX_OMEGA_ERROR = 5.0e-3            # relative error of the angular velocity


def expected_acceleration(G, softening, volume):
    """Acceleration of one sink due to the other at separation R0 (analytic, independent of the code)."""
    x = R0 / softening
    enclosed = erf(x / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * x * x)
    return G * M_SINK * enclosed / R0 ** 2 - 4.0 * np.pi / 3.0 * G * M_SINK * R0 / volume


def main():
    sim = Simulation_Class(

        dim=3,
        boundaries=[(-BOX, BOX)] * 3,
        N=N_GRID,
        total_time=1.0,                 # overwritten below, once the period is known
        h=1.0e-3,                       # overwritten below
        order_of_evolution=2,
        use_gravity=True,
        use_units=True,
        use_sponge=False,
        self_int=False,

    )
    G = float(sim.G)
    sink = SinkNBody(

        simulation=sim,
        N_sinks=2,
        initial_masses=[M_SINK, M_SINK],
        initial_positions=[[-R0 / 2, 0.0, 0.0], [R0 / 2, 0.0, 0.0]],
        initial_velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],

    )
    volume = (2.0 * BOX) ** 3
    a = expected_acceleration(G, sink.softening_bh, volume)
    v = np.sqrt(a * R0 / 2.0)                   # circle of radius R0/2: v^2 / (R0/2) = a
    omega = v / (R0 / 2.0)
    period = 2.0 * np.pi / omega
    h = period / STEPS_PER_ORBIT
    n_steps = N_ORBITS * STEPS_PER_ORBIT
    sim.h, sim.total_time, sim.num_steps = h, n_steps * h, n_steps     # the evolution takes them only in initialize_simulation
    sink.velocities = cp.asarray([[0.0, -v, 0.0], [0.0, v, 0.0]], dtype=cp.float64)
    print(f"2 sinks of {M_SINK:.1e} Msun, separation {R0} kpc, v = {v:.2f} kpc/Gyr, period {period:.5f} Gyr, "
          f"{n_steps} steps of {h:.2e} Gyr")

    sim.add_baryons(sink)
    sim.initialize_simulation()

    traj = [cp.asnumpy(sink.positions).copy()]
    vel = [cp.asnumpy(sink.velocities).copy()]
    original_step = sim.evolution._perform_evolution_step

    def recorded_step(wave_functions, total_density, step, save_step):
        result = original_step(wave_functions, total_density, step, save_step)
        traj.append(cp.asnumpy(sink.positions).copy())
        vel.append(cp.asnumpy(sink.velocities).copy())
        return result

    sim.evolution._perform_evolution_step = recorded_step
    sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)
    traj, vel = np.array(traj), np.array(vel)
    shutil.rmtree(sim.evolution.scribe.snapshot_directory, ignore_errors=True)      # the test leaves no output behind

    rel = traj[:, 1] - traj[:, 0]
    separation = np.linalg.norm(rel, axis=1)
    angle = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    omega_measured = (angle[-1] - angle[0]) / ((len(traj) - 1) * h)

    momentum = np.abs(vel.sum(axis=1)).max() / (M_SINK * v)
    com_shift = np.abs(traj.mean(axis=1) - traj[0].mean(axis=0)).max()
    ang_mom = M_SINK * (traj[:, :, 0] * vel[:, :, 1] - traj[:, :, 1] * vel[:, :, 0]).sum(axis=1)
    ang_mom_error = np.abs(ang_mom / ang_mom[0] - 1.0).max()
    separation_error = np.abs(separation / R0 - 1.0).max()
    omega_error = abs(omega_measured / omega - 1.0)

    print(f"separation: min {separation.min():.4f}, max {separation.max():.4f} kpc (deviation {separation_error:.2e})")
    print(f"angular velocity: measured {omega_measured:.4f}, analytic {omega:.4f} (error {omega_error:.2e})")
    print(f"total momentum / (M v): {momentum:.2e}, centre-of-mass shift {com_shift:.2e} kpc, "
          f"angular momentum change {ang_mom_error:.2e}")

    assert momentum < MAX_MOMENTUM_ERROR, f"the total momentum of the sinks is not conserved ({momentum:.2e})"
    assert ang_mom_error < MAX_ANGULAR_MOMENTUM_ERROR, f"the angular momentum is not conserved ({ang_mom_error:.2e})"
    assert separation_error < MAX_SEPARATION_ERROR, f"the separation is not constant ({separation_error:.2e})"
    assert omega_error < MAX_OMEGA_ERROR, f"the angular velocity does not match the analytic value ({omega_error:.2e})"
    print(f"PASS: circular sink binary, separation {separation_error:.1e}, angular velocity {omega_error:.1e}, "
          f"momentum {momentum:.1e}, angular momentum {ang_mom_error:.1e}")


if __name__ == "__main__":
    main()
