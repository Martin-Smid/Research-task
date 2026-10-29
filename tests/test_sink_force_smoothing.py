"""Test: action = reaction between psi and a sink when the force on the sink is smoothed (SinkNBody smooth_force).

The sink potential acting on psi is a Gaussian of width softening_bh. Without smoothing the sink feels the force of
the environment at a point, so the momentum psi receives from the sink and the momentum the sink receives from psi differ
(by the Gaussian factor in k-space). With smooth_force = True the force the sink feels is convolved with the same Gaussian
and the two are exactly opposite.

Set-up
    A smooth non-uniform density (uniform background plus a Gaussian bump, width of the order of the softening), a sink
    placed exactly on a grid node. Both momentum rates are computed from the same grid forces the code uses:
      rate on psi  = sum over cells of rho * F_sink_potential * dV   (F = -grad of the analytic sink potential),
      rate on sink = M * F_environment at the sink (smoothed or not).

Checks
    1) with smooth_force = True the sum of the two rates is zero to rounding (MAX_BALANCE_ERROR); with False it is not
       (more than MIN_UNSMOOTHED_ERROR), so the test would fail without the feature,
    2) SinkNBody.drift really uses the smoothed force: the acceleration the sink gets in a very short drift equals the smoothed
       (or unsmoothed) force at the node (MAX_DRIFT_ERROR)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # the box is [-BOX, BOX]^3 [kpc]
N_GRID = 64                         # grid resolution (cell 0.3125 kpc)
SOFTENING = 0.6                     # softening_bh [kpc], about 2 cells
M_SINK = 1.0e7                      # sink mass [Msun]
RHO_BACKGROUND = 1.0e6              # background density [Msun/kpc^3]
BUMP_AMPLITUDE = 2.0                # relative height of the density bump
BUMP_WIDTH = 0.6                    # width of the bump [kpc], comparable to the softening so that the smoothing matters
NODE = (3, -2, 1)                   # sink position as an offset in cells from the centre node
MAX_BALANCE_ERROR = 1.0e-9          # |sum of the rates| / |rate on the sink| with smoothing
MIN_UNSMOOTHED_ERROR = 1.0e-2       # the same without smoothing must be larger than this
MAX_DRIFT_ERROR = 1.0e-6            # relative difference of the drift acceleration from the node force


def main():
    sim = Simulation_Class(

        dim=3,
        boundaries=[(-BOX, BOX)] * 3,
        N=N_GRID,
        total_time=1.0e-3,
        h=1.0e-3,
        order_of_evolution=2,
        use_gravity=True,
        use_units=True,
        use_sponge=False,
        self_int=False,

    )
    wave_vector = Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat",    # placeholder, psi is replaced below
        means=[0, 0, 0],
        st_deviations=[0.5, 0.5, 0.5],
        simulation=sim,
        mass=1,
        omega=1,
        momenta=[0, 0.0, 0],
        spin=0,
        desired_soliton_mass=5.0e7,

    )
    sim.add_wave_vector(wave_vector)

    centre = N_GRID // 2
    index = tuple(centre + offset for offset in NODE)
    gx, gy, gz = (cp.asnumpy(g) for g in sim.grids)
    position = [float(gx[index]), float(gy[index]), float(gz[index])]          # exactly on a grid node

    sink = SinkNBody(

        simulation=sim,
        N_sinks=1,
        initial_masses=[M_SINK],
        initial_positions=[position],
        initial_velocities=[[0.0, 0.0, 0.0]],
        softening_bh=SOFTENING,

    )
    sim.add_baryons(sink)
    sim.initialize_simulation()

    # density: uniform background plus a bump away from the sink, psi real
    bump_centre = np.array(position) + np.array([0.9, -0.5, 0.4])
    r2 = (gx - bump_centre[0]) ** 2 + (gy - bump_centre[1]) ** 2 + (gz - bump_centre[2]) ** 2
    density = RHO_BACKGROUND * (1.0 + BUMP_AMPLITUDE * np.exp(-0.5 * r2 / BUMP_WIDTH ** 2))
    sim.wave_functions[0].psi = cp.asarray(np.sqrt(density), dtype=cp.complex128)

    evolution = sim.evolution
    density_grid = evolution._compute_total_density(sim.wave_functions)
    forces_environment = sim.propagator.compute_force_grids_from_potential(sim.propagator.compute_gravity_potential(density_grid))
    forces_sink_potential = sim.propagator.compute_force_grids_from_potential(evolution._compute_sink_potential_analytic_kspace())

    dV = float(sim.dV)
    rate_psi = np.array([float(cp.sum(density_grid * f)) * dV for f in forces_sink_potential])     # momentum rate of psi

    def node_force(grids):
        return np.array([float(g[index]) for g in grids])

    results = {}
    for smooth in (False, True):
        sink.smooth_force = smooth
        grids = sink._smooth_force_grids(forces_environment) if smooth else forces_environment
        rate_sink = M_SINK * node_force(grids)
        balance = np.linalg.norm(rate_psi + rate_sink) / np.linalg.norm(rate_sink)

        # acceleration through the real drift: a very short drift from rest, the sink stays at the node
        sink.positions = cp.asarray([position], dtype=cp.float64)
        sink.velocities = cp.zeros((1, 3), dtype=cp.float64)
        dt = 1.0e-7
        sink.drift(dt, forces_environment)
        acceleration = cp.asnumpy(sink.velocities)[0] / dt
        drift_error = np.linalg.norm(acceleration - node_force(grids)) / np.linalg.norm(node_force(grids))
        results[smooth] = (balance, drift_error)
        print(f"smooth_force = {smooth!s:5}: |rate psi + rate sink| / |rate sink| = {balance:.3e}, "
              f"drift acceleration vs node force: {drift_error:.2e}")

    assert results[True][0] < MAX_BALANCE_ERROR, f"smoothed force: momentum balance error {results[True][0]:.2e}"
    assert results[False][0] > MIN_UNSMOOTHED_ERROR, (
        f"the unsmoothed balance error {results[False][0]:.2e} is too small, the test cannot tell the two apart")
    assert results[True][1] < MAX_DRIFT_ERROR, f"drift does not use the smoothed force (difference {results[True][1]:.2e})"
    assert results[False][1] < MAX_DRIFT_ERROR, f"drift without smoothing differs from the node force by {results[False][1]:.2e}"
    print(f"PASS: smooth_force makes the momentum exchange exact (error {results[True][0]:.1e} instead of {results[False][0]:.1e})")


if __name__ == "__main__":
    main()
