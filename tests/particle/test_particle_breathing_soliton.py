"""Test C: test particle in a time-varying source (breathing soliton), self-convergence in the step h.

Why
    Within one step the particles move in the field computed from the density at the START of the step (frozen
    potential, see Evolution._evolve_order_2). When the source changes, this lag gives a first-order error in h.
    There is no exact solution, so the orbit at h is compared with the orbit at the finest h (Richardson).

Setup
    A soliton (5.3e7 Msun) with the wave function multiplied by SCALE: it is not in equilibrium and breathes. Within
    T_END = 0.2 Gyr the rho_max grows about 3x (the first, compressing phase of the breathing), so the source changes
    significantly. A test particle of negligible mass orbits inside the core (r0 = 1 kpc), where the enclosed mass
    changes. Outside a spherical soliton the field would not change (shell theorem), hence inside.

Checks
    1) The orbit error falls with h at a mean order of at least MIN_MEAN_ORDER. Against the finest h a true order
       p = 1 gives an apparent order of 1.2 and 1.6, p = 2 would give 2.1 and 2.3 (the finest orbit is not exact).
    2) The position error at h = 4e-4 is below MAX_ERROR (multiples of r0).
    Measured (N = 64, 0.2 Gyr): apparent order 1.2 and 1.6, error 1.3e-3 r0 at h = 4e-4. The source changes about 3x;
    for a real galaxy (+-13 % density) we expect an error an order of magnitude smaller (not measured)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from test_particle_kepler_orbit import circular_velocity


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0 / 2.0                    # the box is [-BOX, BOX]^3 [kpc]
N_GRID = 64                         # grid resolution (cell 0.156 kpc)
M_SOLITON = 53090072.0              # soliton mass [Msun]
SCALE = 1.15                        # factor on the wave function: the soliton is not in equilibrium and breathes
R0 = 1.0                            # starting radius of the particle [kpc] (inside the core)
M_PARTICLE = 1.0                    # mass of the test particle [Msun] (negligible)
T_END = 0.2                         # integration length [Gyr]
H_LIST = [4.0e-4, 2.0e-4, 1.0e-4]   # tested h (the stability limit of h for this grid is 6.6e-4)
H_REFERENCE = 5.0e-5                # finest h, against which the error is measured
MIN_MEAN_ORDER = 0.9                # smallest allowed mean apparent order (frozen potential: ~1.4)
MAX_ERROR = 5.0e-3                  # largest allowed position error at h = H_LIST[0] [multiples of r0]


def run_orbit(h):
    """Runs the soliton + test particle with step h and returns the particle positions after every step (n_steps + 1, 3)."""

    sim = Simulation_Class(

        dim=3,                              # 3D simulation
        boundaries=[(-BOX, BOX)] * 3,       # box, the centre is at (0, 0, 0)
        N=N_GRID,                           # grid resolution
        total_time=T_END,                   # how long to integrate [Gyr]
        h=h,                                # time step [Gyr]
        order_of_evolution=2,               # order of evolution
        use_gravity=True,                   # gravity on
        use_units=True,                     # kpc, Msun, Gyr
        use_sponge=False,                   # no sponge
        self_int=False,                     # no self-interaction
        save_max_vals=False,                # save nothing
        m_s=2.5e-22,                        # ULDM particle mass [eV]

    )

    wave_vector = Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat",    # ground-state soliton profile
        means=[0, 0, 0],                                        # centre
        st_deviations=[0.5, 0.5, 0.5],                          # not used for a profile from a file
        simulation=sim,                                         # simulation
        mass=1,                                                 # mass
        omega=1,                                                # frequency
        momenta=[0, 0.0, 0],                                    # no momentum
        spin=0,                                                 # scalar field
        desired_soliton_mass=M_SOLITON,                         # soliton mass [Msun]

    )
    sim.add_wave_vector(wave_vector)

    tracer = Baryons(

        simulation=sim,                 # simulation (grid, boundaries)
        N_particles=1,                  # one test particle
        total_mass=M_PARTICLE,          # negligible mass
        init_profile="uniform",         # the profile does not matter, position and velocity are overwritten right away

    )
    sim.add_baryons(tracer)
    sim.initialize_simulation()

    evolution = sim.evolution
    wave_function = sim.wave_functions[0]
    wave_function.psi = cp.asarray(wave_function.psi) * SCALE   # out-of-equilibrium soliton: it breathes

    # circular velocity from the force the particle actually feels on the grid at the start
    density = evolution._compute_total_density(sim.wave_functions)
    phi_environment, phi_sink = evolution._compute_gravity_fields(density)
    forces = sim.propagator.compute_force_grids_from_potential(phi_environment + phi_sink)
    v0 = circular_velocity(tracer, forces, R0)
    tracer.positions = cp.asarray([[R0, 0.0, 0.0]])
    tracer.velocities = cp.asarray([[0.0, v0, 0.0]])
    tracer.invalidate_density_cache()

    # particle positions after every step (wrap of the evolution step, the same trick as in main.py)
    trajectory = [cp.asnumpy(tracer.positions)[0].copy()]
    original_step = evolution._perform_evolution_step

    def recorded_step(wave_functions, total_density, step, save_step):
        result = original_step(wave_functions, total_density, step, save_step)
        trajectory.append(cp.asnumpy(tracer.positions)[0].copy())
        return result

    evolution._perform_evolution_step = recorded_step
    evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)
    return np.array(trajectory)


def main():
    # this test measures the frozen force; the default (time-centred force) is tested in test_particle_time_centered_force.py
    previous = Simulation_Class.baryon_time_centered_force
    Simulation_Class.baryon_time_centered_force = False
    try:
        run_test()
    finally:
        Simulation_Class.baryon_time_centered_force = previous


def run_test():
    reference = run_orbit(H_REFERENCE)
    print(f"soliton {M_SOLITON:.2e} Msun (psi x {SCALE}), r0 = {R0} kpc, {T_END} Gyr, "
          f"reference h = {H_REFERENCE:.1e} ({len(reference) - 1} steps)")
    print(f"\n{'h [Gyr]':>10} {'steps':>7} {'max position error / r0':>24} {'apparent order':>14}")

    errors = []
    for h in H_LIST:
        orbit = run_orbit(h)
        stride = int(round(h / H_REFERENCE))
        difference = np.linalg.norm(orbit - reference[::stride][: len(orbit)], axis=1)
        errors.append(float(difference.max() / R0))
        order = np.log2(errors[-2] / errors[-1]) if len(errors) > 1 else float("nan")
        print(f"{h:>10.1e} {len(orbit) - 1:>7} {errors[-1]:>24.3e} {order:>14.2f}")

    mean_order = float(np.log2(errors[0] / errors[-1]) / (len(errors) - 1))
    print(f"mean apparent order: {mean_order:.2f} (for a true order 1 it comes out ~1.4, for order 2 ~2.2)")

    assert mean_order >= MIN_MEAN_ORDER, f"particle error does not fall with h: apparent order {mean_order:.2f}"
    assert errors[0] < MAX_ERROR, f"position error {errors[0]:.2e} r0 at h = {H_LIST[0]:.0e} is above the limit {MAX_ERROR}"
    print(f"PASS: particle in the breathing soliton converges (apparent order {mean_order:.2f}), "
          f"error at h = {H_LIST[0]:.0e} is {errors[0]:.1e} r0")


if __name__ == "__main__":
    main()
