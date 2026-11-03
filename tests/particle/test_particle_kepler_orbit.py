"""Test A: time accuracy of the N-body particle integrator (NBody.drift) on a circular orbit around a sink.

Setup
    One test particle (negligible mass) orbits a sink at the centre of the box. The sink field is static, so only
    the TIME INTEGRATION ERROR is measured: the orbit from NBody.drift is compared with a nearly exact orbit (RK4 with
    a step 64x smaller than the smallest tested h) in the SAME interpolated field. The error of the field itself
    (periodic box, resolution, sink softening) does not enter here; it belongs in a separate test.

Three checks
    1) NBody.drift with one substep per window is exactly the textbook DKD (leapfrog).
    2) Plain DKD converges with the step h at a mean order of at least MIN_MEAN_ORDER (theory: 2). The order between
       two neighbouring h fluctuates (1.7-2.6) because the interpolated force is only piecewise linear (C0).
    3) The production setup (NBody.drift chooses the number of substeps itself) keeps the position error below
       MAX_ERROR_PRODUCTION * r0. Note: the production error hardly depends on h, because the number of substeps is
       set by the criterion 0.25 * dx / v, not by the window length.

The sink does not appear in the Poisson equation as a density, its potential is analytic (k-space), so the particle
feels exactly the same field as in a real simulation (forces_total = environment + sink)."""

from contextlib import contextmanager
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np
from tqdm import tqdm

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # the box is [-BOX, BOX]^3 [kpc]
N_GRID = 128                        # grid resolution (cell 0.156 kpc)
M_SINK = 5.0e10                     # mass of the central sink [Msun]
M_PARTICLE = 1.0                    # mass of the test particle [Msun] (negligible, so it does not pull the source)
R0 = 5.0                            # starting radius [kpc] (far from the box walls and from the sink softening)
N_ORBITS = 3                        # number of integrated orbits
STEPS_PER_ORBIT = [20, 40, 80, 160, 320]    # steps per orbit for the tested h (each is half of the previous one)
REF_STEPS_PER_ORBIT = 2560                  # steps of the reference RK4 (divisible by all of the above)
MIN_MEAN_ORDER = 1.7                # smallest allowed mean order of plain DKD
MAX_ERROR_PRODUCTION = 2.0e-3       # largest allowed position error in production [multiples of r0]
MAX_DIFF_TEXTBOOK = 1.0e-9          # largest allowed difference of drift() from textbook DKD [kpc]


def build_system():
    """Sink at the centre + force grids of its field. Returns (sim, (Fx, Fy, Fz))."""

    sim = Simulation_Class(

        dim=3,                              # 3D simulation
        boundaries=[(-BOX, BOX)] * 3,       # box, the centre is at (0, 0, 0)
        N=N_GRID,                           # grid resolution
        total_time=0.01,                    # not used, evolve() is not called
        h=0.0006,                           # not used (below the stability limit of N = 128, so no warning)
        order_of_evolution=2,               # order of evolution
        use_gravity=True,                   # gravity on
        use_units=True,                     # kpc, Msun, Gyr
        use_sponge=False,                   # no sponge
        self_int=False,                     # no self-interaction

    )

    sink = SinkNBody(

        simulation=sim,                             # simulation the sink belongs to
        N_sinks=1,                                  # one sink
        initial_masses=[M_SINK],                    # its mass [Msun]
        initial_positions=[[0.0, 0.0, 0.0]],        # at the centre of the box
        initial_velocities=[[0.0, 0.0, 0.0]],       # at rest

    )
    sim.add_baryons(sink)
    sim.initialize_simulation()

    phi_sink = sim.evolution._compute_sink_potential_analytic_kspace()      # analytic sink potential
    forces = sim.propagator.compute_force_grids_from_potential(phi_sink)    # a = -grad(phi) on the grid
    return sim, forces


def build_particle(sim, position, velocity):
    """One test particle at the given position and with the given velocity."""

    particle = Baryons(

        simulation=sim,                 # simulation (grid, boundaries)
        N_particles=1,                  # one particle
        total_mass=M_PARTICLE,          # negligible mass
        init_profile="uniform",         # the profile does not matter, position and velocity are overwritten right away

    )
    particle.positions = cp.asarray([position], dtype=cp.float64)
    particle.velocities = cp.asarray([velocity], dtype=cp.float64)
    particle.invalidate_density_cache()
    return particle


def acceleration(particle, forces, position):
    """Acceleration at `position` by the same trilinear interpolation that NBody.drift uses."""

    particle.positions = cp.asarray([position], dtype=cp.float64)
    return cp.asnumpy(particle._interpolate_force_trilinear(*forces))[0]


def circular_velocity(particle, forces, radius):
    """Circular velocity from the force the particle ACTUALLY feels on the grid (not from the analytic GM/r)."""

    a = acceleration(particle, forces, [radius, 0.0, 0.0])
    return float(np.sqrt(radius * abs(a[0])))


@contextmanager
def single_substep(sim):
    """NBody.drift splits the window into substeps by the criterion 0.25 * min(dx) / v. With a huge dx there is always
    exactly one substep, so drift() performs exactly one DKD step of length h (the only way to measure the order in h)."""

    real_dx = sim.dx
    sim.dx = [1.0e3] * 3
    try:
        yield
    finally:
        sim.dx = real_dx


def reference_orbit(sim, forces, v0, dt, n_steps):
    """Nearly exact orbit: classical RK4 in the same interpolated field. Returns positions (n_steps + 1, 3)."""

    probe = build_particle(sim, [R0, 0.0, 0.0], [0.0, v0, 0.0])

    def derivative(state):
        return np.concatenate([state[3:], acceleration(probe, forces, state[:3])])

    state = np.array([R0, 0.0, 0.0, 0.0, v0, 0.0])
    positions = [state[:3].copy()]
    for _ in tqdm(range(n_steps), desc="reference RK4 orbit"):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        positions.append(state[:3].copy())
    return np.array(positions)


def textbook_dkd_orbit(sim, forces, v0, h, n_steps):
    """Textbook DKD (half drift, full kick, half drift) on the CPU. Returns positions (n_steps + 1, 3)."""

    probe = build_particle(sim, [R0, 0.0, 0.0], [0.0, 0.0, 0.0])
    x = np.array([R0, 0.0, 0.0])
    v = np.array([0.0, v0, 0.0])
    positions = [x.copy()]
    for _ in range(n_steps):
        x = x + v * h / 2.0
        v = v + acceleration(probe, forces, x) * h
        x = x + v * h / 2.0
        positions.append(x.copy())
    return np.array(positions)


def code_orbit(sim, forces, v0, h, n_steps):
    """Orbit from NBody.drift: n_steps windows of length h. Returns positions (n_steps + 1, 3)."""

    particle = build_particle(sim, [R0, 0.0, 0.0], [0.0, v0, 0.0])
    positions = [cp.asnumpy(particle.positions)[0].copy()]
    for step in range(n_steps):
        particle.drift(dt=h, potential_grid=forces, first_step=(step == 0), last_step=(step == n_steps - 1))
        positions.append(cp.asnumpy(particle.positions)[0].copy())
    return np.array(positions)


def max_error(orbit, reference, steps_per_orbit):
    """Largest distance from the reference (in multiples of r0); the reference is sampled at the same times as the orbit."""

    stride = REF_STEPS_PER_ORBIT // steps_per_orbit
    return float(np.linalg.norm(orbit - reference[::stride][: len(orbit)], axis=1).max() / R0)


def main():
    sim, forces = build_system()
    probe = build_particle(sim, [R0, 0.0, 0.0], [0.0, 0.0, 0.0])

    v0 = circular_velocity(probe, forces, R0)
    period = 2.0 * np.pi * R0 / v0
    print(f"sink {M_SINK:.1e} Msun, r0 = {R0} kpc, N = {N_GRID}, v_circ(grid) = {v0:.2f} kpc/Gyr, "
          f"period = {period:.5f} Gyr, {N_ORBITS} orbits")

    reference = reference_orbit(sim, forces, v0, period / REF_STEPS_PER_ORBIT, N_ORBITS * REF_STEPS_PER_ORBIT)
    radius = np.linalg.norm(reference, axis=1)
    print(f"reference orbit: r in range [{radius.min():.4f}, {radius.max():.4f}] kpc "
          f"(the grid field is not exactly central, so the orbit is not an exact circle)")

    # --- 1) + 2) plain DKD: one substep per window
    print("\nPlain DKD (substep criterion switched off):")
    print(f"{'steps/orbit':>11} {'h [Gyr]':>10} {'max position error / r0':>24} {'order':>6}")
    errors = []
    with single_substep(sim):
        for steps_per_orbit in STEPS_PER_ORBIT:
            h = period / steps_per_orbit
            n_steps = N_ORBITS * steps_per_orbit
            orbit = code_orbit(sim, forces, v0, h, n_steps)
            errors.append(max_error(orbit, reference, steps_per_orbit))
            order = np.log2(errors[-2] / errors[-1]) if len(errors) > 1 else float("nan")
            print(f"{steps_per_orbit:>11} {h:>10.6f} {errors[-1]:>24.3e} {order:>6.2f}")

        check_steps = 80
        textbook = textbook_dkd_orbit(sim, forces, v0, period / check_steps, N_ORBITS * check_steps)
        from_code = code_orbit(sim, forces, v0, period / check_steps, N_ORBITS * check_steps)
    diff = float(np.abs(textbook - from_code).max())
    mean_order = float(np.log2(errors[0] / errors[-1]) / (len(errors) - 1))
    print(f"mean order over {len(errors) - 1} step halvings: {mean_order:.2f}")
    print(f"largest difference drift() (1 substep) vs textbook DKD: {diff:.2e} kpc")

    # --- 3) production setup
    print("\nProduction NBody.drift (number of substeps chosen by the criterion):")
    print(f"{'steps/orbit':>11} {'h [Gyr]':>10} {'max position error / r0':>24}")
    production = []
    for steps_per_orbit in (40, 80, 160):
        h = period / steps_per_orbit
        orbit = code_orbit(sim, forces, v0, h, N_ORBITS * steps_per_orbit)
        production.append(max_error(orbit, reference, steps_per_orbit))
        print(f"{steps_per_orbit:>11} {h:>10.6f} {production[-1]:>24.3e}")

    assert diff < MAX_DIFF_TEXTBOOK, f"drift() with one substep is not the textbook DKD (difference {diff:.2e} kpc)"
    assert mean_order >= MIN_MEAN_ORDER, f"mean order of plain DKD is {mean_order:.2f}, expected >= {MIN_MEAN_ORDER}"
    assert max(production) < MAX_ERROR_PRODUCTION, (
        f"production position error {max(production):.2e} r0 is above the limit {MAX_ERROR_PRODUCTION:.1e}")
    print(f"\nPASS: drift() is the textbook DKD (difference {diff:.1e} kpc), mean order {mean_order:.2f}, "
          f"production position error {max(production):.1e} r0")


if __name__ == "__main__":
    main()
