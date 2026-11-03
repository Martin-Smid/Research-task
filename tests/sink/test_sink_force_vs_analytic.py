"""Test B: sink force on the grid against the analytic formula (where is the field trustworthy?).

What is measured
    A sink at the centre of the box creates its potential analytically in k-space (mass smeared by a Gaussian of width
    sigma = softening_bh = 1.5 dx, periodic box, zero mode removed). The grid force a = -grad(phi) is computed from it
    by FFT derivatives and interpolated trilinearly for the particles (NBody._interpolate_force_trilinear).
    Here this force is compared with the exact formula on spheres of different radii and in many directions.

Analytic reference (the source is a Gaussian in a periodic box with a homogeneous neutralising background)
    a_r(r) = G M / r^2 * [erf(x) - 2 x exp(-x^2) / sqrt(pi)]  -  (4 pi / 3) G M r / L^3,    x = r / (sqrt(2) sigma)
    The first term is the force of the smeared mass (for r >> sigma it is Newton's GM/r^2), the second is the effect of
    the periodic box (the homogeneous background that removes k = 0; the images of the source do not change it further
    for a cubic lattice up to the order (r/L)^5).
    It is 6.5 % of Newton's force at r = L/4, so one must not measure against the bare GM/r^2.

What is checked
    For radii between R_MIN_CELLS cells and R_MAX_FRACTION * L the grid force must lie within MAX_ERROR (largest
    relative error over all directions) and its deviation from the radial direction (grid anisotropy) must be below
    MAX_TANGENTIAL. In addition, the error averaged over all directions (systematic shift of the force) must fit into
    MAX_BIAS. The error against bare Newton is printed too, to show how much the box differs.

Measured results (N = 64, 128, 256)
    - The mean error is ~1e-5 to 1e-4: the formula is right and the grid sink force has no systematic shift.
    - The largest error over directions at small r falls with resolution (grid, softening), but from r ~ 4 kpc on it
      does not depend on N and grows as (r/L)^4: this is the cubic anisotropy of the periodic lattice (the force points
      towards the box corners), not a code error. It is 3.5 % at r = 6 kpc and 19 % at 8 kpc, so beyond L/4 from the
      centre the sink field is no longer isotropic."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np
from scipy.special import erf

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # the box is [-BOX, BOX]^3 [kpc], L = 2 BOX
M_SINK = 5.0e10                     # sink mass [Msun]
GRIDS = [64, 128, 256]              # tested resolutions
RADII_KPC = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]    # radii of the spheres on which the force is measured [kpc]
N_DIRECTIONS = 200                  # number of directions on each sphere (uniform, Fibonacci spiral)
R_MIN_CELLS = 6.0                   # smallest checked radius [number of cells]
R_MAX_FRACTION = 0.25               # largest checked radius [fraction of the box length L]
MAX_ERROR = 0.02                    # largest allowed relative error of the radial force (over all directions)
MAX_TANGENTIAL = 0.02               # largest allowed share of the tangential force component (grid anisotropy)
MAX_BIAS = 2.0e-3                   # largest allowed error averaged over directions (systematic shift of the force)
H_UNUSED = 1.0e-4                   # not used; below the stability limit of every grid (N = 256: 1.65e-4), so no warning


def build_system(n_grid):
    """Sink at the centre + force grids of its field. Returns (sim, sink, (Fx, Fy, Fz))."""

    sim = Simulation_Class(

        dim=3,                              # 3D simulation
        boundaries=[(-BOX, BOX)] * 3,       # box, the centre is at (0, 0, 0)
        N=n_grid,                           # grid resolution
        total_time=0.01,                    # not used, evolve() is not called
        h=H_UNUSED,                         # not used
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
    return sim, sink, forces


def sphere_directions(n_points):
    """Uniformly distributed unit directions (Fibonacci spiral). Shape (n_points, 3)."""

    index = np.arange(n_points) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / n_points)
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * index
    return np.column_stack((np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)))


def analytic_inward_force(radius, sigma, g_const, box_length):
    """Inward force magnitude: smeared source (Gaussian) minus the effect of the homogeneous background of the periodic box."""

    x = radius / (np.sqrt(2.0) * sigma)
    gauss = g_const * M_SINK / radius ** 2 * (erf(x) - 2.0 * x * np.exp(-x ** 2) / np.sqrt(np.pi))
    background = 4.0 * np.pi / 3.0 * g_const * M_SINK * radius / box_length ** 3
    return gauss - background


def measure_radius(sim, forces, radius, directions):
    """Grid force on a sphere of the given radius. Returns (inward radial component, tangential share) for each direction."""

    particle = Baryons(

        simulation=sim,                 # simulation (grid, boundaries)
        N_particles=len(directions),    # one test particle per direction
        total_mass=1.0,                 # the mass does not matter, we only read the force
        init_profile="uniform",         # positions are overwritten right away

    )
    particle.positions = cp.asarray(radius * directions, dtype=cp.float64)
    force = cp.asnumpy(particle._interpolate_force_trilinear(*forces))
    inward = -np.einsum("ij,ij->i", force, directions)                       # -a . r_hat
    tangential = np.linalg.norm(force + inward[:, None] * directions, axis=1) / np.linalg.norm(force, axis=1)
    return inward, tangential


def main():
    directions = sphere_directions(N_DIRECTIONS)
    box_length = 2.0 * BOX
    worst_error, worst_tangential, worst_bias = 0.0, 0.0, 0.0

    for n_grid in GRIDS:
        sim, sink, forces = build_system(n_grid)
        dx = float(min(sim.dx))
        sigma = float(sink.softening_bh)
        g_const = float(sim.G)
        print(f"\nN = {n_grid}: dx = {dx:.4f} kpc, sigma (softening) = {sigma:.4f} kpc, L = {box_length} kpc")
        print(f"{'r [kpc]':>8} {'r/dx':>6} {'mean error':>12} {'max |error|':>12} {'tangential':>13}"
              f" {'vs Newton':>16}  checked")
        for radius in RADII_KPC:
            inward, tangential = measure_radius(sim, forces, radius, directions)
            expected = analytic_inward_force(radius, sigma, g_const, box_length)
            error = inward / expected - 1.0
            vs_newton = float(np.mean(inward) / (g_const * M_SINK / radius ** 2) - 1.0)
            checked = R_MIN_CELLS * dx <= radius <= R_MAX_FRACTION * box_length
            print(f"{radius:>8.2f} {radius / dx:>6.1f} {np.mean(error):>+12.2e} {np.abs(error).max():>12.2e}"
                  f" {tangential.max():>13.2e} {vs_newton:>+16.2e}  {'yes' if checked else '-'}")
            if checked:
                worst_error = max(worst_error, float(np.abs(error).max()))
                worst_tangential = max(worst_tangential, float(tangential.max()))
                worst_bias = max(worst_bias, abs(float(np.mean(error))))

    assert worst_bias < MAX_BIAS, f"mean force error {worst_bias:.2e} is above the limit {MAX_BIAS} (systematic shift)"
    assert worst_error < MAX_ERROR, f"grid force differs from the analytic one by up to {worst_error:.2e} (limit {MAX_ERROR})"
    assert worst_tangential < MAX_TANGENTIAL, (
        f"tangential force component is up to {worst_tangential:.2e} of its magnitude (limit {MAX_TANGENTIAL})")
    print(f"\nPASS: sink force on the grid matches the analytic one (mean error {worst_bias:.1e}, "
          f"largest error {worst_error:.1e}, largest tangential component {worst_tangential:.1e})")


if __name__ == "__main__":
    main()
