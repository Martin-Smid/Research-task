"""Test: sink-sink gravity of Gaussian-extended sinks (SinkNBody smooth_force) uses the effective width s^2 = 2 eps^2.

A sink with a Gaussian of width eps both acts (source) and feels (receiver) through that Gaussian, so two such sinks
interact like two Gaussians: the potential of one at the position of the other is -G M erf(r / (sqrt(2) s)) / r with
s^2 = eps^2 + eps^2 = 2 eps^2. With smooth_force = False the receiver is a point and s = eps.

Checks (the references are analytic, independent of the code)
    1) _pair_accelerations equals the analytic acceleration with s = sqrt(2) eps for smooth_force = True and with s = eps
       for False (MAX_FORCE_ERROR); the two differ visibly (MIN_DIFFERENCE), so the test fails if the width is not changed,
    2) the energy Diagnostics uses for a smoothed sink (deposit_to_grid_smoothed against the analytic sink potential)
       changes with the separation like the analytic Gaussian pair energy with s^2 = 2 eps^2 (MAX_ENERGY_ERROR),
    3) the acceleration is minus the gradient of that energy (energy and force use the same kernel, MAX_GRADIENT_ERROR)."""

from pathlib import Path
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
N_GRID = 64                         # cell 0.3125 kpc
M_SINK = 5.0e10                     # mass of each sink [Msun]
SOFTENING = 0.5                     # eps [kpc], 1.6 cells
SEPARATION = 1.25                   # r [kpc] = 4 cells (sinks sit on grid nodes)
DELTA_CELLS = 1                     # separation change for the energy difference [cells]
MAX_FORCE_ERROR = 1.0e-10           # relative error of the pair acceleration
MIN_DIFFERENCE = 1.0e-2             # smoothed and point-receiver accelerations must differ by more than this
MAX_ENERGY_ERROR = 2.0e-3           # relative error of the energy difference (CIC on nodes is exact, Gaussians are resolved)
MAX_GRADIENT_ERROR = 2.0e-2         # acceleration vs finite-difference gradient of the energy over one cell


def analytic_potential(G, r, s):
    return -G * M_SINK * erf(r / (np.sqrt(2.0) * s)) / r


def analytic_acceleration(G, r, s, volume):
    x = r / s
    enclosed = erf(x / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * x * x)
    return G * M_SINK * enclosed / r ** 2 - 4.0 * np.pi / 3.0 * G * M_SINK * r / volume


def make_sink(sim, smooth, separation):
    dx = 2.0 * BOX / N_GRID
    return SinkNBody(

        simulation=sim,
        N_sinks=2,
        initial_masses=[M_SINK, M_SINK],
        initial_positions=[[0.0, 0.0, 0.0], [separation, 0.0, 0.0]],
        initial_velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        softening_bh=SOFTENING,
        smooth_force=smooth,

    )


def grid_pair_energy(sim, sink):
    """1/2 integral rho_s phi_s dV the way Diagnostics does it (rho_s smoothed for an extended sink)."""
    kx, ky, kz = sim.k_space
    k2 = kx * kx + ky * ky + kz * kz
    k2_safe = cp.where(k2 > 0.0, k2, 1.0)
    rho_s = sink.deposit_to_grid_smoothed()
    rho_point = sink.deposit_to_grid()
    phi_k = -4.0 * np.pi * float(sim.G) * cp.fft.fftn(rho_point) * cp.exp(-0.5 * k2 * SOFTENING ** 2) / k2_safe
    phi_k[k2 == 0.0] = 0.0
    phi_s = cp.fft.ifftn(phi_k).real
    return 0.5 * float(cp.sum(rho_s * phi_s)) * float(sim.dV)


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
    G = float(sim.G)
    volume = (2.0 * BOX) ** 3
    dx = 2.0 * BOX / N_GRID

    # 1) pair acceleration of sink 0 (it points to +x towards sink 1)
    acc = {}
    for smooth in (False, True):
        a = float(cp.asnumpy(make_sink(sim, smooth, SEPARATION)._pair_accelerations())[0, 0])
        s = SOFTENING * (np.sqrt(2.0) if smooth else 1.0)
        expected = analytic_acceleration(G, SEPARATION, s, volume)
        error = abs(a / expected - 1.0)
        acc[smooth] = a
        print(f"smooth_force = {smooth}: acceleration {a:.6e}, analytic (s = {s:.4f}) {expected:.6e}, error {error:.1e}")
        assert error < MAX_FORCE_ERROR, f"the pair acceleration does not match the analytic value for smooth_force={smooth}"
    difference = abs(acc[True] / acc[False] - 1.0)
    print(f"relative difference between the two kernels: {difference:.3f}")
    assert difference > MIN_DIFFERENCE, "the smoothed pair force is not different from the point-receiver force"

    # 2) energy difference between two separations: grid (Diagnostics model) against the analytic Gaussian pair energy
    r1, r2 = SEPARATION, SEPARATION + DELTA_CELLS * dx
    s = np.sqrt(2.0) * SOFTENING
    dU_grid = grid_pair_energy(sim, make_sink(sim, True, r2)) - grid_pair_energy(sim, make_sink(sim, True, r1))
    dU_exact = M_SINK * (analytic_potential(G, r2, s) - analytic_potential(G, r1, s))
    # periodic Poisson solver (zero mean density): the pair energy has the extra term -2 pi G M^2 r^2 / (3 V),
    # whose derivative is the mean-density force -(4 pi / 3) G M r / V of the pair acceleration
    dU_exact -= 2.0 * np.pi * G * M_SINK ** 2 * (r2 ** 2 - r1 ** 2) / (3.0 * volume)
    energy_error = abs(dU_grid / dU_exact - 1.0)
    print(f"energy change over {DELTA_CELLS} cell: grid {dU_grid:.6e}, analytic Gaussian(s^2 = 2 eps^2) {dU_exact:.6e}, error {energy_error:.1e}")
    assert energy_error < MAX_ENERGY_ERROR, "the logged sink-sink energy does not use the effective width sqrt(2) eps"

    # 2b) the analytic energy Diagnostics uses (potential_energy_analytic) gives the same difference to rounding
    dU_analytic = (make_sink(sim, True, r2).potential_energy_analytic() - make_sink(sim, True, r1).potential_energy_analytic())
    analytic_error = abs(dU_analytic / dU_exact - 1.0)
    print(f"analytic sink energy change: {dU_analytic:.6e}, error {analytic_error:.1e}")
    assert analytic_error < MAX_FORCE_ERROR, "potential_energy_analytic does not match the analytic Gaussian pair energy"

    # 3) force = -dU/dr (mid-point acceleration against the finite difference of the energy), per sink pair
    a_mid = float(cp.asnumpy(make_sink(sim, True, 0.5 * (r1 + r2))._pair_accelerations())[0, 0])
    a_fd = dU_grid / (r2 - r1) / M_SINK         # sink 0 is pulled towards +x, i.e. towards a smaller separation: a = +dU/dr / M
    gradient_error = abs(a_mid / a_fd - 1.0)
    print(f"acceleration {a_mid:.6e} vs -dU/dr / M {a_fd:.6e}, error {gradient_error:.1e}")
    assert gradient_error < MAX_GRADIENT_ERROR, "the pair force is not the gradient of the logged energy"
    print("PASS: the pair force and the energy of smoothed sinks use s^2 = 2 eps^2")


if __name__ == "__main__":
    main()
