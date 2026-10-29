"""Test: the 1D (DressedSoliton.py) and 3D (DressedSoliton3D.py) ground-state solvers of the soliton with a central black hole.

Checks
    1) 1D, free soliton: U0, omega and Phi(r) agree with resources/solitons/GroundState(1).dat (MAX_FILE_ERROR),
    2) 1D, black hole of large mass: the mass of the Phi(0) = 1 profile equals the analytic hydrogen-like value pi hbar_tilde^2 / (G kappa^3) (MAX_HYDROGEN_ERROR),
    3) 3D (imaginary time, periodic grid) with the black hole at the centre has the same density along the x axis as the 1D profile of the same
       soliton mass, black hole mass and Gaussian width (MAX_PROFILE_ERROR, relative to the central density, where rho > 1e-3 rho_c); the 3D state
       converged (residual below MAX_RESIDUAL),
    4) 3D with the black hole at an offset and pin_centroid: the centroid of psi stays on the pinned point (MAX_CENTROID_SHIFT)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

import DressedSoliton as ds
import DressedSoliton3D as ds3


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
M_EV = 1.0e-21                      # particle mass [eV]
M_SOL = 5.57e7                      # soliton mass [Msun]
M_BH = 4.0e6                        # black hole mass [Msun]
HALF_BOX = 0.75                     # the box is [-HALF_BOX, HALF_BOX]^3 [kpc]
N_GRID = 128                        # cell 11.7 pc
EPS = 0.0117                        # Gaussian width of the black hole = 1 cell [kpc]
MAX_FILE_ERROR = 1.0e-5
MAX_HYDROGEN_ERROR = 1.0e-4
MAX_PROFILE_ERROR = 3.0e-3
MAX_RESIDUAL = 5.0e-4
MAX_CENTROID_SHIFT = 1.0e-6         # [kpc]


def main():
    path = PROJECT_ROOT / "resources" / "solitons" / "GroundState(1).dat"
    ref = np.loadtxt(path)
    with open(path) as f:
        u0_ref, omega_ref = (float(x) for x in f.readline()[1:].split())
    free = ds.solve(0.0)
    inner = free["phi"] > 1e-3
    phi_error = np.max(np.abs(free["phi"][inner] - np.interp(free["r"], ref[:, 0], ref[:, 1])[inner]))
    print(f"1D free soliton: U0 {free['U0']:.6f} / {u0_ref:.6f}, omega {free['omega']:.6f} / {omega_ref:.6f}, max |Phi - file| {phi_error:.1e}")
    assert abs(free["U0"] - u0_ref) < MAX_FILE_ERROR and abs(free["omega"] - omega_ref) < MAX_FILE_ERROR and phi_error < MAX_FILE_ERROR

    ht = ds.hbar_tilde(M_EV)
    kappa = ds.G * 1.0e8 / ht ** 2
    d = ds.solve(kappa, 0.0, r_max=3.0)
    expected = np.pi * ht ** 2 / (ds.G * kappa ** 3)
    error = abs(ds.physical_mass(d, ht) / expected - 1.0)
    print(f"1D hydrogen limit: mass {ds.physical_mass(d, ht):.5e}, analytic {expected:.5e}, error {error:.1e}")
    assert error < MAX_HYDROGEN_ERROR

    sim = ds3.make_sim(N_GRID, HALF_BOX, M_EV)
    centre = int(np.argmin(np.abs(cp.asnumpy(sim.grids[0])[:, 0, 0])))
    assert abs(float(cp.asnumpy(sim.grids[0])[centre, 0, 0])) < 1e-12, "x = 0 is not a grid node"
    res = ds3.relax(sim, M_SOL, M_BH, (0.0, 0.0, 0.0), EPS, verbose=False)
    sol = ds.solve_physical(M_EV, M_BH, M_SOL, EPS)
    rho = cp.asnumpy(res["psi"].real ** 2 + res["psi"].imag ** 2)
    x = np.abs(cp.asnumpy(sim.grids[0])[:, 0, 0])
    line = rho[:, centre, centre]
    rho_1d = ht ** 2 / ds.G * np.interp(x, sol["r"], sol["phi"], right=0.0) ** 2
    rho_c = rho_1d.max()
    selected = rho_1d > 1e-3 * rho_c
    profile_error = np.max(np.abs(line[selected] - rho_1d[selected])) / rho_c
    print(f"3D vs 1D (M_sol {M_SOL:.2e}, M_BH {M_BH:.1e}, eps {EPS}): central density {line.max():.4e} / {rho_c:.4e}, max profile difference "
          f"{profile_error:.1e} of rho_c, residual {res['residual']:.1e}, {res['steps']} steps")
    assert res["residual"] < MAX_RESIDUAL, "the 3D relaxation did not converge"
    assert profile_error < MAX_PROFILE_ERROR, "the 3D ground state differs from the 1D solution"

    pin = (0.0, 0.0, 0.0)
    pinned = ds3.relax(sim, M_SOL, M_BH, (0.1, 0.0, 0.0), EPS, pin_centroid=pin, max_steps=600, verbose=False)
    rho_p = pinned["psi"].real ** 2 + pinned["psi"].imag ** 2
    shift = max(abs(s) for s in ds3._centroid(rho_p, sim, pin))
    print(f"3D pinned (black hole at x = 0.1): centroid shift {shift:.1e} kpc after {pinned['steps']} steps")
    assert shift < MAX_CENTROID_SHIFT
    print("PASS: 1D solver = GroundState file and hydrogen limit, 3D relaxation = 1D profile, pinned centroid")


if __name__ == "__main__":
    main()
