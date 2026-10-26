"""Regression check: a vacuum cell between two diverging flows must not create mass.

Density has a local minimum in the vacuum cell (slope 0) while the momentum is monotonic (slope != 0), so
independent limiting gave a face with finite momentum and rho = rho_floor: face velocity ~1e11, Rusanov fluxes
of 1e10, negative densities that the clamp turned into created mass (main.py disk with the cos^2 window)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas


def main():
    n, dx, dt = 16, 0.156, 1e-3
    gas = NBodyGas.__new__(NBodyGas)
    gas.dim = 1
    gas.rho_floor, gas.e_floor, gas.gamma = 1e-12, 1e-10, 5.0 / 3.0

    rho = cp.full(n, 1.5e3)
    m = cp.zeros(n)
    rho[[13, 14, 15, 0, 1, 2, 3]] = cp.asarray([1.5e3, 6.1e2, 1.4e2, 1e-12, 1.4e2, 6.1e2, 1.5e3])
    m[[13, 14, 15, 1, 2, 3]] = cp.asarray([-5.4, -1.5, -0.17, 0.17, 1.5, 5.4])  # flow away from cell 0
    E = rho * 94.0 + 0.5 * m * m / cp.maximum(rho, 1e-12)  # cs = 10 kpc/Gyr

    rho1, moms1, E1 = gas._sweep_axis(rho, [m], E, 0, dt, dx)
    mass0, mass1 = float(rho.sum()), float(rho1.sum())
    print(f"rho around the hole after one sweep: {[f'{float(r):.2e}' for r in rho1[[15, 0, 1]]]}")
    assert float(rho1.min()) > -1e-6 * float(rho.max()), f"negative density: {float(rho1.min()):.3e}"
    assert float(rho1.max()) <= 1.01 * float(rho.max()), f"density spike: {float(rho1.max()):.3e}"
    assert abs(mass1 - mass0) <= 1e-12 * mass0, "sweep does not conserve mass"
    print("PASS: a vacuum hole between diverging flows creates no negative density or mass")


if __name__ == "__main__":
    main()
