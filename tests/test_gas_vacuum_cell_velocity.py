"""Regression check: a cell emptied by the hydro update must not keep its momentum.

When the update drives a density below rho_floor the density is clamped, but the momentum used to stay,
so v = m / rho_floor exploded (1e12 and more) and later overflowed to NaN."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas


def main():
    shape = (4, 4, 4)
    gas = NBodyGas.__new__(NBodyGas)
    gas.dim = 3
    gas.dx_list = [1.0, 1.0, 1.0]
    gas.rho_floor = 1e-12
    gas.e_floor = 1e-10
    gas.rho = cp.ones(shape, dtype=cp.float64)
    gas.vel = [cp.zeros(shape, dtype=cp.float64) for _ in range(gas.dim)]
    gas.vx, gas.vy, gas.vz = gas.vel
    gas.E = cp.ones(shape, dtype=cp.float64)

    def emptying_sweep(rho, moms, energy, axis, dt, dx):
        if axis == 0:  # the update empties one cell but leaves momentum and energy in it
            rho = rho.copy()
            moms = [m.copy() for m in moms]
            energy = energy.copy()
            rho[1, 1, 1] = -1e-3
            moms[0][1, 1, 1] = 1.0
            energy[1, 1, 1] = 5.0
        return rho, moms, energy

    gas._sweep_axis = emptying_sweep
    gas._hydro_step_nd(0.1)

    speed = cp.sqrt(sum(v * v for v in gas.vel))
    print(f"emptied cell: rho = {float(gas.rho[1, 1, 1]):.1e}, |v| = {float(speed[1, 1, 1]):.3e}, "
          f"E = {float(gas.E[1, 1, 1]):.3e}")
    assert bool(cp.isfinite(gas.E).all()) and bool(cp.isfinite(speed).all())
    assert float(speed.max()) < 1e3, f"velocity in the emptied cell exploded: {float(speed.max()):.3e}"
    assert float(gas.rho[1, 1, 1]) == gas.rho_floor
    print("PASS: an emptied cell does not keep momentum (no v = m/rho_floor blow-up)")


if __name__ == "__main__":
    main()
