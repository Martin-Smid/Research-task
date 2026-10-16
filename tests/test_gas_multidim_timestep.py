"""Regression check: every directional hydro sweep must advance the full dt."""

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
    gas.dx_list = [1.0, 1.5, 2.0]
    gas.rho_floor = 1e-12
    gas.e_floor = 1e-10
    gas.rho = cp.ones(shape, dtype=cp.float64)
    gas.vel = [cp.zeros(shape, dtype=cp.float64) for _ in range(gas.dim)]
    gas.vx, gas.vy, gas.vz = gas.vel
    gas.E = cp.ones(shape, dtype=cp.float64)

    calls = []

    def record_sweep(rho, moms, energy, axis, dt, dx):
        calls.append((axis, dt, dx))
        return rho, moms, energy

    gas._sweep_axis = record_sweep
    requested_dt = 0.3
    gas._hydro_step_nd(requested_dt)

    assert len(calls) == gas.dim, f"Expected {gas.dim} sweeps, got {len(calls)}"
    assert [axis for axis, _, _ in calls] == [0, 1, 2]
    assert all(dt == requested_dt for _, dt, _ in calls), (
        f"Every sweep must receive dt={requested_dt}; calls were {calls}"
    )

    print("PASS: 3D hydro performed one sweep per axis")
    print(f"PASS: all directional sweeps received the full dt={requested_dt}")


if __name__ == "__main__":
    main()
