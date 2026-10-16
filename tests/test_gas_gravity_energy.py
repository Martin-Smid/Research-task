"""Regression check: a gravity kick must update gas total energy."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas


def main():
    gas = NBodyGas.__new__(NBodyGas)
    gas.dim = 3
    gas.rho = cp.asarray([2.0, 3.0], dtype=cp.float64)
    gas.vel = [
        cp.asarray([1.0, -2.0], dtype=cp.float64),
        cp.asarray([0.5, 1.5], dtype=cp.float64),
        cp.asarray([0.0, -0.25], dtype=cp.float64),
    ]
    gas.vx, gas.vy, gas.vz = gas.vel

    internal_energy = cp.asarray([7.0, 11.0], dtype=cp.float64)
    kinetic_before = 0.5 * gas.rho * sum(v * v for v in gas.vel)
    gas.E = internal_energy + kinetic_before

    ax = cp.asarray([0.2, -0.1], dtype=cp.float64)
    ay = cp.asarray([0.0, 0.3], dtype=cp.float64)
    az = cp.asarray([-0.4, 0.2], dtype=cp.float64)
    gas._gravity_kick(ax, ay, az, dt=0.25)

    kinetic_after = 0.5 * gas.rho * sum(v * v for v in gas.vel)
    internal_after = gas.E - kinetic_after

    cp.testing.assert_allclose(internal_after, internal_energy, rtol=0.0, atol=1e-12)
    cp.testing.assert_allclose(
        gas.E - (internal_energy + kinetic_before),
        kinetic_after - kinetic_before,
        rtol=0.0,
        atol=1e-12,
    )

    print("PASS: gravity changes gas total energy by exactly the kinetic-energy change")
    print("PASS: gas internal energy is unchanged by the gravity kick")


if __name__ == "__main__":
    main()
