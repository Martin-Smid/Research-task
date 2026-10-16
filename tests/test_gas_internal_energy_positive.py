"""Regression check: the gravity energy correction must never push the gas internal energy
E - 1/2 rho v^2 below its floor, even for cold gas falling in a steep potential."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import cupy as cp

from test_gas_gravity_work_conservation import KM_S_TO_KPC_GYR, build


def main():
    sim, gas = build()
    prop = sim.propagator
    rho0 = gas.rho.copy()
    phi = prop.solve_poisson(rho0)
    forces = prop.compute_force_grids_from_potential(phi)
    gas.cs = 0.1 * KM_S_TO_KPC_GYR  # cold gas: tiny internal energy, so any gravity-work mismatch shows
    gas.vel = [1e-2 * a for a in forces]
    gas.vx, gas.vy, gas.vz = gas.vel
    gas.E = gas.rho * gas.cs ** 2 / (gas.gamma * (gas.gamma - 1.0)) + 0.5 * gas.rho * sum(v * v for v in gas.vel)
    gas.phi_total = phi

    def potential_update(d_rho):
        new_phi = prop.solve_poisson(rho0 + d_rho)
        return prop.compute_force_grids_from_potential(new_phi), new_phi

    gas.potential_update = potential_update
    gas.drift(2e-3, forces)

    eint = gas.E - 0.5 * gas.rho * sum(v * v for v in gas.vel)
    margin = eint - gas.rho * gas.e_floor
    worst = float(cp.min(margin / cp.maximum(gas.rho, 1e-300)))
    negative = int(cp.sum(margin < -1e-12 * cp.abs(gas.E)))  # round-off of E - K scales with E
    print(f"cells below the internal-energy floor: {negative}; worst (E-K-floor)/rho = {worst:.3e}")
    assert negative == 0, f"{negative} cells have internal energy below the floor"
    print("PASS: gas internal energy stays above its floor")


if __name__ == "__main__":
    main()
