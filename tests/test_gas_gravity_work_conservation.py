"""Regression check: over one gas drift, energy gained by the gas plus the change of the
gravitational energy W must cancel (energy-conserving gravity-hydro coupling)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Simulation_Class import Simulation_Class

KM_S_TO_KPC_GYR = 1.0227121650537077


def build():
    sim = Simulation_Class(dim=3, boundaries=[(-10, 10)] * 3, N=32, total_time=0.001, h=1e-3,
                           order_of_evolution=2, use_gravity=True, static_potential=None,
                           save_max_vals=False, m_s=2.5e-22, use_sponge=False, self_int=False)
    x, y, z = (np.array(g) for g in sim.grids)
    R = np.sqrt(x ** 2 + y ** 2)
    rho = np.exp(-R / 4.0) * np.exp(-np.abs(z) / 0.8) + 1e-12
    rho *= 5e10 / (rho.sum() * sim.dV)
    zeros = np.zeros_like(rho)
    gas = NBodyGas(simulation=sim, rho=rho, vx=zeros, vy=zeros, vz=zeros, name="gas",
                   cs=18.0 * KM_S_TO_KPC_GYR, gamma=5 / 3, tcool=None, cfl=0.4)
    sim.add_baryons(gas)
    sim.initialize_simulation()
    return sim, gas


def energy_residual(use_flux_work, dt=1e-5, corrector=False):
    """Return (dE_gas + dW) / |gravity work| over one drift of a collapsing disk.
    corrector=True also hands the gas a potential_update(d_rho) -> (forces, phi) like Evolution does."""
    sim, gas = build()
    prop, dV = sim.propagator, float(sim.dV)
    phi = prop.solve_poisson(gas.rho)
    forces = prop.compute_force_grids_from_potential(phi)
    gas.vel = [1e-2 * a for a in forces]  # infall, so gravity does real work
    gas.vx, gas.vy, gas.vz = gas.vel
    gas.E = gas.rho * gas.cs ** 2 / (gas.gamma * (gas.gamma - 1.0)) + 0.5 * gas.rho * sum(v * v for v in gas.vel)
    work = float(cp.sum(gas.rho * sum(v * a for v, a in zip(gas.vel, forces)))) * dV * dt

    def W():
        return 0.5 * float(cp.sum(gas.rho * prop.solve_poisson(gas.rho))) * dV

    gas.phi_total = phi if use_flux_work else None
    if corrector:
        rho0 = gas.rho.copy()

        def potential_update(d_rho):
            new_phi = prop.solve_poisson(rho0 + d_rho)
            return prop.compute_force_grids_from_potential(new_phi), new_phi

        gas.potential_update = potential_update
    W0, E0 = W(), float(cp.sum(gas.E)) * dV
    gas.drift(dt, forces)
    return ((float(cp.sum(gas.E)) * dV - E0) + (W() - W0)) / abs(work)


def main():
    old = energy_residual(use_flux_work=False)
    new = energy_residual(use_flux_work=True)
    print(f"cell-centred work (old): (dE_gas + dW)/|work| = {old:+.2e}")
    print(f"interface-flux work (new): (dE_gas + dW)/|work| = {new:+.2e}")
    assert abs(new) < 1e-3, f"gravity work does not balance dW: {new:+.2e}"
    print("PASS: gas gravity work balances the change of W")

    # A long window (frozen Phi is no longer a good approximation): the corrector must beat it.
    frozen = energy_residual(use_flux_work=True, dt=2e-3)
    corrected = energy_residual(use_flux_work=True, dt=2e-3, corrector=True)
    print(f"dt=2e-3 frozen potential:   (dE_gas + dW)/|work| = {frozen:+.2e}")
    print(f"dt=2e-3 potential corrector: (dE_gas + dW)/|work| = {corrected:+.2e}")
    assert abs(corrected) < 0.1 * abs(frozen), "corrector does not reduce the energy error"
    print("PASS: refreshed potential makes the gas energy error second order")


if __name__ == "__main__":
    main()
