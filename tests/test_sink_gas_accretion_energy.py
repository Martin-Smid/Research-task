"""Regression check: gas accreted by a sink takes its energy with it.

accrete_from_gas lowered the density of the cells around the sink but left their energy, so a cell drained to the
floor kept its whole energy in almost no mass (specific energy and sound speed exploded, the gas CFL then asked
for 1e8 substeps). Now E scales like rho: the specific energy and the velocity of every cell stay the same, the
gas loses exactly what the sink gains, and the sink receives the momentum of what it ate."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


def main():
    sim = Simulation_Class(dim=3, boundaries=[(0, 20)] * 3, N=16, total_time=0.01, h=0.001, order_of_evolution=2,
                           use_gravity=True, use_units=True, use_sponge=False, self_int=False)
    x, y, z = [cp.asarray(g) for g in sim.grids]
    rho = 1.0e3 * (1.0 + 0.3 * cp.sin(x) * cp.cos(y))
    gas = NBodyGas(simulation=sim, rho=rho, vx=0.5 * y, vy=-0.5 * x, vz=0.1 * z, cs=10.0, gamma=5.0 / 3.0, name="gas")
    sink = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[1.0e9], initial_positions=[[10.3, 10.2, 10.1]],
                     initial_velocities=[[0.0, 0.0, 0.0]])
    sim.add_baryons(gas)
    sim.add_baryons(sink)

    rho0, E0 = gas.rho.copy(), gas.E.copy()
    spec0 = E0 / rho0
    mass0 = float(rho0.sum()) * sim.dV
    p0 = [float((rho0 * v).sum()) * sim.dV for v in (gas.vx, gas.vy, gas.vz)]
    accreted = sink.accrete_from_gas(gas, dt=1.0e-3, sound_speed=10.0)

    removed = mass0 - float(gas.rho.sum()) * sim.dV
    changed = int((gas.rho != rho0).sum())
    spec_error = float(cp.abs(gas.E / gas.rho - spec0).max() / spec0.max())
    print(f"accreted {accreted:.4e} Msun from {changed} cells; gas lost {removed:.4e}; "
          f"max change of the specific energy {spec_error:.2e} (relative); sink velocity {cp.asnumpy(sink.velocities)[0]}")
    assert changed > 0 and accreted > 0, "nothing was accreted, the test setup is wrong"
    assert abs(removed - accreted) <= 1e-9 * accreted, "gas loss and sink gain differ"
    assert spec_error < 1e-12, f"specific energy of the touched cells changed: {spec_error:.2e}"
    assert bool(cp.isfinite(gas.E).all())
    print("PASS: accretion removes energy together with mass (specific energy unchanged)")


if __name__ == "__main__":
    main()
