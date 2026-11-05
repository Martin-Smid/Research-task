"""Test: Bondi accretion with the local sound speed and the relative velocity (SinkNBody.bondi_local).

A sink at rest sits in uniform gas that moves with v = 15 and is hotter than gas.cs says: the true sound speed is 20, gas.cs is
10. The default (constant gas.cs) must stay unchanged; with bondi_local the accreted mass changes by
cs_const^3 / (cs_local^2 + v^2)^(3/2) (the density is uniform, so the CIC interpolation is exact)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class

CS_CONSTANT, CS_TRUE, V_GAS = 10.0, 20.0, 15.0


def accreted(local):
    sim = Simulation_Class(dim=3, boundaries=[(-10, 10)] * 3, N=16, total_time=0.01, h=1.0e-3, order_of_evolution=2,
                           use_gravity=True, use_units=True, use_sponge=False, self_int=False)
    shape = (16,) * 3
    rho = cp.full(shape, 1.0e4)
    gas = NBodyGas(simulation=sim, rho=rho, vx=cp.full(shape, V_GAS), vy=cp.zeros(shape), vz=cp.zeros(shape), cs=CS_CONSTANT,
                   gamma=5 / 3, name="gas")
    gas.E = rho * CS_TRUE ** 2 / ((5 / 3) * (2 / 3)) + 0.5 * rho * V_GAS ** 2           # internal energy of cs = 20
    sink = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[1.0e8], initial_positions=[[0.3, 0.2, 0.1]],
                     initial_velocities=[[0.0, 0.0, 0.0]])
    sink.bondi_local = local
    sim.add_baryons(gas)
    sim.add_baryons(sink)
    return sink.accrete_from_gas(gas, dt=1.0e-5)


def main():
    constant, local = accreted(False), accreted(True)
    expected = CS_CONSTANT ** 3 / (CS_TRUE ** 2 + V_GAS ** 2) ** 1.5
    print(f"accreted: constant cs {constant:.4e}, local cs + v_rel {local:.4e}; ratio {local / constant:.5f}, expected {expected:.5f}")
    assert constant > 0.0 and abs(local / constant / expected - 1.0) < 1.0e-6
    print("PASS: local-cs Bondi-Hoyle accretion")


if __name__ == "__main__":
    main()
