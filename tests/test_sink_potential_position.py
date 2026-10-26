"""Regression check: the analytic sink potential must be centred on the sink, whatever the box boundaries.

The sink enters the k-space potential as a phase exp(-i k x). FFT index 0 is the lower box boundary, so without
subtracting it a sink at (0, 0, 0) in a [-10, 10] box pulled towards the corner (-10, -10, -10)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


def sink_potential_minimum(boundaries, position):
    sim = Simulation_Class(dim=3, boundaries=boundaries, N=32, total_time=0.01, h=0.001, order_of_evolution=2,
                           use_gravity=True, use_units=True, use_sponge=False, self_int=False)
    sink = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[1.0e9], initial_positions=[position],
                     initial_velocities=[[0.0, 0.0, 0.0]])
    sim.add_baryons(sink)
    sim.initialize_simulation()
    phi = sim.evolution._compute_sink_potential_analytic_kspace()
    index = np.unravel_index(int(cp.argmin(phi)), phi.shape)
    found = [float(cp.asnumpy(g)[index]) for g in sim.grids]
    return found, list(sim.dx)


def main():
    for boundaries, position in (([(-10, 10)] * 3, [0.0, 0.0, 0.0]),
                                 ([(-10, 10)] * 3, [2.5, -3.75, 1.25]),
                                 ([(0, 20)] * 3, [10.0, 10.0, 10.0]),
                                 ([(-25, 25), (-10, 10), (5, 15)], [4.0, -2.0, 9.0])):
        found, dx = sink_potential_minimum(boundaries, position)
        print(f"box {boundaries[0]}.. sink at {position}: potential minimum at {[round(f, 3) for f in found]}")
        assert all(abs(f - p) <= d for f, p, d in zip(found, position, dx)), "sink potential is not centred on the sink"
    print("PASS: the sink potential is centred on the sink for any box boundaries")


if __name__ == "__main__":
    main()
