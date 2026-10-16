"""Regression check: analytic sink gravity must not also enter Poisson density."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp

from resources.Classes.Evolution_Class import Evolution_Class


class DummySimulation:
    N = 4
    dim = 1
    h_bar_tilde = 1.0


class SinkDummy:
    def deposit_to_grid(self):
        return cp.full((4,), 100.0, dtype=cp.float64)


class GasDummy:
    def deposit_to_grid(self):
        return cp.full((4,), 3.0, dtype=cp.float64)


class WaveDummy:
    def __init__(self):
        self.psi = cp.ones((4,), dtype=cp.complex128)

    def calculate_density(self):
        return cp.full((4,), 2.0, dtype=cp.float64)


class RecordingPropagator:
    def __init__(self, expected_density):
        self.expected_density = expected_density
        self.received_density = None
        self.gravity_calls = 0

    def compute_gravity_potential(self, density):
        self.gravity_calls += 1
        cp.testing.assert_allclose(density, self.expected_density)
        return cp.zeros_like(density, dtype=cp.float64)

    def compute_total_propagator(self, density, **kwargs):
        self.received_density = density.copy()
        cp.testing.assert_allclose(density, self.expected_density)
        return cp.ones_like(density, dtype=cp.complex128)


def main():
    simulation = DummySimulation()
    simulation.baryonic_matter = [SinkDummy(), GasDummy()]

    evolution = Evolution_Class.__new__(Evolution_Class)
    evolution.simulation = simulation
    evolution.h = 0.2
    evolution.coefficients = {"full": 1.0}
    evolution.static_propagators = {}

    wave = WaveDummy()
    expected_environment_density = cp.full((4,), 5.0, dtype=cp.float64)
    total_density = evolution._compute_total_density([wave])
    cp.testing.assert_allclose(total_density, expected_environment_density)

    evolution.propagator = RecordingPropagator(expected_environment_density)
    sink_potential = cp.full((4,), 0.25, dtype=cp.float64)
    evolution._compute_sink_potential_analytic_kspace = lambda: sink_potential

    evolution._kick_all_wave_functions(
        [wave], total_density, is_first_step=False, is_last_step=False
    )
    assert evolution.propagator.gravity_calls == 1

    expected_wave = cp.exp(
        -1j * sink_potential * evolution.h / simulation.h_bar_tilde
    )
    cp.testing.assert_allclose(wave.psi, expected_wave, rtol=0.0, atol=1e-14)

    print("PASS: Poisson density contains wave + gas but excludes the sink")
    print("PASS: the wave receives the analytic sink propagator exactly once")


if __name__ == "__main__":
    main()
