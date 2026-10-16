"""Regression check: gas internal energy must be counted once in U_iso, however many
baryonic systems (sinks, particles, ...) share the simulation."""

from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resources.Classes.Diagnostics_Class import Diagnostics


class NBodyGas:  # name matters: Diagnostics recognises gas by class name
    def internal_energy(self):
        return 7.0


class SinkNBody:  # name matters: Diagnostics recognises sinks by class name
    E_diss_kin_total = 0.0
    E_diss_formation_total = 0.0


def u_iso(components):
    diagnostics = Diagnostics.__new__(Diagnostics)
    diagnostics.simulation = SimpleNamespace(baryonic_matter=components)
    diagnostics._compute_kinetic_energy = lambda wave_functions: (0.0, 0.0, 0.0)
    diagnostics._compute_potential_energy = lambda *args: (0.0, 0.0, 0.0)
    return diagnostics.compute_energy([], None, 0.0)["U_iso"]


def main():
    gas_only = u_iso([NBodyGas()])
    gas_and_sink = u_iso([NBodyGas(), SinkNBody()])
    assert gas_only == 7.0, f"gas alone: U_iso={gas_only}, expected 7.0"
    assert gas_and_sink == 7.0, f"gas + sink: U_iso={gas_and_sink}, expected 7.0 (counted once per system?)"
    print("PASS: gas internal energy is counted once, with or without a sink")


if __name__ == "__main__":
    main()
