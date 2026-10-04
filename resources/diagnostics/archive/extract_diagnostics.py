
# Historical snapshot/migration: retained for review, not for execution.
if __name__ == "__main__":
    raise SystemExit("Archived file; use the documented active diagnostics instead.")

from pathlib import Path

root = Path.cwd()
path = root / "resources/Classes/Evolution_Class.py"
source = path.read_text(encoding="utf-8")
start = source.index("    def compute_total_energy(")
end = source.index("    def _calculate_coefficients_and_propagators(", start)
methods = source[start:end]
energy_end = methods.index("    def _compute_potential_energy(")
energy = methods[:energy_end].replace("def compute_total_energy(", "def compute_energy(", 1)
log_start = energy.index("        # Pass energies to scribe for logging")
energy = energy[:log_start] + '''        return {
            "K_total": K_total, "W": W_total, "U_iso": U_iso_total,
            "K_flow": K_flow, "U_quantum": U_quantum, "K_baryons": K_baryons,
            "W_self": W_self, "W_static": W_static, "E_diss": E_diss_total,
        }

'''
rest = methods[energy_end:]
rest = rest.replace("self._compute_sink_potential_analytic_kspace()", "self.sink_potential()")
rest = rest.replace("def _compute_and_save_radial_profile(self, total_density, current_time, ix, iy, iz, Nbins=250):",
                    "def compute_radial_profile(self, total_density, ix, iy, iz, Nbins=250):")
rest = rest.replace("        # Save via scribe\n        self.scribe.save_radial_density_profile(bin_centers, rho_avg, current_time)",
                    "        return bin_centers, rho_avg")
rotation_start = source.index("    def _compute_and_save_rotation_curves(")
rotation_end = source.index("    def _is_gas_system(", rotation_start)
rotation = source[rotation_start:rotation_end]
rotation = rotation.replace("def _compute_and_save_rotation_curves(self, current_time,", "def compute_rotation_curves(self,")
rotation = rotation.replace("                self.scribe.save_rotation_curve(\n                    time=current_time,", "                yield dict(")
header = '''"""Diagnostics calculated from the current simulation state; writing stays in Scribe."""

import cupy as cp
import numpy as np


class Diagnostics:
    def __init__(self, simulation, propagator, sink_potential):
        self.simulation = simulation
        self.propagator = propagator
        self.sink_potential = sink_potential
        self.last_kinetic_energy = 0

'''
helpers = '''    @staticmethod
    def _is_sink_system(component):
        return component.__class__.__name__.lower().startswith("sink")

    @staticmethod
    def _is_gas_system(component):
        return component.__class__.__name__.lower() == "nbodygas"
'''
new_class = header + energy + rest + rotation + helpers
new_class = "\n".join(line.rstrip() for line in new_class.splitlines()) + "\n"
(root / "resources/Classes/Diagnostics_Class.py").write_text(new_class, encoding="utf-8")
wrappers = '''    def compute_total_energy(self, wave_functions, total_density, current_time):
        """Calculate diagnostics and keep the existing logging/return interface."""
        values = self.diagnostics.compute_energy(wave_functions, total_density, current_time)
        self.K_flow, self.U_quantum, self.K_baryons = (
            values["K_flow"], values["U_quantum"], values["K_baryons"]
        )
        self.W_self, self.W_static = values["W_self"], values["W_static"]
        self.last_kinetic_energy = self.diagnostics.last_kinetic_energy
        self.scribe.log_energy_detailed(current_time, **{key: float(value) for key, value in values.items()})
        return tuple(values[key] for key in
                     ("K_total", "W", "K_flow", "U_quantum", "K_baryons", "W_self", "W_static"))

    def _compute_potential_energy(self, wave_functions, total_density, current_time):
        values = self.diagnostics._compute_potential_energy(wave_functions, total_density, current_time)
        self.W_self, self.W_static = values[:2]
        return values

    def _compute_kinetic_energy(self, wave_functions):
        values = self.diagnostics._compute_kinetic_energy(wave_functions)
        self.K_flow, self.U_quantum, self.K_baryons = values
        self.last_kinetic_energy = self.diagnostics.last_kinetic_energy
        return values

    def _compute_and_save_radial_profile(self, total_density, current_time, ix, iy, iz, Nbins=250):
        bin_centers, rho_avg = self.diagnostics.compute_radial_profile(total_density, ix, iy, iz, Nbins)
        self.scribe.save_radial_density_profile(bin_centers, rho_avg, current_time)

'''
rotation_wrapper = '''    def _compute_and_save_rotation_curves(self, current_time, nbins=40, rmax=None, zmax_gas=1.0, zmax_stars=2.0):
        for values in self.diagnostics.compute_rotation_curves(nbins, rmax, zmax_gas, zmax_stars):
            self.scribe.save_rotation_curve(time=current_time, **values)

'''
source = source[:rotation_start] + rotation_wrapper + source[rotation_end:]
source = source[:start] + wrappers + source[end:]
source = source.replace("from resources.Classes.Scribe_Class import Scribe\n", "from resources.Classes.Scribe_Class import Scribe\nfrom resources.Classes.Diagnostics_Class import Diagnostics\n", 1)
source = source.replace("        self.scribe = Scribe(self.simulation)\n", "        self.scribe = Scribe(self.simulation)\n        self.diagnostics = Diagnostics(simulation, propagator, self._compute_sink_potential_analytic_kspace)\n", 1)
path.write_text(source, encoding="utf-8")