
# Historical snapshot/migration: retained for review, not for execution.
if __name__ == "__main__":
    raise SystemExit("Archived file; use the documented active diagnostics instead.")

from pathlib import Path
root = Path.cwd()
path = root / "resources/Classes/Diagnostics_Class.py"
source = path.read_text(encoding="utf-8")
source = source.replace("import numpy as np\n", "import numpy as np\nfrom resources.Classes.Rotation_Curve_Class import RotationCurves\n", 1)
source = source.replace("        self.last_kinetic_energy = 0\n", "        self.last_kinetic_energy = 0\n        self.rotation_curves = RotationCurves(simulation)\n", 1)
start = source.index("    def compute_rotation_curves(")
end = source.index("    @staticmethod", start)
source = source[:start] + '''    def compute_rotation_curves(self, nbins=40, rmax=None, zmax_gas=1.0, zmax_stars=2.0,
                                mass_fraction=0.5):
        """Find each component's own frame and yield its GPU-computed rotation curve."""
        components = self.simulation.baryonic_matter
        names = [getattr(component, "name", f"baryons_{i}") for i, component in enumerate(components)]
        for i, component in enumerate(components):
            if self._is_sink_system(component):
                continue
            name = names[i] if names.count(names[i]) == 1 else f"{names[i]}_{i}"
            try:
                if self._is_gas_system(component):
                    data = self.rotation_curves.gas_data(component)
                    zmax = zmax_gas
                elif hasattr(component, "positions") and hasattr(component, "m_particle"):
                    data = self.rotation_curves.baryon_data(component)
                    zmax = zmax_stars
                else:
                    continue
                R, vphi, sigma, weights, frame = self.rotation_curves.calculate(
                    data, nbins, rmax, zmax, mass_fraction
                )
                yield dict(component_name=name, R_centers=R, vphi_mean=vphi,
                           vphi_std=sigma, weights=weights, frame=frame)
            except Exception as error:
                print(f"[Diagnostics] Warning: rotation curve for {name}: {error}")

''' + source[end:]
path.write_text(source, encoding="utf-8")
for filename, gas in (("NBody.py", False), ("NBodyGas.py", True)):
    path = root / "resources/Classes/Nbody_classes" / filename
    source = path.read_text(encoding="utf-8")
    start = source.index("    def compute_rotation_curve(")
    signature = '''    def compute_rotation_curve(self, nbins=40, center=None, zmax=None, rmax=None,
                               axis=None, mass_fraction=0.5'''
    if gas:
        # Keep mass_weighted in its historical positional slot.
        signature = '''    def compute_rotation_curve(self, nbins=40, center=None, zmax=None, rmax=None,
                               mass_weighted=True, axis=None, mass_fraction=0.5'''
    method = "gas_data" if gas else "baryon_data"
    weighted = ", mass_weighted=mass_weighted" if gas else ""
    source = source[:start] + signature + '''):
        """Return a curve in the component's own frame; weights are masses by default."""
        from resources.Classes.Rotation_Curve_Class import RotationCurves
        rotation = RotationCurves(self.simulation)
        data = rotation.''' + method + '''(self, center, axis)
        return rotation.calculate(data, nbins, rmax, zmax, mass_fraction''' + weighted + ''')[:4]
'''
    path.write_text(source, encoding="utf-8")
path = root / "resources/Classes/Evolution_Class.py"
source = path.read_text(encoding="utf-8").replace("                    rmax=15.0,\n", "                    rmax=None,\n", 1)
path.write_text(source, encoding="utf-8")