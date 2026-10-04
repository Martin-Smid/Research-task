
# Historical snapshot/migration: retained for review, not for execution.
if __name__ == "__main__":
    raise SystemExit("Archived file; use the documented active diagnostics instead.")

from pathlib import Path
path = Path("resources/Classes/Diagnostics_Class.py")
source = path.read_text(encoding="utf-8")
source = source.replace('"""Compute all energy components (waves + baryons) and log them."""',
                        '"""Return energy components; Evolution passes them to Scribe."""')
source = source.replace('        #E_rad_total = 0.0   might need later\n', '')
source = source.replace('                # **Optional: if you already track radiated energy on gas**\n                #if hasattr(sys, "E_radiated"):\n                #    E_rad_total += float(sys.E_radiated)\n', '')
source = source.replace('                for sys in self.simulation.baryonic_matter:\n',
                        '                # Preserve the legacy gas accumulation during this extraction.\n                for sys in self.simulation.baryonic_matter:\n', 1)
source = source.replace('        rho_w = total_density\n', '        rho_w = total_density  # Waves and non-sink baryons, as supplied by Evolution.\n', 1)
source = source.replace('  # from wave density only', '')
source = source.replace('        dx = self.simulation.dx\n', '')
source = source.replace('Compute and save rotation curves for all baryonic components that implement',
                        'Yield rotation curves for all baryonic components that implement')
while '\n\n\n' in source:
    source = source.replace('\n\n\n', '\n\n')
path.write_text(source, encoding="utf-8")