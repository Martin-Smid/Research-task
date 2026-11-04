"""Test: spin_gravity_comparison.initial_fields builds an n-component field of equal-mass independent Gaussian components for
half-integer s (n = 2 s + 1 = 4, 6, 8, 9 from distinct irreducible spins 0, 1, 2), and leaves integer s unchanged.

Checks: the number of fields is 2 s + 1, all components carry the same mass, the total mass is the requested one, and
variance(rho / rho_mean) * n = 1 within 5 % averaged over seeds (the 1 / n law of independent components).
Before the per-part weighting the n = 6 case gave variance * n = 1.86 (one component carried a 5 times larger mass)."""

from pathlib import Path
import importlib.util
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location("spin_gravity_comparison", PROJECT_ROOT / "tests" / "spin" / "spin_gravity_comparison.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

for spin, n in ((1.5, 4), (2.5, 6), (3.5, 8), (4, 9)):
    values, spreads = [], []
    for seed in (1, 2, 3):
        fields = module.initial_fields(spin, seed, 64)
        assert len(fields) == n, (spin, len(fields))
        rho = sum(np.abs(f) ** 2 for f in fields)
        masses = [float(np.sum(np.abs(f) ** 2)) for f in fields]
        spreads.append(max(masses) / min(masses))
        values.append(np.var(rho / rho.mean()) * n)
        assert abs(rho.mean() - module.RHO_MEAN) < 1e-6 * module.RHO_MEAN
    assert max(spreads) < 1.0001, (spin, spreads)
    assert abs(np.mean(values) - 1.0) < 0.05, (spin, values)
print("PASS")
