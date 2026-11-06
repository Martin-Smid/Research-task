"""Test: with Simulation_Class.baryon_time_centered_force = True particles feel a force centred in time, so the orbit
in a time-varying source is second order in the step h (the frozen force of test_particle_breathing_soliton is first order).

Same setup as test_particle_breathing_soliton.py (test particle in the core of a breathing soliton, self-convergence
against the finest h). With the frozen force the apparent order is 1.2 and 1.6 (mean ~1.4) and the error at h = 4e-4 is
1.3e-3 r0; for a true order 2 the apparent order is ~2.1 and ~2.3.

Checks
    1) the mean apparent order is at least MIN_MEAN_ORDER,
    2) the position error at h = 4e-4 is at least 4x smaller than with the frozen force (MAX_ERROR)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import numpy as np

import test_particle_breathing_soliton as frozen
from resources.Classes.Simulation_Class import Simulation_Class

MIN_MEAN_ORDER = 1.8            # frozen force: ~1.4, a second-order scheme: ~2.2
MAX_ERROR = 3.0e-4              # frozen force: 1.3e-3 r0 at h = 4e-4 (multiples of r0)


def main():
    previous = Simulation_Class.baryon_time_centered_force
    Simulation_Class.baryon_time_centered_force = True
    try:
        reference = frozen.run_orbit(frozen.H_REFERENCE)
        print(f"time-centred force, reference h = {frozen.H_REFERENCE:.1e} ({len(reference) - 1} steps)")
        print(f"\n{'h [Gyr]':>10} {'steps':>7} {'max position error / r0':>24} {'apparent order':>14}")

        errors = []
        for h in frozen.H_LIST:
            orbit = frozen.run_orbit(h)
            stride = int(round(h / frozen.H_REFERENCE))
            difference = np.linalg.norm(orbit - reference[::stride][: len(orbit)], axis=1)
            errors.append(float(difference.max() / frozen.R0))
            order = np.log2(errors[-2] / errors[-1]) if len(errors) > 1 else float("nan")
            print(f"{h:>10.1e} {len(orbit) - 1:>7} {errors[-1]:>24.3e} {order:>14.2f}")
    finally:
        Simulation_Class.baryon_time_centered_force = previous

    mean_order = float(np.log2(errors[0] / errors[-1]) / (len(errors) - 1))
    print(f"mean apparent order: {mean_order:.2f}")
    assert mean_order >= MIN_MEAN_ORDER, f"apparent order {mean_order:.2f} is below {MIN_MEAN_ORDER}: the force is not centred in time"
    assert errors[0] < MAX_ERROR, f"position error {errors[0]:.2e} r0 at h = {frozen.H_LIST[0]:.0e} is above {MAX_ERROR}"
    print(f"PASS: time-centred force gives apparent order {mean_order:.2f}, error {errors[0]:.1e} r0 at h = {frozen.H_LIST[0]:.0e}")


if __name__ == "__main__":
    main()
