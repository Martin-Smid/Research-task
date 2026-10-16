"""Regression check: the CFL time step must be re-evaluated before every gas substep.

It used to be computed once per window, so a flow that accelerates inside the window (e.g. after a collapse)
ran with a stale step and the true Courant number grew past 1 (instability, then NaN)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas


def make_gas(estimates):
    """Gas whose CFL estimate follows `estimates` (one value per call) and which records every hydro step."""
    shape = (2, 2, 2)
    gas = NBodyGas.__new__(NBodyGas)
    gas.dim = 3
    gas.max_substeps = 100
    gas.rho = cp.ones(shape, dtype=cp.float64)
    gas.vel = [cp.zeros(shape, dtype=cp.float64) for _ in range(3)]
    gas.vx, gas.vy, gas.vz = gas.vel
    gas.E = cp.ones(shape, dtype=cp.float64)
    calls = {"estimate": 0, "dt": []}

    def estimate():
        value = estimates[min(calls["estimate"], len(estimates) - 1)]
        calls["estimate"] += 1
        return value

    gas._compute_cfl_timestep = estimate
    gas._get_acceleration_grids = lambda potential_grid: (0.0, 0.0, 0.0)
    gas._gravity_kick = lambda ax, ay, az, dt: 0.0
    gas._hydro_step = lambda dt: calls["dt"].append(dt)
    gas._apply_cooling = lambda dt: None
    return gas, calls


def main():
    # The allowed step shrinks from 0.5 to 0.25 while the window of length 1.0 is being advanced.
    gas, calls = make_gas([0.5, 0.25, 0.25, 0.25])
    gas.drift(1.0, None)
    steps = calls["dt"]
    print(f"substeps used: {steps}")
    assert abs(sum(steps) - 1.0) < 1e-12, f"substeps do not add up to the window: {steps}"
    assert steps == [0.5, 0.25, 0.25], f"substeps ignore the current CFL estimate: {steps}"

    # A window whose estimate stays large is advanced in one step, as before.
    gas, calls = make_gas([5.0])
    gas.drift(1.0, None)
    assert calls["dt"] == [1.0], f"unexpected substeps for a relaxed CFL: {calls['dt']}"

    # A non-finite estimate (NaN/inf in the state) must give a clear error.
    gas, calls = make_gas([float("nan")])
    try:
        gas.drift(1.0, None)
    except FloatingPointError as error:
        print(f"non-finite estimate raises FloatingPointError: {error}")
    else:
        raise AssertionError("a NaN CFL estimate was accepted")
    print("PASS: gas substeps follow the current CFL estimate")


if __name__ == "__main__":
    main()
