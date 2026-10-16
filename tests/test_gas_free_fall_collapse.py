"""Physics validation: free-fall collapse of a cold uniform gas sphere (gravity + hydro, gas only).

A pressure-free uniform sphere collapses homologously: every mass shell follows r = r0 cos^2(eta) with
t / t_ff = (2/pi) (eta + sin(eta) cos(eta)), t_ff = sqrt(3 pi / (32 G rho0)). We compare the radius that
contains half of the mass with this exact solution, check that the collapse reaches its densest point near
t_ff, and that free_fall_estimate() agrees with the formula."""

from pathlib import Path
import contextlib
import io
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas, free_fall_estimate
from resources.Classes.Simulation_Class import Simulation_Class

N = 64
MASS = 1.0e11          # Msun
RADIUS = 5.0           # kpc
KM_S_TO_KPC_GYR = 1.0227121650537077


def analytic_radius_fraction(tau):
    """r(t)/r0 of every shell at tau = t / t_ff (0 <= tau <= 1)."""
    lo, hi = 0.0, np.pi / 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (2 / np.pi) * (mid + np.sin(mid) * np.cos(mid)) < tau:
            lo = mid
        else:
            hi = mid
    return float(np.cos(0.5 * (lo + hi)) ** 2)


def run_collapse():
    """Returns (t / t_ff, r50 / r50_0, rho_max / rho0) sampled every few steps."""
    estimate = free_fall_estimate(MASS, RADIUS, print_summary=False)
    h, t_ff = estimate["h"], estimate["t_ff"]
    with contextlib.redirect_stdout(io.StringIO()):  # the engine is chatty
        sim = Simulation_Class(dim=3, boundaries=[(-10, 10)] * 3, N=N, total_time=estimate["total_time"], h=h,
                               order_of_evolution=2, use_gravity=True, static_potential=None,
                               save_max_vals=False, m_s=2.5e-22, use_sponge=False, self_int=False)
        x, y, z = (np.array(g) for g in sim.grids)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        rho = np.where(r < RADIUS, 1.0, 1e-3)
        rho *= MASS / (rho.sum() * sim.dV)
        zeros = np.zeros_like(rho)
        gas = NBodyGas(simulation=sim, rho=rho, vx=zeros, vy=zeros, vz=zeros, name="gas",
                       cs=1.0 * KM_S_TO_KPC_GYR, gamma=5 / 3, tcool=None, e_floor=1e-10,
                       rho_floor=1e-12, cfl=0.4)
        sim.add_baryons(gas)

    order = np.argsort(r.ravel())
    radii = r.ravel()[order]

    def r50():
        cumulative = cp.asnumpy(gas.rho).ravel()[order].cumsum()
        return radii[np.searchsorted(cumulative, 0.5 * cumulative[-1])]

    r50_start, rho_start = r50(), float(gas.rho.max())
    samples, step = [], [0]
    original_drift = gas.drift

    def recording_drift(dt, potential_grid, first_step=False, last_step=False):
        original_drift(dt, potential_grid, first_step, last_step)
        step[0] += 1
        if step[0] % 5 == 0:
            samples.append((step[0] * h / t_ff, r50() / r50_start, float(gas.rho.max()) / rho_start))

    gas.drift = recording_drift
    with contextlib.redirect_stdout(io.StringIO()):
        sim.evolve(save_every=10 ** 9, diagnostics_every=10 ** 9)
    if sim.snapshot_directory:  # remove the output folder this test just created
        shutil.rmtree(sim.snapshot_directory, ignore_errors=True)
    return np.array(samples)


def main():
    # helper agrees with the formula (1e12 Msun in 5 kpc: t_ff = 5.85 Myr)
    estimate = free_fall_estimate(1.0e12, 5.0, print_summary=True)
    assert abs(estimate["t_ff"] - 0.005853) < 5e-5, f"t_ff off: {estimate['t_ff']}"
    assert abs(estimate["h"] - estimate["t_ff"] / 100) < 1e-15

    samples = run_collapse()
    tau, r50, rho_max = samples[:, 0], samples[:, 1], samples[:, 2]
    worst = 0.0
    for t_check, tolerance in ((0.25, 0.02), (0.5, 0.04), (0.75, 0.08)):
        i = int(np.argmin(np.abs(tau - t_check)))
        expected = analytic_radius_fraction(tau[i])
        worst = max(worst, abs(r50[i] - expected))
        print(f"t/t_ff = {tau[i]:.2f}: r50/r50_0 = {r50[i]:.3f}  analytic {expected:.3f}  (|diff| {abs(r50[i] - expected):.3f})")
        assert abs(r50[i] - expected) < tolerance, f"r50 at t/t_ff={tau[i]:.2f} off by {abs(r50[i] - expected):.3f}"
    t_peak = float(tau[int(np.argmax(rho_max))])
    print(f"densest point at t/t_ff = {t_peak:.2f} (peak rho_max/rho_0 = {rho_max.max():.3g}); worst |diff| = {worst:.3f}")
    assert 0.9 <= t_peak <= 1.25, f"collapse peak at t/t_ff = {t_peak:.2f}, expected near 1"
    assert rho_max.max() > 20, "the sphere did not collapse"
    print("PASS: cold-sphere collapse follows the free-fall solution")


if __name__ == "__main__":
    main()
