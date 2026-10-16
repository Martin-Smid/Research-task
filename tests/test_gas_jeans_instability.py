"""Physics validation: Jeans instability of a self-gravitating gas (full Evolution path, gas only).

Uniform gas rho0 with a small density perturbation delta0*cos(kx), initially at rest. Linear theory:
    omega^2 = cs^2 k^2 - 4 pi G rho0
    unstable (omega^2 < 0): delta(t) = delta0 * cosh(gamma t),  gamma = sqrt(4 pi G rho0 - cs^2 k^2)
    stable   (omega^2 > 0): delta(t) = delta0 * cos(omega t)
Unstable case: the logarithmic growth slope d ln(delta)/dt at late time is compared with gamma*tanh(gamma t)
(independent of any constant time offset). Stable case: the signed amplitude is compared with cos(omega t)."""

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import contextlib
import io

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Simulation_Class import Simulation_Class

N = 32
H = 5e-4
RHO0 = 1.0e7      # Msun / kpc^3
DELTA0 = 1.0e-3
BOX = 20.0        # kpc, box is [-10, 10]^3


def run_jeans(cs, t_end):
    """Evolve the perturbed gas; returns (G, k, times, signed amplitude of the cos(kx) mode / delta0)."""
    k = 2 * np.pi / BOX
    with contextlib.redirect_stdout(io.StringIO()):  # the engine is chatty
        sim = Simulation_Class(dim=3, boundaries=[(-10, 10)] * 3, N=N, total_time=t_end, h=H,
                               order_of_evolution=2, use_gravity=True, static_potential=None,
                               save_max_vals=False, m_s=2.5e-22, use_sponge=False, self_int=False)
        x = np.array(sim.grids[0])
        rho = RHO0 * (1 + DELTA0 * np.cos(k * x))
        zeros = np.zeros_like(rho)
        gas = NBodyGas(simulation=sim, rho=rho, vx=zeros, vy=zeros, vz=zeros, cs=cs, gamma=5 / 3,
                       cfl=0.4, name="gas")
        # Adiabatic perturbation: P = P0 (rho/rho0)^gamma, so the mode is a pure acoustic/Jeans mode
        # (uniform specific energy would also excite a non-propagating entropy mode).
        pressure = RHO0 * cs ** 2 / (5 / 3) * (rho / RHO0) ** (5 / 3)
        gas.E = cp.asarray(pressure / (5 / 3 - 1.0))
        sim.add_baryons(gas)

    x_line = x[:, 0, 0]
    times, amplitude = [], []
    original_drift = gas.drift

    def recording_drift(dt, potential_grid, first_step=False, last_step=False):
        original_drift(dt, potential_grid, first_step, last_step)
        profile = cp.asnumpy(gas.rho.mean(axis=(1, 2))) / RHO0 - 1.0
        times.append((len(times) + 1) * H)
        amplitude.append(2 * np.mean(profile * np.cos(k * x_line)) / DELTA0)

    gas.drift = recording_drift
    with contextlib.redirect_stdout(io.StringIO()):
        sim.evolve(save_every=10 ** 9, diagnostics_every=10 ** 9)
    if sim.snapshot_directory:  # remove the output folder this test just created
        shutil.rmtree(sim.snapshot_directory, ignore_errors=True)
    return sim.G, k, np.array(times), np.array(amplitude)


def main():
    # ---- unstable mode: growth rate ----
    G, k, t, a = run_jeans(cs=20.0, t_end=0.14)
    gamma = np.sqrt(4 * np.pi * G * RHO0 - 20.0 ** 2 * k ** 2)
    i1, i2 = int(round(0.10 / H)) - 1, int(round(0.14 / H)) - 1
    slope = (np.log(a[i2]) - np.log(a[i1])) / (t[i2] - t[i1])
    expected = gamma * np.tanh(gamma * 0.5 * (t[i1] + t[i2]))
    growth_error = slope / expected - 1.0
    print(f"unstable: gamma_analytic = {gamma:.3f}/Gyr, late-time slope = {slope:.3f} vs {expected:.3f}"
          f" ({growth_error * 100:+.2f} %); amplitude grew x{a[-1]:.2f}")
    assert abs(growth_error) < 0.01, f"Jeans growth rate off by {growth_error * 100:+.2f} % (reference ~ +0.2 %)"
    assert a[-1] > 5.0, "perturbation did not grow as expected"

    # ---- stable mode: pressure-supported oscillation ----
    G, k, t, a = run_jeans(cs=100.0, t_end=0.14)
    omega = np.sqrt(100.0 ** 2 * k ** 2 - 4 * np.pi * G * RHO0)
    worst = 0.0
    for t_check in (0.03, 0.06, 0.09, 0.12):
        i = int(round(t_check / H)) - 1
        worst = max(worst, abs(a[i] - np.cos(omega * t[i])))
        print(f"stable:   t={t_check:.2f}: amplitude/delta0 = {a[i]:+.4f}  vs cos(omega t) = {np.cos(omega * t[i]):+.4f}")
    print(f"stable:   omega_analytic = {omega:.3f}/Gyr, worst |difference| = {worst:.3e}")
    assert worst < 0.03, f"stable-mode oscillation deviates from cos(omega t) by {worst:.3e}"
    print("PASS: Jeans growth rate and stable oscillation agree with linear theory")


if __name__ == "__main__":
    main()
