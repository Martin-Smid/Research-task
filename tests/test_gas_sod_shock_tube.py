"""Physics validation: Sod shock tube (no gravity) against the exact Riemann solution.

1-D periodic box [0, 1) with the high state (rho=1, P=1) on [0.25, 0.75) and the low state
(rho=0.125, P=0.1) elsewhere, gamma=1.4. The jump at x=0.75 is a standard Sod problem; at t=0.1 the
waves of the two mirrored jumps have not met. We compare the density on x >= 0.5 with the exact solution.
Checks: mass conservation, no over/undershoot, L1 error below a tolerance, and convergence with N."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Simulation_Class import Simulation_Class

GAMMA = 1.4
T_END = 0.1
JUMP = 0.75
LEFT = (1.0, 0.0, 1.0)    # rho, u, P (high state, left of the jump)
RIGHT = (0.125, 0.0, 0.1)


def exact_density(xs, t, x0):
    """Exact Riemann solution (Toro) for the Sod states; returns rho(x, t)."""
    rl, ul, pl = LEFT
    rr, ur, pr = RIGHT
    g = GAMMA
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def f(p, rk, pk, ck):
        if p > pk:  # shock
            return (p - pk) * np.sqrt((2 / ((g + 1) * rk)) / (p + (g - 1) / (g + 1) * pk))
        return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)  # rarefaction

    p_star = 0.5 * (pl + pr)
    for _ in range(100):  # Newton iteration for the star pressure
        F = f(p_star, rl, pl, cl) + f(p_star, rr, pr, cr) + (ur - ul)
        dp = 1e-8 * p_star
        dF = (f(p_star + dp, rl, pl, cl) + f(p_star + dp, rr, pr, cr)
              - f(p_star, rl, pl, cl) - f(p_star, rr, pr, cr)) / dp
        new = max(p_star - F / dF, 1e-8)
        converged = abs(new - p_star) < 1e-12
        p_star = new
        if converged:
            break
    u_star = 0.5 * (ul + ur) + 0.5 * (f(p_star, rr, pr, cr) - f(p_star, rl, pl, cl))

    rho = np.empty_like(xs)
    for i, x in enumerate(xs):
        s = (x - x0) / t
        if s <= u_star:  # left of the contact: rarefaction fan (Sod) or shock
            if p_star > pl:
                shock = ul - cl * np.sqrt((g + 1) / (2 * g) * p_star / pl + (g - 1) / (2 * g))
                star = rl * ((p_star / pl + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p_star / pl + 1))
                rho[i] = rl if s < shock else star
            else:
                c_star = cl * (p_star / pl) ** ((g - 1) / (2 * g))
                head, tail = ul - cl, u_star - c_star
                if s < head:
                    rho[i] = rl
                elif s > tail:
                    rho[i] = rl * (p_star / pl) ** (1 / g)
                else:
                    c = 2 / (g + 1) * (cl + (g - 1) / 2 * (ul - s))
                    rho[i] = rl * (c / cl) ** (2 / (g - 1))
        else:  # right of the contact: shock
            shock = ur + cr * np.sqrt((g + 1) / (2 * g) * p_star / pr + (g - 1) / (2 * g))
            star = rr * ((p_star / pr + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p_star / pr + 1))
            rho[i] = star if s < shock else rr
    return rho


def run_sod(n):
    """Evolve the Sod problem with NBodyGas; returns (cell positions, density, initial density)."""
    sim = Simulation_Class(dim=1, boundaries=[(0.0, 1.0)], N=n, total_time=T_END, h=1e-3,
                           order_of_evolution=2, use_gravity=False, static_potential=None,
                           save_max_vals=False, m_s=2.5e-22, use_sponge=False, self_int=False,
                           use_units=False)
    x = np.array(sim.grids[0]).ravel()
    x = x - x.min()
    high = (x >= 0.25) & (x < 0.75)
    rho0 = np.where(high, 1.0, 0.125)
    pressure = np.where(high, 1.0, 0.1)
    zeros = np.zeros_like(rho0)
    gas = NBodyGas(simulation=sim, rho=rho0, vx=zeros, vy=zeros, vz=zeros, cs=1.0, gamma=GAMMA, cfl=0.4,
                   name="sod", rho_floor=1e-12, e_floor=1e-12, max_substeps=10 ** 6)
    gas.E = cp.asarray(pressure / (GAMMA - 1.0))  # fluid at rest
    no_force = (cp.zeros(n), cp.zeros(n), cp.zeros(n))
    t = 0.0
    while t < T_END - 1e-14:
        dt = min(gas._compute_cfl_timestep(), T_END - t)
        gas.drift(dt, no_force)
        t += dt
    return x, cp.asnumpy(gas.rho), rho0


def l1_error(n):
    x, rho, rho0 = run_sod(n)
    right_half = x >= 0.5
    err = float(np.mean(np.abs(rho[right_half] - exact_density(x[right_half], T_END, JUMP))))
    mass_change = abs(rho.sum() - rho0.sum()) / rho0.sum()
    return err, mass_change, float(rho.min()), float(rho.max())


def main():
    err200, mass200, rho_min, rho_max = l1_error(200)
    err400, mass400, _, _ = l1_error(400)
    print(f"N=200: L1(rho) = {err200:.3e}   N=400: L1(rho) = {err400:.3e}   ratio = {err400 / err200:.2f}")
    print(f"mass change: {mass200:.1e} (N=200), {mass400:.1e} (N=400); rho range at N=200: [{rho_min:.4f}, {rho_max:.4f}]")

    assert max(mass200, mass400) < 1e-12, "mass is not conserved"
    assert rho_min > 0.125 - 1e-3 and rho_max < 1.0 + 1e-3, "density over/undershoots the initial states"
    assert err200 < 1.5e-2, f"Sod L1 error at N=200 too large: {err200:.3e} (reference ~9.6e-3)"
    assert err400 < 8.0e-3, f"Sod L1 error at N=400 too large: {err400:.3e} (reference ~5.3e-3)"
    assert err400 < 0.75 * err200, "Sod error does not decrease with resolution"
    print("PASS: Sod shock tube agrees with the exact solution and converges with resolution")


if __name__ == "__main__":
    main()
