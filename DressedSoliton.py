"""Stationary ground state of the spherically symmetric Schroedinger-Poisson system with a central black hole.

Radial system (lengths in kpc, hbar_tilde = hbar / m in kpc^2/Gyr, the same variables as ModosFinder.py and the thesis, Eq. 5.3-5.5)
    1/2 (Phi'' + 2 Phi' / r) = (U + U_BH - omega) Phi,     U'' + 2 U' / r = 4 pi Phi^2,
    U_BH(r) = -kappa erf(r / (sqrt(2) eps)) / r   (Gaussian black hole of width eps, the sink potential of the code),
                or -kappa / r  for eps = 0 (point mass),     kappa = G M_BH / hbar_tilde^2   [1/kpc].
Density rho = (hbar_tilde^2 / G) Phi^2, total mass M = (hbar_tilde^2 / G) 4 pi integral Phi^2 r^2 dr.

Solved by shooting in the single eigenvalue E: with V(0) = 0 (V'' + 2 V' / r = 4 pi Phi^2) the equation reads
1/2 Delta Phi = (V + U_BH - E) Phi, Phi(0) = 1. The ground state is the lowest E for which Phi is nodeless and decays (bisection between
"Phi crosses zero", E too high, and "Phi turns upward", E too low). Then c = lim (V + M~(r) / r) gives U = V - c, omega = E - c, U0 = -c.
Unlike the free soliton, the family is NOT invariant under rescaling by the mass alone: Phi(0) = 1 fixes the central density
rho_0 = hbar_tilde^2 / G, and for another central density Phi(0) = a^2 the black hole must be rescaled, kappa_tilde = kappa / a,
lengths r_tilde = a r (see rescale()).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf

try:                                 # the same G as Simulation_Class (astropy, kpc^3 / (Msun Gyr^2)); ModosFinder.py uses 4.49864e-6, 3e-5 higher
    from astropy import constants as _constants
    G = float(_constants.G.to("kpc3 / (Msun Gyr2)").value)
except Exception:
    G = 4.4985021514695e-6
HBAR_OVER_M_1EV = 1.960695038        # hbar / m for m = 1e-21 eV [kpc^2/Gyr], hbar_tilde ~ 1/m (checked against Simulation_Class below)


def hbar_tilde(m_ev):
    """hbar / m [kpc^2 / Gyr] for a particle of mass m_ev [eV] (hbar_tilde = 1.9607 kpc^2/Gyr at 1e-21 eV, scales as 1/m)."""
    return HBAR_OVER_M_1EV * 1.0e-21 / m_ev


def u_bh(r, kappa, eps):
    r = np.asarray(r, dtype=float)
    if kappa == 0.0:
        return np.zeros_like(r)
    if eps > 0.0:
        safe = np.where(r > 0.0, r, 1.0)
        return np.where(r > 0.0, -kappa * erf(safe / (np.sqrt(2.0) * eps)) / safe, -kappa * np.sqrt(2.0 / np.pi) / eps)
    return -kappa / r


def _initial_state(E, kappa, eps, r0):
    """Series of the regular solution at small r0: Phi, Phi', V, V'."""
    if eps > 0.0:
        c = -kappa * np.sqrt(2.0 / np.pi) / eps - E                         # V(0) = 0
        a = c / 3.0
        y, yp = 1.0 + a * r0 ** 2, 2.0 * a * r0
    else:
        a = (kappa ** 2 - E) / 3.0
        y, yp = 1.0 - kappa * r0 + a * r0 ** 2, -kappa + 2.0 * a * r0
    return [y, yp, 2.0 * np.pi / 3.0 * r0 ** 2, 4.0 * np.pi / 3.0 * r0]


def _integrate(E, kappa, eps, r_max, dense=False):
    r0 = 1.0e-7 if eps == 0.0 else 1.0e-6

    def rhs(r, s):
        y, yp, V, Vp = s
        return [yp, 2.0 * (V + u_bh(r, kappa, eps) - E) * y - 2.0 * yp / r, Vp, 4.0 * np.pi * y * y - 2.0 * Vp / r]

    def node(r, s):                                                         # Phi crosses zero: E too high
        return s[0]
    node.terminal, node.direction = True, -1

    def blowup(r, s):                                                       # Phi grows far above its central value: E too low
        return s[0] - 1.0e3
    blowup.terminal, blowup.direction = True, 1

    def turn(r, s):                                                         # Phi has decreased and turns upward: E too low
        return s[1] if s[0] < 0.5 else -1.0
    turn.terminal, turn.direction = True, 1

    sol = solve_ivp(rhs, [r0, r_max], _initial_state(E, kappa, eps, r0), method="DOP853", rtol=1e-12, atol=1e-14,
                    events=[node, blowup, turn], dense_output=dense)
    if sol.t_events[0].size:
        return +1, sol                                                      # node: raise E too high
    return -1, sol                                                          # no node: blow-up (E too low) or still decaying at r_max


def _find_bracket(grid, kappa, eps, r_max):
    last_low = None
    for E in grid:
        state, _ = _integrate(E, kappa, eps, r_max)
        if state < 0:
            last_low = E
        elif state > 0 and last_low is not None:
            return last_low, E
    return None


def solve(kappa, eps=0.0, r_max=60.0, n_scan=240, tol=1e-13, n_points=4096, y_cut=1e-8, E_guess=None):
    """Ground state for kappa = G M_BH / hbar_tilde^2 [1/kpc] and Gaussian width eps [kpc] (0 = point mass), Phi(0) = 1.
    E_guess (optional) narrows the first scan of the eigenvalue to a window around it (faster inside a root finder)."""
    bracket = None
    if E_guess is not None:
        w = 0.5 * max(1.0, abs(E_guess))
        bracket = _find_bracket(np.linspace(E_guess - w, E_guess + w, 40), kappa, eps, r_max)
    if bracket is None:
        bracket = _find_bracket(np.linspace(-0.6 * kappa ** 2 - 10.0, 40.0, n_scan), kappa, eps, r_max)
    if bracket is None:
        raise RuntimeError("no bracket for the ground state in the scanned energy range")
    lo, hi = bracket
    while hi - lo > tol * max(1.0, abs(hi)):
        mid = 0.5 * (lo + hi)
        state, _ = _integrate(mid, kappa, eps, r_max)
        if state < 0:
            lo = mid
        else:
            hi = mid
    E = 0.5 * (lo + hi)
    _, sol = _integrate(E, kappa, eps, r_max, dense=True)
    r_end = sol.t[-1]
    r = np.linspace(1.0e-6, r_end, n_points)
    y, yp, V, Vp = sol.sol(r)
    # keep the part where Phi is still a decaying solution: stop where Phi is minimal or smaller than y_cut
    keep = np.where(y < y_cut)[0]
    imin = int(np.argmin(np.where(np.arange(len(y)) > 0, np.abs(y), np.inf)))
    stop = keep[0] if keep.size else imin
    stop = min(stop, imin) if imin > 0 else stop
    r, y, V, Vp = r[:stop + 1], y[:stop + 1], V[:stop + 1], Vp[:stop + 1]
    y = np.clip(y, 0.0, None)
    mass_dimless = 4.0 * np.pi * np.trapz(y ** 2 * r ** 2, r)               # integral of 4 pi Phi^2 r^2
    c = V[-1] + mass_dimless / r[-1]                                         # V -> c - M~/r
    return dict(r=r, phi=y, U=V - c, omega=E - c, U0=-c, E=E, mass_dimless=mass_dimless, kappa=kappa, eps=eps)


def physical_mass(sol, hbar_t):
    """Total mass of the profile with Phi(0) = 1 [Msun]: (hbar_tilde^2 / G) 4 pi integral Phi^2 r^2 dr."""
    return hbar_t ** 2 / G * sol["mass_dimless"]


def rescale(sol, a):
    """Another central amplitude Phi(0) = a^2 of the same family: the solution with kappa_tilde = kappa / a rescaled to kappa."""
    return dict(r=sol["r"] / a, phi=a ** 2 * sol["phi"], U=a ** 2 * sol["U"], omega=a ** 2 * sol["omega"], U0=a ** 2 * sol["U0"], E=a ** 2 * sol["E"],
                mass_dimless=a * sol["mass_dimless"], kappa=a * sol["kappa"], eps=sol["eps"] / a)


def residual(sol, eps=None):
    """Largest relative residual of the radial equations on the profile (finite differences, interior points)."""
    r, y, U = sol["r"], sol["phi"], sol["U"]
    eps = sol["eps"] if eps is None else eps
    dr = r[1] - r[0]
    d1 = np.gradient(y, dr)
    d2 = np.gradient(d1, dr)
    lhs = 0.5 * (d2 + 2.0 * d1 / r)
    rhs = (U + u_bh(r, sol["kappa"], eps) - sol["omega"]) * y
    sel = slice(20, len(r) // 2)
    scale = np.max(np.abs(rhs[sel]))
    return float(np.max(np.abs(lhs[sel] - rhs[sel])) / scale)


def solve_physical(m_ev, m_bh, m_sol, eps=0.0, **kwargs):
    """Ground state of a soliton of total mass m_sol [Msun] around a central black hole of mass m_bh [Msun] (Gaussian width eps [kpc]),
    particle mass m_ev [eV]. Finds the central amplitude Phi(0) = a^2 by a root finder on the mass, using the scaling of rescale().
    Returns a dictionary with r [kpc], phi (rho = hbar_tilde^2 / G phi^2), U, omega, ... and the physical scalars:
    mass, rho_central [Msun/kpc^3], hbar_tilde, kappa."""
    from scipy.optimize import brentq

    ht = hbar_tilde(m_ev)
    kappa = G * m_bh / ht ** 2
    target = m_sol * G / ht ** 2                                    # 4 pi integral Phi^2 r^2 dr in the physical variables
    free = solve(0.0, **kwargs)
    a0 = target / free["mass_dimless"]                              # amplitude of the free soliton of this mass
    cache, state = {}, {"E": free["E"], "a": 1.0}

    def f(log_a):
        a = float(np.exp(log_a))
        s = solve(kappa / a, eps * a, E_guess=state["E"] * (a / state["a"]) ** 2 if state["a"] else None, **kwargs)
        state["E"], state["a"] = s["E"], a
        cache[log_a] = (a, s)
        return float(np.log(a * s["mass_dimless"] / target))

    log_a = brentq(f, np.log(a0 * 0.5), np.log(a0 * 60.0), xtol=1e-10)
    a, s = cache[log_a]
    out = rescale(s, a)
    out.update(mass=ht ** 2 / G * out["mass_dimless"], rho_central=ht ** 2 / G * out["phi"][0] ** 2, hbar_tilde=ht, a=a, mass_bh=m_bh)
    return out


def write_profile(path, sol):
    """Write the profile in the format of GroundState(1).dat / the Modo*.dat files ("#U0 omega", "#r phi U", then r phi U), with r in kpc and
    phi in the convention of the code (rho = hbar_tilde^2 / G phi^2). The code rescales a file by desired_soliton_mass / (mass of the file):
    pass desired_soliton_mass equal to sol["mass"] (a factor different from 1 is exact for a free soliton but NOT with a black hole)."""
    with open(path, "w") as f:
        f.write(f"#{sol['U0']!r} {sol['omega']!r}\n#r phi U\n")
        for i, (r, y, u) in enumerate(zip(sol["r"], sol["phi"], sol["U"])):
            f.write(f"{0.0 if i == 0 else r!r} {y!r} {u!r}\n")        # the grid node at r = 0 needs a value: the code gives 0 outside the table


def selftest(path="resources/solitons/GroundState(1).dat"):
    ref = np.loadtxt(path)
    with open(path) as f:
        U0_ref, omega_ref = (float(x) for x in f.readline()[1:].split())
    s = solve(0.0)
    phi_ref = np.interp(s["r"], ref[:, 0], ref[:, 1])
    inner = s["phi"] > 1e-3
    print(f"free soliton: U0 {s['U0']:.6f} (file {U0_ref:.6f}), omega {s['omega']:.6f} (file {omega_ref:.6f}), "
          f"max |Phi - Phi_file| {np.max(np.abs(s['phi'][inner] - phi_ref[inner])):.1e}, residual {residual(s):.1e}")
    ht = hbar_tilde(1e-21)
    kappa = G * 1.0e8 / ht ** 2                                     # hydrogen-like limit Phi = exp(-kappa r): M = pi hbar_tilde^2 / (G kappa^3)
    d = solve(kappa, 0.0, r_max=3.0)
    print(f"hydrogen limit (M_BH = 1e8, m = 1e-21 eV): mass {physical_mass(d, ht):.4e}, analytic {np.pi * ht ** 2 / (G * kappa ** 3):.4e}")
    sol = solve_physical(1e-21, 4.0e6, 5.57e7, 0.005)
    print(f"m = 1e-21 eV, M_sol = 5.57e7, M_BH = 4e6, eps = 5 pc: mass {sol['mass']:.4e}, central density {sol['rho_central']:.4e} Msun/kpc^3, "
          f"a = {sol['a']:.3f}, residual {residual(sol):.1e}")


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Ground state of the radial Schroedinger-Poisson system with a central black hole (units kpc, Msun, Gyr).")
    p.add_argument("--selftest", action="store_true", help="check against GroundState(1).dat, the hydrogen limit and a physical example")
    p.add_argument("--m", type=float, default=1e-21, help="ULDM particle mass [eV]")
    p.add_argument("--M_sol", type=float, default=5.57e7, help="total mass of the soliton [Msun]")
    p.add_argument("--M_bh", type=float, default=0.0, help="black hole mass [Msun] (0: free soliton)")
    p.add_argument("--eps", type=float, default=0.0, help="Gaussian width of the black hole [kpc] (0: point mass); use the softening_bh of the run")
    p.add_argument("--out", default=None, help="output file; default resources/solitons/Dressed_<m>_<M_sol>_<M_bh>_<eps>.dat (next to GroundState(1).dat)")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        sol = solve_physical(args.m, args.M_bh, args.M_sol, args.eps)
        ht = sol["hbar_tilde"]
        print(f"particle mass {args.m:.3e} eV  (hbar/m = {ht:.5f} kpc^2/Gyr), black hole {args.M_bh:.4e} Msun, Gaussian width {args.eps} kpc")
        print(f"soliton mass          : {sol['mass']:.8e} Msun   (asked {args.M_sol:.8e})")
        print(f"central density       : {sol['rho_central']:.6e} Msun/kpc^3")
        print(f"eigenvalue omega, U0  : {sol['omega']:.6f}, {sol['U0']:.6f}")
        print(f"residual of the ODE   : {residual(sol):.1e}   (should be below ~1e-4)")
        print(f"profile extends to    : r = {sol['r'][-1]:.4f} kpc, {len(sol['r'])} points")
        out = args.out
        if out is None:
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "solitons",
                               f"Dressed_m{args.m:.0e}_Msol{args.M_sol:.3e}_Mbh{args.M_bh:.2e}_eps{args.eps}.dat")
        write_profile(out, sol)
        print("profile written to", out)
        print("use it with:  Wave_vector_class(packet_type=<that path>, ..., desired_soliton_mass=" + repr(sol["mass"]) + ")")