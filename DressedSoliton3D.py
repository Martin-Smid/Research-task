"""Ground state of the 3D Schroedinger-Poisson system on the simulation grid, with an optional black hole held at a fixed position.

Imaginary-time Strang splitting with the same operators as the real-time code (Propagator_Class): kinetic factor exp(-tau hbar_tilde k^2 / 2),
potential factor exp(-tau V / hbar_tilde), V = periodic Poisson potential of psi (mean density subtracted) + analytic potential of the Gaussian black
hole (the sink potential, width eps, mean-density term removed), mass renormalised to m_sol after every step. The result is the lowest state of this
periodic box, so it is the exact stationary initial condition for the grid scheme (no isolated-space assumption, any black hole position).

pin_centroid = (x, y, z): the centroid of psi is moved back to this point after every step. Without it the ground state with the black hole at an
offset simply sits on the black hole; with it you get the soliton held at the pinned point and deformed by the tide of the black hole (a constrained
state, not stationary: releasing it makes soliton and black hole fall towards each other).

Units kpc, Msun, Gyr. Typical use (GPU, cupy):

    from DressedSoliton3D import make_sim, relax, save, load_into
    sim = make_sim(N=192, half_box=1.0, m_ev=1e-21)
    res = relax(sim, m_sol=5.57e7, m_bh=4e6, bh_position=(0, 0, 0), eps=0.005)
    save("dressed.npz", sim, res)
    ... later, after sim.initialize_simulation():  load_into(sim.wave_functions[0], "dressed.npz")
"""

import numpy as np
import cupy as cp


def make_sim(N, half_box, m_ev, dim=3):
    """A Simulation_Class used only as the provider of the grid, k-space and the constants G, hbar_tilde (nothing is written to disk)."""
    from resources.Classes.Simulation_Class import Simulation_Class

    return Simulation_Class(dim=dim, boundaries=[(-half_box, half_box)] * dim, N=N, total_time=1.0e-3, h=1.0e-3, order_of_evolution=2,
                            use_gravity=True, use_units=True, use_sponge=False, self_int=False, m_s=m_ev)


def _centroid(rho, sim, target):
    """Centroid of rho relative to target with the minimum-image convention [kpc]."""
    out = []
    total = cp.sum(rho)
    for g, (lo, hi), t in zip(sim.grids, sim.boundaries, target):
        length = hi - lo
        rel = (cp.asarray(g) - t + 0.5 * length) % length - 0.5 * length
        out.append(float(cp.sum(rho * rel) / total))
    return out


def relax(sim, m_sol, m_bh=0.0, bh_position=(0.0, 0.0, 0.0), eps=0.0, psi0=None, tau=None, tol=1e-9, max_steps=60000,
          check_every=100, pin_centroid=None, residual_target=3.0e-4, tau_factor=4.0, max_levels=5, verbose=True):
    """Imaginary-time relaxation to the ground state of total mass m_sol [Msun]. Returns a dictionary with psi (cupy complex128, |psi|^2 = density),
    mu (the chemical potential <H> in 1/Gyr), residual (||(H - mu) psi|| / (|mu| ||psi||)), steps and the parameters."""
    G, hb, dV = float(sim.G), float(sim.h_bar_tilde), float(sim.dV)
    kx, ky, kz = (cp.asarray(k) for k in sim.k_space)
    k2 = kx * kx + ky * ky + kz * kz
    zero = k2 == 0
    k2_safe = cp.where(zero, 1.0, k2)
    lo = [float(b[0]) for b in sim.boundaries]

    v_bh = 0.0
    if m_bh > 0.0:
        phase = cp.exp(-1j * (kx * (bh_position[0] - lo[0]) + ky * (bh_position[1] - lo[1]) + kz * (bh_position[2] - lo[2])))
        phi_k = (-4.0 * np.pi * G * (m_bh / dV)) * phase * cp.exp(-0.5 * k2 * eps ** 2) / k2_safe
        phi_k[zero] = 0.0
        v_bh = cp.fft.ifftn(phi_k).real

    def potential(psi):
        rho = (psi.real ** 2 + psi.imag ** 2)
        rho_k = cp.fft.fftn(rho - cp.mean(rho))
        phi_k = -4.0 * np.pi * G * rho_k / k2_safe
        phi_k[zero] = 0.0
        return cp.fft.ifftn(phi_k).real + v_bh

    def normalise(psi):
        return psi * np.sqrt(m_sol / (float(cp.sum(psi.real ** 2 + psi.imag ** 2)) * dV))

    # length scale of the state: the smaller of the soliton size and the Bohr radius of the black hole
    r_half = 4.0 * hb ** 2 / (G * m_sol)
    length = max(min(r_half, hb ** 2 / (G * m_bh)) if m_bh > 0 else r_half, 2.0 * min(float(d) for d in sim.dx))
    if tau is None:
        tau = 0.05 * length ** 2 / hb                               # a fraction of the inverse level spacing hbar / length^2

    if psi0 is None:
        r2 = sum((cp.asarray(g) - c) ** 2 for g, c in zip(sim.grids, bh_position if m_bh > 0 else (0.0, 0.0, 0.0)))
        psi = cp.exp(-r2 / (4.0 * (0.5 * length) ** 2)).astype(cp.complex128)
    else:
        psi = cp.asarray(psi0, dtype=cp.complex128)
    psi = normalise(psi)

    # The fixed point of the split step is an eigenstate of a slightly modified Hamiltonian (error ~ tau^1.3...2), so the time step is reduced by
    # `tau_factor` after every level until the state stops changing (mu converged) and the residual of the exact discrete H is below residual_target.
    mu_old, mu, residual, step = None, float("nan"), np.inf, 0
    for level in range(max_levels):
        kinetic = cp.exp(-tau * hb * k2 / 2.0)
        mu_old = None
        while step < max_steps:
            step += 1
            v = potential(psi)
            psi = psi * cp.exp(-tau * v / (2.0 * hb))
            psi = cp.fft.ifftn(kinetic * cp.fft.fftn(psi))
            v = potential(psi)
            psi = psi * cp.exp(-tau * v / (2.0 * hb))
            psi = normalise(psi)
            if pin_centroid is not None:
                shift = _centroid(psi.real ** 2 + psi.imag ** 2, sim, pin_centroid)
                if any(abs(s) > 1e-12 for s in shift):
                    phase_shift = cp.exp(1j * (kx * shift[0] + ky * shift[1] + kz * shift[2]))      # translate psi by -shift
                    psi = cp.fft.ifftn(phase_shift * cp.fft.fftn(psi))
            if step % check_every == 0:
                v = potential(psi)
                h_psi = cp.fft.ifftn(hb * k2 / 2.0 * cp.fft.fftn(psi)) + v / hb * psi
                mu = float(cp.sum(cp.conj(psi) * h_psi).real * dV / m_sol)
                residual = float(cp.sqrt(cp.sum(cp.abs(h_psi - mu * psi) ** 2) * dV / m_sol)) / abs(mu)
                if verbose:
                    print(f"level {level} tau {tau:.2e} step {step:6d}: mu = {mu:.10e}, relative residual {residual:.2e}", flush=True)
                if mu_old is not None and abs(mu - mu_old) < tol * abs(mu):
                    break
                mu_old = mu
        if residual < residual_target or step >= max_steps:
            break
        tau /= tau_factor
    return dict(psi=psi, mu=mu, residual=residual, steps=step, tau=tau, m_sol=m_sol, m_bh=m_bh,
                bh_position=tuple(bh_position), eps=eps, N=sim.N, boundaries=[tuple(b) for b in sim.boundaries], m_ev=None)


def radial_profile(rho, sim, center=(0.0, 0.0, 0.0), n_bins=60, r_max=None):
    """Spherically averaged density rho(r) around center [kpc]: returns (r_centres, rho_mean)."""
    r2 = sum((cp.asarray(g) - c) ** 2 for g, c in zip(sim.grids, center))
    r = cp.sqrt(r2).ravel()
    r_max = float(r_max if r_max is not None else sim.boundaries[0][1])
    edges = cp.linspace(0.0, r_max, n_bins + 1)
    idx = cp.clip(cp.searchsorted(edges, r, side="right") - 1, 0, n_bins - 1)
    sums = cp.bincount(idx, weights=rho.ravel(), minlength=n_bins)
    counts = cp.bincount(idx, minlength=n_bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return cp.asnumpy(centres), cp.asnumpy(sums / cp.maximum(counts, 1))


def save(path, sim, result):
    np.savez_compressed(path, psi=cp.asnumpy(result["psi"]), N=sim.N, boundaries=np.array(result["boundaries"]), mu=result["mu"],
                        residual=result["residual"], m_sol=result["m_sol"], m_bh=result["m_bh"], bh_position=np.array(result["bh_position"]),
                        eps=result["eps"], h_bar_tilde=float(sim.h_bar_tilde))


def load_into(wave_function, path):
    """Replace psi of an initialised Wave_function by the saved ground state (the grid must be the same)."""
    d = np.load(path)
    psi = cp.asarray(d["psi"])
    if psi.shape != wave_function.psi.shape:
        raise ValueError(f"grid mismatch: file {psi.shape}, simulation {wave_function.psi.shape}")
    wave_function.psi = psi
    return dict(m_sol=float(d["m_sol"]), m_bh=float(d["m_bh"]), bh_position=tuple(d["bh_position"]), eps=float(d["eps"]))


if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="3D imaginary-time ground state with an optional fixed black hole (units kpc, Msun, Gyr).")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--half_box", type=float, default=1.0, help="the box is [-half_box, half_box]^3 [kpc]")
    p.add_argument("--m", type=float, default=1e-21, help="particle mass [eV]")
    p.add_argument("--M_sol", type=float, default=5.57e7)
    p.add_argument("--M_bh", type=float, default=0.0)
    p.add_argument("--bh_pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--eps", type=float, default=0.0, help="Gaussian width of the black hole [kpc] (use softening_bh of the run)")
    p.add_argument("--pin", type=float, nargs=3, default=None, help="pin the centroid of psi at this point")
    p.add_argument("--out", default=None, help="output file; default resources/solitons/Dressed3D_<N>_<box>_<M_sol>_<M_bh>_<eps>.npz")
    a = p.parse_args()
    sim = make_sim(a.N, a.half_box, a.m)
    res = relax(sim, a.M_sol, a.M_bh, tuple(a.bh_pos), a.eps, pin_centroid=tuple(a.pin) if a.pin else None)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "solitons",
                                f"Dressed3D_N{a.N}_box{a.half_box}_Msol{a.M_sol:.3e}_Mbh{a.M_bh:.2e}_eps{a.eps}.npz")
    save(out, sim, res)
    print(f"saved {out}: mu {res['mu']:.8e}, residual {res['residual']:.1e}, steps {res['steps']}")
    print("use it after sim.initialize_simulation():  DressedSoliton3D.load_into(sim.wave_functions[0], '" + out + "')")
