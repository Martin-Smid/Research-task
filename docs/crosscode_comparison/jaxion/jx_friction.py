"""Dynamical friction of a moving gravitating point in a uniform ULDM background, using jaxion (unmodified).

Units are the native jaxion ones (kpc, km/s, Msun; time unit kpc/(km/s) = 0.977792 Gyr); results are converted to
kpc/Gyr at the end. Three ways of representing the moving mass:
  native : jaxion's own particle (particles.num_particles = 1), CIC deposit + CIC-interpolated spectral force (point
           receiver, effective softening ~ one cell, no eps parameter).
  point  : Gaussian-softened mass (exp(-k^2 eps^2/2) in k space) added through Simulation.custom_density; psi feels the
           Gaussian, the mass feels the force at its centre (exact trigonometric interpolation of -grad V).
  smooth : same Gaussian mass, but the mass feels the force averaged with its own Gaussian (action = reaction).
Only the public hooks custom_density / custom_kick / custom_drift / callback are used; the library is not edited.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # shared GPU: allocate on demand, report the real peak

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import jaxdecomp as jd  # noqa: E402
import jaxion  # noqa: E402
from jaxion.constants import constants  # noqa: E402

GYR_PER_TU = 3.0856775814913673e16 / 3.15576e16        # (kpc / (km/s)) in Gyr
BOX = 9.6                                               # box is [-BOX, BOX]^3
V_KPC_GYR = 30.0
M_PARTICLE_EV = 2.5e-22
RHO_M_MAX = 3.0e12
RHO_MAX = 3.0e5
REC_GYR = 1.6e-3


def run_case(mq, n, eps, mode, safety, rho0_override=None, out_dir="out", tag=""):
    G = constants["gravitational_constant"]
    L = 2.0 * BOX
    v0 = V_KPC_GYR * GYR_PER_TU                          # km/s
    t_end = 0.5 * L / v0                                 # half a box of travel (TU)
    params = {
        "physics": {"particles": mode == "native"},
        "domain": {"box_size": L, "resolution_multiplier": n // 32},
        "time": {"end": t_end, "safety_factor": safety},
        "output": {"save": False, "num_checkpoints": 1},
        "quantum": {"m_22": M_PARTICLE_EV / 1e-22},
    }
    # particle mass is only known after hbar/m, which needs a Simulation: build once with a dummy mass
    if mode == "native":
        params["particles"] = {"num_particles": 1, "particle_mass": 1.0}
    sim = jaxion.Simulation(params)
    hbar_t = 1.0 / sim.m_per_hbar                        # hbar/m [kpc km/s]
    mass = hbar_t * v0 / (G * mq)
    rho0 = min(RHO_MAX, RHO_M_MAX / mass) if rho0_override is None else rho0_override
    if mode == "native":
        params["particles"]["particle_mass"] = mass
        sim = jaxion.Simulation(params)
    dx, nt = sim.dx, sim.nt
    dt = t_end / nt
    lam = hbar_t / v0
    f_rel = 4.0 * np.pi * rho0 * (G * mass / v0) ** 2
    print(f"## mq={mq} N={n} eps={eps} mode={mode}: M={mass:.4e} rho0={rho0:.4e} dx={dx:.4f} lambda={lam:.4f} "
          f"hbar/m={hbar_t / GYR_PER_TU:.4f} kpc^2/Gyr G={G * GYR_PER_TU**-2:.5e} kpc^3/(Msun Gyr^2) "
          f"nt={nt} dt={dt * GYR_PER_TU:.3e} Gyr", flush=True)

    kx, ky, kz = sim.kgrid
    k_sq = kx**2 + ky**2 + kz**2
    gauss = jnp.exp(-0.5 * k_sq * eps**2)
    n_cells = n**3
    dV = dx**3
    vol = L**3
    rec = max(1, round(REC_GYR / GYR_PER_TU / dt))

    x0 = 0.5 * L - 0.25 * L                               # starts 0.25 box before the centre, ends 0.25 box after
    y0 = 0.5 * L
    pos0 = jnp.array([x0, y0, y0])

    sim.state["psi"] = jnp.full(sim.shape, np.sqrt(rho0), dtype=jnp.complex128)
    state_extra = {
        "vlog": jnp.zeros(nt + 1).at[0].set(v0),
        "plog": jnp.zeros(nt // rec + 1),
    }
    if mode == "native":
        sim.state["pos"] = pos0[None, :]
        sim.state["vel"] = jnp.array([[v0, 0.0, 0.0]])
    else:
        state_extra["bh_pos"] = pos0
        state_extra["bh_vel"] = jnp.array([v0, 0.0, 0.0])
    sim.state.update(state_extra)

    def phase(pos):
        # e^{-i k.(X - dx/2)}: the grid points are x_j = (j + 1/2) dx
        s = pos - 0.5 * dx
        return jnp.exp(-1j * (kx * s[0] + ky * s[1] + kz * s[2]))

    if mode != "native":
        def custom_density(state):
            rho_hat = (mass / vol) * n_cells * gauss * phase(state["bh_pos"])
            return jnp.real(jd.fft.pifft3d(rho_hat))

        def custom_kick(state, V, dt_):
            a_hat = -1j * jd.fft.pfft3d(V)
            if mode == "point":
                ph = jnp.conj(phase(state["bh_pos"]))   # e^{+i k.(X - dx/2)}
                acc = jnp.stack([jnp.real(jnp.sum(a_hat * k * ph)) / n_cells * 1.0 for k in (kx, ky, kz)])
            else:
                rho_g = jnp.real(jd.fft.pifft3d((1.0 / vol) * n_cells * gauss * phase(state["bh_pos"])))  # rho_G / M
                acc = jnp.stack([jnp.sum(rho_g * jnp.real(jd.fft.pifft3d(a_hat * k))) * dV
                                 for k in (kx, ky, kz)])
            state["bh_vel"] = state["bh_vel"] + acc * dt_
            return state

        def custom_drift(state, k_sq_, dt_):
            state["bh_pos"] = jnp.mod(state["bh_pos"] + state["bh_vel"] * dt_, L)
            return state

        sim.custom_density = custom_density
        sim.custom_kick = custom_kick
        sim.custom_drift = custom_drift

    def psi_px(psi):
        dpsi = jd.fft.pifft3d(1j * kx * jd.fft.pfft3d(psi))
        return hbar_t * jnp.sum(jnp.imag(jnp.conj(psi) * dpsi)) * dV

    def callback(i, state):
        vx = state["vel"][0, 0] if mode == "native" else state["bh_vel"][0]
        state["vlog"] = state["vlog"].at[i + 1].set(vx)
        j = (i + 1) // rec
        state["plog"] = jax.lax.cond(
            (i + 1) % rec == 0,
            lambda p: p.at[j].set(psi_px(state["psi"])),
            lambda p: p,
            state["plog"],
        )

    sim.callback = callback
    t_wall = time.time()
    sim.run()
    wall = time.time() - t_wall
    try:
        peak_gb = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30
    except Exception:
        peak_gb = float("nan")
    print(f"   run time {wall:.1f} s  gpu peak memory {peak_gb:.2f} GiB", flush=True)

    vlog = np.asarray(sim.state["vlog"]) / GYR_PER_TU     # kpc/Gyr
    plog = np.asarray(sim.state["plog"])
    p0 = 0.0
    t = np.arange(nt + 1) * dt * GYR_PER_TU
    # drag coefficient: central differences over +-rec steps, sampled every rec steps (as in the reference scan)
    idx = np.arange(rec, nt - rec + 1, rec)
    mass_g = mass                                          # Msun
    # F = M dv/dt in Msun kpc/Gyr^2 ; F_rel in the same units: rebuild in kpc/Gyr units
    G_g = G * GYR_PER_TU**-2 * (1.0)                       # kpc^3/(Msun Gyr^2)  (km/s -> kpc/Gyr conversion squared)
    f_rel_g = 4.0 * np.pi * rho0 * (G_g * mass_g / V_KPC_GYR) ** 2
    c_t = -mass_g * (vlog[idx + rec] - vlog[idx - rec]) / (2.0 * rec * dt * GYR_PER_TU) / f_rel_g
    lam_kpc = lam
    b = V_KPC_GYR * t[idx] / lam_kpc                       # v t / lambda
    win = b > 7.0
    t_p = t[::rec][: len(plog)]
    pbh = mass_g * (V_KPC_GYR - vlog[::rec][: len(plog)])  # momentum lost by the point (kpc Msun / Gyr)
    pps = plog / GYR_PER_TU                                # km/s -> kpc/Gyr
    with np.errstate(divide="ignore", invalid="ignore"):
        bal = np.where(pbh > 1e-9 * mass_g * V_KPC_GYR, pps / pbh - 1.0, np.nan)
    late = t_p >= 0.5 * t_p[-1]
    res = dict(mq=mq, n=n, eps=eps, mode=mode, mass=mass, rho0=rho0, lam=lam_kpc, dx=dx, nt=nt, wall=wall, peak_gb=peak_gb, dt_gyr=dt * GYR_PER_TU,
               t=t_p, t_c=t[idx], c=c_t, b=b, vlog_end=vlog[-1], slowdown=1 - vlog[-1] / V_KPC_GYR,
               c_mean=float(np.mean(c_t[win])) if rho0 > 0 else np.nan,
               p_psi=pps, p_lost=pbh, bal=bal, bal_late=float(np.nanmean(bal[late])) if rho0 > 0 else np.nan,
               dv_total=vlog[-1] - V_KPC_GYR)
    print(f"   slowdown {100 * res['slowdown']:.3f} %  <C>(b>7) = {res['c_mean']:.4f}  P_psi/P_lost-1 (late) = "
          f"{100 * res['bal_late']:+.2f} %  dv = {res['dv_total']:.5f} kpc/Gyr", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"jx_{mode}_MQ{mq:g}_N{n}_eps{eps:g}{tag}.npz"), **res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mq", default="17.4,1,0.3")
    ap.add_argument("--n", default="128")
    ap.add_argument("--eps", default="0.075")
    ap.add_argument("--mode", default="point")
    ap.add_argument("--safety", type=float, default=0.5)
    ap.add_argument("--rho0", type=float, default=None, help="override background density (0 = self-force test)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    print("devices:", jax.devices(), flush=True)
    for mode in a.mode.split(","):
        for n in map(int, a.n.split(",")):
            for eps in map(float, a.eps.split(",")):
                for mq in map(float, a.mq.split(",")):
                    run_case(mq, n, eps, mode, a.safety, a.rho0, a.out, a.tag)
