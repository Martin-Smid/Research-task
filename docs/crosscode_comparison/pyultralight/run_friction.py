"""Test 1 (dynamical friction of a moving point mass in a uniform periodic ULDM background) with PyUL_NBody.

Runs one case and writes <out>/result_<tag>.npz. PyUL_NBody itself is not modified; this driver only builds the config
(GenerateConfig), calls evolve() and post-processes its outputs.

Dimensionless mode (default): code units with hbar/m = 1, G = 1 (the units PyUL uses internally: kinetic phase k^2/2, Poisson
4 pi rho), length unit = lambda = hbar/(m v), velocity unit = v. M_Q, eps/lambda, L/lambda, rho are then exactly the requested ones.
Physical mode (--phys): kpc, Msun, Gyr / km/s inputs converted by PyUL's own constants (tiny constant mismatches, reported).
"""
import argparse, io, json, multiprocessing, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.expanduser("~/projects/pyultralight/nbody"))
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--mq", type=float, required=True)           # M_Q = hbar v / (G M)
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--eps_kpc", type=float, default=None)       # Plummer radius [kpc]
ap.add_argument("--eps_cells", type=float, default=None)     # Plummer radius [cells]
ap.add_argument("--H", type=float, default=None)             # time step [Gyr]; default: PyUL's own dt = dx^2/pi (step_factor = 1)
ap.add_argument("--x0_frac", type=float, default=-0.25)      # start x / L (-0.25 -> -4.8 kpc, as in the reference runs)
ap.add_argument("--travel", type=float, default=0.5)         # travel distance / L
ap.add_argument("--momentum", type=int, default=1)           # save psi momentum every step (slow)
ap.add_argument("--energy", type=int, default=1)
ap.add_argument("--phys", type=int, default=0)
ap.add_argument("--post_only", type=int, default=0)
ap.add_argument("--rho_override", type=float, default=None)  # Msun/kpc^3
a = ap.parse_args()
a.out = os.path.abspath(os.path.expanduser(a.out))

nthr = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
multiprocessing.cpu_count = lambda: nthr                    # PyUL uses all node cores otherwise
sys.stdin = io.StringIO("2.5e-22" + chr(10))                         # the library asks for the axion mass at import
sys.path.insert(0, a.repo)
os.chdir(a.repo)
import numpy as np
import PyUltraLight2 as P

# ---- user's physical parameters
V = 30.0; BOX = 9.6; L = 2 * BOX; T = 0.32
G_USER = 4.4985e-6; HBAR_USER = 7.844
lam = HBAR_USER / V                                          # 0.26147 kpc
rho_user = {17.4: 3.0e5, 1.0: 5.7e4, 0.3: 1.7e4}[a.mq]
M_user = HBAR_USER * V / (G_USER * a.mq)
if a.mq == 17.4: M_user = 3.0e6
if a.mq == 1.0: M_user = 5.23e7
if a.mq == 0.3: M_user = 1.744e8
if a.rho_override: rho_user = a.rho_override
dx = L / a.N
eps_kpc = a.eps_kpc if a.eps_kpc is not None else a.eps_cells * dx
Tt = a.travel * L / V                                        # run time (0.32 Gyr for travel = 0.5)
x0 = a.x0_frac * L

# code-unit values (hbar/m = G = 1, length unit = lam, velocity unit = V)
ell = lam
Mc = G_USER * M_user / (ell * V ** 2)
rhoc = rho_user * G_USER * ell ** 2 / V ** 2
tc = lambda t_gyr: t_gyr * V / ell
MQ_eff = 1.0 / Mc                                            # = hbar v / (G M) in code units (v_c = 1)

dt_default = (L / ell / a.N) ** 2 / np.pi                    # PyUL's own step (code units)
if a.H is None:
    S = int(tc(Tt) / dt_default + 1)
else:
    S = int(round(tc(Tt) / (a.H * V / ell)))
    S = max(S, int(tc(Tt) / dt_default + 1))                 # PyUL only accepts save_number >= its own minimum step count
h_code = tc(Tt) / S
H_gyr = h_code * ell / V

so = "NBody DF" + (" Energy" if a.energy else "") + (" Momentum" if a.momentum else "")
os.makedirs(a.out, exist_ok=True)
name = a.tag
if a.phys:
    # physical mode: kpc, Msun, Gyr (velocity in km/s: PyUL has no kpc/Gyr)
    kpc_m = P.parsec * 1e3
    gyr_s = 365 * 24 * 3600 * 1e9
    v_kms = V * kpc_m / gyr_s / 1e3
    rP_code = P.GenPlummer(eps_kpc, "kpc")
    G_eff = P.G * P.solar_mass * gyr_s ** 2 / kpc_m ** 3
    hbar_eff = P.hbar / P.axion_mass * gyr_s / kpc_m ** 2
    print(f"phys mode: G_eff = {G_eff:.6e} kpc^3/Msun/Gyr^2 (user {G_USER}), hbar/m_eff = {hbar_eff:.5f} kpc^2/Gyr (user {HBAR_USER})")
    print(f"M_Q (PyUL constants) = {hbar_eff * V / (G_eff * M_user):.5f}")
    cfg = dict(NS=4, length=L, length_units="kpc", resol=a.N, duration=Tt, duration_units="Gyr", step_factor=1.0, save_number=S,
               save_options=so, save_path=a.out, save_format="npy", s_mass_unit="solar_masses", s_position_unit="kpc",
               s_velocity_unit="km/s", solitons=[], start_time=0, m_mass_unit="solar_masses", m_position_unit="kpc",
               m_velocity_unit="km/s", particles=[[M_user, [x0, 0.0, 0.0], [v_kms, 0.0, 0.0]]], embeds=[], Uniform=True,
               Density=rho_user * 1e-9, density_unit="MSol/pc3", a=rP_code, UVel=[0, 0, 0], NoInteraction=True, Name=name)
else:
    L_c = L / ell
    cfg = dict(NS=4, length=L_c, length_units="", resol=a.N, duration=tc(Tt), duration_units="", step_factor=1.0, save_number=S,
               save_options=so, save_path=a.out, save_format="npy", s_mass_unit="", s_position_unit="", s_velocity_unit="",
               solitons=[], start_time=0, m_mass_unit="", m_position_unit="", m_velocity_unit="",
               particles=[[Mc, [x0 / ell, 0.0, 0.0], [1.0, 0.0, 0.0]]], embeds=[], Uniform=True, Density=rhoc, density_unit="",
               a=ell / eps_kpc, UVel=[0, 0, 0], NoInteraction=True, Name=name)
print(f"case M_Q={a.mq} N={a.N} dx={dx:.4f} kpc eps={eps_kpc:.5f} kpc = {eps_kpc/dx:.3f} cells = {eps_kpc/lam:.3f} lambda; "
      f"steps={S}, H={H_gyr:.3e} Gyr; code: M={Mc:.5g} rho={rhoc:.5g} M_Q_eff={MQ_eff:.5f}; threads={nthr}", flush=True)
if not a.post_only:
    P.GenerateConfig(**cfg)
loc = os.path.join(a.out, name)
t0 = time.time()
if not a.post_only:
    P.evolve(a.out, name, DumpFinal=True)
wall = time.time() - t0

# ---------------------------------------------------------------- post-processing (code units; C is dimensionless)
out = os.path.join(loc, "Outputs")
S_eff = len([f for f in os.listdir(os.path.join(out, "NBody")) if f.endswith(".npy")]) - 1
state = np.array([np.load(os.path.join(out, "NBody", f"NTM_#{k:03d}.npy")) for k in range(0, S_eff + 1)])
acc = np.array([np.load(os.path.join(out, "DF", f"DYF_#{k:03d}.npy")) for k in range(0, S_eff + 1)])
# NBody_#0 is the initial state; #k is the state after step k
Mp = Mc if not a.phys else None
if a.phys:
    Mp = float(P.convert(M_user, "solar_masses", "m"))
vx = state[:, 3]; xx = state[:, 0]
time_c = np.arange(S_eff + 1) * ((tc(Tt) if not a.phys else float(P.convert(Tt, "Gyr", "t"))) / S_eff)
rho_c = rhoc if not a.phys else float(P.convert(rho_user * 1e-9, "MSol/pc3", "d"))
v0 = vx[0]
Frel = 4 * np.pi * rho_c * (Mp / v0) ** 2
# C(t): same estimator as the reference (central difference over +-1.6e-3 Gyr, sampled every 1.6e-3 Gyr)
h_gyr_eff = Tt / S_eff
rec = max(1, round(1.6e-3 / h_gyr_eff))
idx = np.arange(rec, S_eff - rec + 1, rec)
tt_gyr = idx * h_gyr_eff
c_sim = -Mp * (vx[idx + rec] - vx[idx - rec]) / (2 * rec * time_c[1]) / Frel
b = V * tt_gyr / lam
win = b >= 7.0
# alternative: instantaneous force from DF (acceleration at the end-of-step RK stage)
c_df = -Mp * acc[1:, 0] / Frel
tt_df = np.arange(1, S_eff + 1) * h_gyr_eff
win_df = V * tt_df / lam >= 7.0
res = dict(mq=a.mq, N=a.N, eps_kpc=eps_kpc, eps_cells=eps_kpc / dx, eps_lambda=eps_kpc / lam, H_gyr=h_gyr_eff, steps=S_eff,
           t=tt_gyr, b=b, c=c_sim, mean_c=float(np.mean(c_sim[win])), mean_c_df=float(np.mean(c_df[win_df])),
           slowdown=float(1 - vx[-1] / v0), vx=vx, x=xx, t_all=tt_df, wall_s=wall, M_Q_eff=MQ_eff, phys=a.phys)
# momentum bookkeeping
if a.momentum:
    mom = np.array([np.load(os.path.join(out, "Momentum", f"PMT_#{k:03d}.npy")) for k in range(0, S_eff + 1)])
    # MVR_#(k+1) is psi after the kick at the time of NBody_#k (see notes)
    p_psi = mom[1:, 0]
    lost = Mp * (v0 - vx[:-1])
    n = len(p_psi)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(lost > 1e-9 * Mp * v0, p_psi / lost - 1.0, np.nan)
    res.update(p_psi=p_psi, p_lost=lost, p_err=err, p_err_late=float(np.nanmean(err[n // 2:])), p_err_end=float(err[-1]))
# independent end-state momentum from the dumped psi: Im(psi* d psi/dx) dV, spectral derivative
psiF = np.load(os.path.join(loc, "Final_psi.npy"))
Lc = L / ell if not a.phys else float(P.convert(L, "kpc", "l"))
k = 2 * np.pi * np.fft.fftfreq(a.N, Lc / a.N)
dpsi = np.fft.ifftn(1j * k[:, None, None] * np.fft.fftn(psiF))
p_end = float(np.sum(np.imag(np.conj(psiF) * dpsi)) * (Lc / a.N) ** 3)
vF = float(np.load(os.path.join(loc, "Final_TM.npy"))[3])
res.update(p_end_indep=p_end, v_final=vF, momentum_sum_end_rel=float((p_end + Mp * (vF - v0)) / (Mp * v0)),
           p_end_over_lost=float(p_end / (Mp * (v0 - vF))) if v0 != vF else np.nan)
if a.energy:
    res["energy_tot"] = np.load(os.path.join(out, "egylist.npy"))
np.savez(os.path.join(a.out, f"result_{a.tag}.npz"), **res)
print(f"RESULT {a.tag}: mean C(b>7) = {res['mean_c']:.4f} (from DF accel: {res['mean_c_df']:.4f}); slowdown {100*res['slowdown']:.2f} %; "
      f"p_end/lost-1 = {res['p_end_over_lost']-1:+.4f}; (p_psi+M dv)/Mv0 = {res['momentum_sum_end_rel']:+.3e}; wall {wall:.0f} s", flush=True)
