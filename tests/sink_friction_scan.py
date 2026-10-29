from pathlib import Path
import os
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import sici

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody


# =====================================================================================================
# SCANS OF THE DYNAMICAL FRICTION OF A SINK IN A UNIFORM ULDM BACKGROUND (units: kpc, Msun, Gyr)
#
# Runs the set-up of sink_friction.py (sink moving at V_SINK through uniform psi, half a box of travel) for a list of cases.
# Two scans, chosen by SCAN:
#   "mq"        T1: quantum Mach number M_Q = hbar v / (G M) from 100 to 0.3 (the sink mass changes, the speed is fixed, the
#               background density is lowered for heavy sinks so that the sink slows down by only a few per cent).
#               Compared with the exact linear theory for this box (valid for M_Q >> 1) and with the exact point-source
#               steady-state result of Lancaster et al. 2020 (arXiv:1909.06381, Eq. 4.21, b = v t) valid for all M_Q.
#   "softening" T3: M_Q = 17.4 and the softening of the sink from 1.5 to 0.25 cells. Besides the force it checks the
#               conservation of the total momentum: the momentum gained by psi, P_psi = integral of hbar Im(psi* grad psi) dV,
#               must equal the momentum lost by the sink, M (v0 - v). The sink feels the wake at a point while psi feels the
#               Gaussian sink, so the two are not exactly equal; the difference is reported as a function of the softening.
# Output: a table, and plots in resources/data/sink_friction_scan_<SCAN>/ (the output of the single runs is deleted).
# =====================================================================================================

SCAN = os.environ.get("SCAN", "mq")   # "mq", "softening" or "t2" (T2 convergence: fixed physical softening, M_Q list from MQ_LIST)
SMOOTH_FORCE = os.environ.get("SMOOTH_FORCE", "0") == "1"   # SinkNBody smooth_force: the sink feels the environment force smoothed with the Gaussian of
                                  # its own potential (action = reaction); the linear theory then has the Gaussian squared

# ---------------------------------------------------- common settings
V_SINK = 30.0                     # speed of the sink [kpc/Gyr]
BOX = 9.6                         # the box is [-BOX, BOX]^3, i.e. 19.2 kpc
N = int(os.environ.get("N", 128))             # grid resolution (cell 0.15 kpc at 128)
H = float(os.environ.get("H", 4.0e-4))        # time step [Gyr] (stability limit 6.1e-4 at N = 128, scales as 1 / N^2)
M_S = 2.5e-22                     # ULDM particle mass [eV]
TRAVEL_FRACTION = float(os.environ.get("TRAVEL_FRACTION", 0.5))   # the sink travels this fraction of the box length
RECORD_EVERY = max(1, round(1.6e-3 / H))     # the theory and the momentum are evaluated every this many steps (1.6e-3 Gyr)
EPS_PHYS = float(os.environ.get("EPS_PHYS", 0.075))                  # T2: softening_bh in kpc, the same physical value for every N (0.29 lambda at v = 30)
TAG = os.environ.get("TAG", "")   # T2: suffix of the output folder
FIT_FROM_B = 7.0                  # averages are taken for v t / lambda above this
RHO_MAX = 3.0e5                   # density of the background [Msun/kpc^3] (lowered for heavy sinks, see below)
RHO_M_MAX = 3.0e12                # upper limit of rho * M, which keeps the speed change of the sink at a few per cent

if SCAN == "mq":
    CASES = [dict(mach_q=q, softening_cells=0.5) for q in (100.0, 30.0, 10.0, 3.0, 1.0, 0.3)]
elif SCAN == "t2":
    CASES = [dict(mach_q=float(q), softening_cells=EPS_PHYS / (2.0 * BOX / N)) for q in os.environ.get("MQ_LIST", "17.4,1,0.3").split(",")]
else:
    CASES = [dict(mach_q=17.4, softening_cells=c) for c in (3.0, 1.5, 1.0, 0.5, 0.25)]

L_BOX = 2.0 * BOX
TOTAL_TIME = TRAVEL_FRACTION * L_BOX / V_SINK
OUTPUT_DIR = os.path.join("resources", "data", f"sink_friction_scan_{SCAN}" + ("_smooth" if SMOOTH_FORCE else "") + TAG)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def literature_point_source(mach_q, b_values):
    """C_rel of Lancaster et al. 2020, Eq. 4.21: exact point source, steady state, cutoff b (in units of lambda).

    1F1(i/M_Q; 1; i q) is obtained from its differential equation z w'' + (1 - z) w' - a w = 0 (z = i q), and
    |Gamma(1 - i x)|^2 = pi x / sinh(pi x). For large M_Q this tends to Eq. 4.26, Cin(2b) + sin(2b)/(2b) - 1."""
    a = 1j / mach_q
    b_values = np.atleast_1d(np.asarray(b_values, dtype=float))
    q0 = 1.0e-6
    z0 = 1j * q0
    w0 = 1.0 + a * z0 + a * (a + 1.0) / 4.0 * z0 ** 2
    dw0 = 1j * (a + a * (a + 1.0) / 2.0 * z0)

    def rhs(q, y):
        z = 1j * q
        d2w_dz2 = (a * y[0] - (1.0 - z) * (y[1] / 1j)) / z
        return [y[1], -d2w_dz2]

    q = np.geomspace(q0, 2.0 * b_values.max(), 60000)
    w = solve_ivp(rhs, [q0, q[-1]], [w0, dw0], t_eval=q, rtol=1e-11, atol=1e-13, method="DOP853").y[0]
    f = np.abs(w) ** 2

    def cumulative(g):
        return np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(q))))

    i0 = cumulative(f) + q0                                          # integral of f, plus the piece on [0, q0]
    i1 = cumulative(q * f) + q0 ** 2 / 2.0                           # integral of q f
    i2 = cumulative(f * np.log(q)) + q0 * (np.log(q0) - 1.0)         # integral of f ln q
    x = 1.0 / mach_q
    prefactor = 0.5 * mach_q * np.exp(np.pi * x) * np.pi * x / np.sinh(np.pi * x)
    result = []
    for b in b_values:
        j0, j1, j2 = (np.interp(2.0 * b, q, arr) for arr in (i0, i1, i2))
        result.append(prefactor * (j1 / b - (2.0 - np.log(2.0 * b)) * j0 - j2))
    return np.array(result)


# constants of the unit system (G and hbar/m are set when a Simulation_Class is built; no output is created before initialize)
probe = Simulation_Class(dim=3, boundaries=[(-BOX, BOX)] * 3, N=N, total_time=H, h=H, order_of_evolution=2,
                         use_gravity=True, use_units=True, use_sponge=False, self_int=False, m_s=M_S)
G = float(probe.G)
HBAR = float(probe.h_bar_tilde)                   # hbar / m [kpc^2/Gyr]
DX = min(probe.dx)
del probe
LAMBDA = HBAR / V_SINK                            # lambda = hbar / (m v) of the paper [kpc]
H_MAX = 4.0 / (3.0 * np.pi) * DX ** 2 / HBAR * 0.5
print(f"scan '{SCAN}' (smooth_force = {SMOOTH_FORCE}): {len(CASES)} cases, box {L_BOX} kpc, N = {N} (cell {DX:.3f} kpc), v = {V_SINK} kpc/Gyr, hbar/m = {HBAR:.3f}, "
      f"lambda = {LAMBDA:.3f} kpc ({LAMBDA / DX:.1f} cells), step {H} (limit {H_MAX:.2e}), {round(TOTAL_TIME / H)} steps per case")
if H > H_MAX:
    print("WARNING: H is above the stability limit of the split-step scheme")


def run_case(mach_q, softening_cells):
    """Runs one case and returns a dictionary with the measured and the predicted drag coefficient and the momentum balance."""

    m_sink = HBAR * V_SINK / (G * mach_q)                         # M_Q = hbar v / (G M)
    rho = min(RHO_MAX, RHO_M_MAX / m_sink)
    eps = softening_cells * DX
    f_rel = 4.0 * np.pi * rho * (G * m_sink / V_SINK) ** 2        # F_rel of the paper [Msun kpc/Gyr^2]
    volume = L_BOX ** 3
    print(f"\n######## M_Q = {mach_q}: sink {m_sink:.3e} Msun, background {rho:.3e} Msun/kpc^3, softening {eps:.3f} kpc "
          f"= {softening_cells} cells = {eps / LAMBDA:.2f} lambda")

    sim = Simulation_Class(dim=3, boundaries=[(-BOX, BOX)] * 3, N=N, total_time=TOTAL_TIME, h=H, order_of_evolution=2,
                           use_gravity=True, use_units=True, use_sponge=False, self_int=False, m_s=M_S)
    wave_vector = Wave_vector_class(
        packet_type="resources/solitons/GroundState(1).dat", means=[0, 0, 0], st_deviations=[0.5, 0.5, 0.5], simulation=sim,
        mass=1, omega=1, momenta=[0, 0.0, 0], spin=0, desired_soliton_mass=5.0e7)       # placeholder, psi is replaced below
    sim.add_wave_vector(wave_vector)
    x_start = -TRAVEL_FRACTION * L_BOX / 2.0
    sink = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[m_sink], initial_positions=[[x_start, 0.0, 0.0]],
                     initial_velocities=[[V_SINK, 0.0, 0.0]], softening_bh=eps, smooth_force=SMOOTH_FORCE)
    sim.add_baryons(sink)
    sim.initialize_simulation()
    wave_function = sim.wave_functions[0]
    wave_function.psi = cp.full(wave_function.psi.shape, np.sqrt(rho), dtype=cp.complex128)

    # ---- linear theory for this exact set-up (see sink_friction.py for the derivation)
    kx, ky, kz = (cp.asarray(k) for k in sim.k_space)
    k2 = kx * kx + ky * ky + kz * kz
    nonzero = k2 > 0
    gauss = cp.exp(-0.5 * k2 * eps ** 2)
    omega_sq = (HBAR ** 2 * k2 ** 2 / 4.0 - 4.0 * np.pi * G * rho).astype(cp.complex128)
    omega_k = cp.sqrt(omega_sq)
    kappa = kx * V_SINK
    denominator = omega_sq - kappa ** 2
    near_resonance = cp.abs(denominator) < 1.0e-9 * (cp.abs(omega_sq) + kappa ** 2 + 1.0)
    kappa = cp.where(near_resonance, kappa * (1.0 + 1.0e-6) + 1.0e-9, kappa)
    denominator = omega_sq - kappa ** 2
    kernel = gauss ** 2 if SMOOTH_FORCE else gauss                 # the sink acts through gauss, and feels through gauss if smoothed
    weight = cp.where(nonzero, 1j * kx * kernel / cp.where(nonzero, k2, 1.0), 0.0)

    def linear_theory_force(t):
        response = ((1j * kappa * cp.sin(omega_k * t) - omega_k * cp.cos(omega_k * t)) * cp.exp(1j * kappa * t)
                    + omega_k) / (denominator * omega_k)
        return float(16.0 * np.pi ** 2 * G ** 2 * m_sink ** 2 * rho / volume * cp.sum(weight * response).real)

    def psi_momentum_x():
        """Momentum of psi along x, integral of hbar Im(psi* d psi/dx) dV [Msun kpc/Gyr] (psi at the end of a step; the closing
        half kick of the step is not applied yet, a small offset of the order h)."""
        psi = wave_function.psi
        d_psi = cp.fft.ifftn(1j * kx * cp.fft.fftn(psi))
        return float(HBAR * cp.sum(cp.imag(cp.conj(psi) * d_psi)) * sim.dV)

    # ---- run
    n_steps = sim.num_steps
    velocity = np.zeros(n_steps + 1)
    velocity[0] = V_SINK
    n_record = n_steps // RECORD_EVERY
    momentum_psi = np.zeros(n_record + 1)
    original_step = sim.evolution._perform_evolution_step

    def step_and_record(wave_functions, total_density, step, save_step):
        result = original_step(wave_functions, total_density, step, save_step)
        velocity[step + 1] = float(cp.asnumpy(sink.velocities)[0, 0])
        if (step + 1) % RECORD_EVERY == 0:
            momentum_psi[(step + 1) // RECORD_EVERY] = psi_momentum_x()
        return result

    sim.evolution._perform_evolution_step = step_and_record
    sim.evolution.diagnostics.analytic_sink_energy = True            # analytic sink-sink energy (no grid artefact)
    sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=max(1, n_steps // 10))
    energy_total = np.loadtxt(os.path.join(sim.evolution.scribe.snapshot_directory, "energy.csv"), delimiter=",", skiprows=1,
                              usecols=4)                              # E_total column
    energy_drift = float(np.abs(energy_total - energy_total[0]).max() / (0.5 * m_sink * (V_SINK ** 2 - velocity[-1] ** 2)))

    time = np.arange(n_steps + 1) * H
    half = RECORD_EVERY
    index = np.arange(half, n_steps - half + 1, RECORD_EVERY)
    t_sim = time[index]
    c_sim = -m_sink * (velocity[index + half] - velocity[index - half]) / (2.0 * half * H) / f_rel

    t_theory = time[::RECORD_EVERY][1:]
    c_theory = -np.array([linear_theory_force(t) for t in t_theory]) / f_rel
    b_tilde = V_SINK * t_sim / LAMBDA
    c_exact = literature_point_source(mach_q, b_tilde)
    c_theory_at_sim = np.interp(t_sim, t_theory, c_theory)

    t_momentum = time[::RECORD_EVERY]
    impulse = m_sink * (V_SINK - velocity[::RECORD_EVERY])                       # momentum lost by the sink
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum_error = np.where(impulse > 1.0e-9 * m_sink * V_SINK, momentum_psi / impulse - 1.0, np.nan)

    window = b_tilde >= FIT_FROM_B
    late = t_momentum >= 0.5 * t_momentum[-1]
    result = dict(
        mach_q=mach_q, softening_cells=softening_cells, eps_over_lambda=eps / LAMBDA, mass=m_sink, rho=rho,
        slowdown=1.0 - velocity[-1] / V_SINK, t_sim=t_sim, b_tilde=b_tilde, c_sim=c_sim, c_theory=c_theory_at_sim,
        c_exact=c_exact, mean_sim=float(np.mean(c_sim[window])), mean_theory=float(np.mean(c_theory_at_sim[window])),
        mean_exact=float(np.mean(c_exact[window])), t_momentum=t_momentum, momentum_error=momentum_error,
        momentum_error_late=float(np.nanmean(momentum_error[late])),
        energy_drift=energy_drift, momentum_offset=float((momentum_psi[-1] - impulse[-1]) / (m_sink * V_SINK)))
    print(f"speed {V_SINK:.3f} -> {velocity[-1]:.4f} kpc/Gyr ({100 * result['slowdown']:.2f} % slower); mean C for b > {FIT_FROM_B}: "
          f"simulation {result['mean_sim']:.3f}, linear theory {result['mean_theory']:.3f}, exact steady state {result['mean_exact']:.3f}; "
          f"momentum of psi / momentum lost by the sink - 1 = {100 * result['momentum_error_late']:+.2f} % (late half of the run)")

    np.savez(os.path.join(OUTPUT_DIR, f"case_MQ{mach_q:g}_N{N}_H{H:.1e}.npz"), **{k: v for k, v in result.items()})
    shutil.rmtree(sim.evolution.scribe.snapshot_directory, ignore_errors=True)    # keep only the plots (the folder exists after evolve)
    del sim, sink, wave_function, kx, ky, kz, k2, gauss, omega_sq, omega_k, kappa, denominator, weight
    cp.get_default_memory_pool().free_all_blocks()
    return result


results = [run_case(**case) for case in CASES]


# =====================================================================================================
# TABLE AND PLOTS
# =====================================================================================================
print(f"\n{'M_Q':>6} {'eps/lambda':>10} {'slowdown':>9} {'C sim':>7} {'C linear':>9} {'C exact':>8} {'sim/lin-1':>10} {'sim/exact-1':>12} {'P_psi/J-1':>10}")
for r in results:
    print(f"{r['mach_q']:>6.1f} {r['eps_over_lambda']:>10.2f} {100 * r['slowdown']:>8.2f}% {r['mean_sim']:>7.3f} {r['mean_theory']:>9.3f} "
          f"{r['mean_exact']:>8.3f} {100 * (r['mean_sim'] / r['mean_theory'] - 1):>+9.1f}% {100 * (r['mean_sim'] / r['mean_exact'] - 1):>+11.1f}% "
          f"{100 * r['momentum_error_late']:>+9.2f}%")
print("(C averaged for v t / lambda > %.0f; linear theory needs a slowdown of a few per cent and M_Q >> 1; the exact steady-state "
      "formula has no box, no softening and no switch-on transient)" % FIT_FROM_B)

n_cases = len(results)
n_cols = min(3, n_cases)
n_rows = int(np.ceil(n_cases / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), squeeze=False)
for ax, r in zip(axes.flat, results):
    ax.plot(r["t_sim"], r["c_sim"], label="simulation")
    ax.plot(r["t_sim"], r["c_theory"], "--", color="black", label="linear theory (this box)")
    ax.plot(r["t_sim"], r["c_exact"], ":", color="gray", label="exact point source, steady state")
    ax.set_title(f"M_Q = {r['mach_q']:g}, eps = {r['eps_over_lambda']:.2f} lambda, slowdown {100 * r['slowdown']:.1f} %")
    ax.set_xlabel("time [Gyr]")
    ax.set_ylabel("C_rel")
axes.flat[0].legend(fontsize=8)
for ax in list(axes.flat)[n_cases:]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "friction_curves.png"), dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for r in results:
    label = f"M_Q = {r['mach_q']:g}" if SCAN == "mq" else f"eps = {r['eps_over_lambda']:.2f} lambda"
    axes[0].plot(r["t_momentum"], 100 * r["momentum_error"], label=label)
axes[0].axhline(0.0, color="gray", linestyle=":")
axes[0].set_xlabel("time [Gyr]")
axes[0].set_ylabel("P_psi / (momentum lost by the sink) - 1  [%]")
axes[0].set_title("momentum balance between psi and the sink")
axes[0].legend(fontsize=8)
x_values = [r["mach_q"] if SCAN == "mq" else r["eps_over_lambda"] for r in results]
axes[1].plot(x_values, [r["mean_sim"] for r in results], "o-", label="simulation")
axes[1].plot(x_values, [r["mean_theory"] for r in results], "s--", color="black", label="linear theory (this box)")
axes[1].plot(x_values, [r["mean_exact"] for r in results], "^:", color="gray", label="exact point source, steady state")
axes[1].set_xlabel("quantum Mach number M_Q" if SCAN == "mq" else "softening / lambda")
if SCAN == "mq":
    axes[1].set_xscale("log")
axes[1].set_ylabel(f"mean C_rel for v t / lambda > {FIT_FROM_B:g}")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "friction_summary.png"), dpi=130)
print(f"Plots saved in: {OUTPUT_DIR}")
