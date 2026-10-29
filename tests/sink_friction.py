from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sici

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody


# =====================================================================================================
# DYNAMICAL FRICTION OF A SINK (BLACK HOLE) MOVING THROUGH A UNIFORM ULDM BACKGROUND (units: kpc, Msun, Gyr)
#
# A sink of mass M_SINK starts moving with speed V_SINK through a uniform wave function (density RHO_BACKGROUND).
# Its gravity makes a wake in the wave function; the wake pulls the sink back (dynamical friction). The same set-up
# was simulated by Lancaster et al. 2020 (arXiv:1909.06381): a perturber travels half of a periodic box.
#
# What is compared (all three give the drag force F in units of F_rel = 4 pi rho (G M / v)^2, the C_rel of that paper):
#   1) simulation: F = M dv/dt of the sink in this code (psi kicked by the sink potential, force on the sink from the
#      density of psi on the grid, sink moved by NBody.drift),
#   2) linear theory for exactly this set-up (periodic box, Gaussian sink of width softening_bh, switched on at t = 0,
#      self-gravity of psi included): a discrete sum over the box modes, evaluated below. It should agree with 1)
#      closely while the sink is only weakly perturbed (small Delta v / v, M_Q >> 1),
#   3) literature, infinite medium, steady state, point source (Eq. 4.26 of the paper): C = Cin(2b) + sin(2b)/(2b) - 1,
#      with b = v t / lambda (lambda = hbar / (m v)) as the stand-in for the cutoff. Only indicative: it has no box,
#      no softening and no switch-on transient, so expect a similar size and growth, not the same wiggles.
# =====================================================================================================

# ---------------------------------------------------- background and sink
RHO_BACKGROUND = 3.0e5            # density of the uniform wave function [Msun/kpc^3]
M_SINK = 3.0e6                    # mass of the sink [Msun]
V_SINK = 30.0                     # speed of the sink [kpc/Gyr] (1 kpc/Gyr = 0.978 km/s)

# ---------------------------------------------------- simulation
BOX = 9.6                         # the box is [-BOX, BOX]^3, i.e. 19.2 kpc
N = 256                           # grid resolution (cell 0.15 kpc)
H = 1.0e-4                        # time step [Gyr] (must stay below the stability limit printed below)
M_S = 2.5e-22                     # ULDM particle mass [eV]
SOFTENING_CELLS = 0.25             # width of the Gaussian sink potential (softening_bh) in cells; smaller = closer to a point mass
                                  # (the literature curve is for a point source), but at some point the grid no longer resolves it
                                  # (the default of SinkNBody is 1.5 cells; 1.5 / 1.0 / 0.5 gave a mean C of 2.70 / 2.95 / 3.17)
TRAVEL_FRACTION = 0.5             # the sink travels this fraction of the box length (more would meet its own wake)
RECORD_EVERY = 4                  # the theory is evaluated every this many steps
FIT_FROM_B = 7.0                  # the agreement is judged for v t / lambda above this (as in the paper)

L_BOX = 2.0 * BOX
TOTAL_TIME = TRAVEL_FRACTION * L_BOX / V_SINK

sim = Simulation_Class(

    dim=3, # 3D simulation
    boundaries=[(-BOX, BOX)] * 3, # box, the centre is at (0,0,0)
    N=N, # grid resolution
    total_time=TOTAL_TIME, # total time [Gyr]
    h=H, # time step [Gyr]
    order_of_evolution=2,
    use_gravity=True, # gravity on (the wave function feels its own wake too)
    use_units=True, # kpc, Msun, Gyr
    use_sponge=False,
    self_int=False,
    m_s=M_S, # ULDM particle mass [eV]

)

wave_vector = Wave_vector_class(

    packet_type="resources/solitons/GroundState(1).dat", # only a placeholder, psi is replaced by a uniform one below
    means=[0, 0, 0],
    st_deviations=[0.5, 0.5, 0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0, 0.0, 0],
    spin=0,
    desired_soliton_mass=5.0e7,

)
sim.add_wave_vector(wave_vector)

x_start = -TRAVEL_FRACTION * L_BOX / 2.0      # the sink moves along +x from -L/4 to +L/4
sink = SinkNBody(

    simulation=sim,
    N_sinks=1, # one sink
    initial_masses=[M_SINK], # its mass
    initial_positions=[[x_start, 0.0, 0.0]], # starts on the x axis
    initial_velocities=[[V_SINK, 0.0, 0.0]], # moving along +x
    softening_bh=SOFTENING_CELLS * 2.0 * BOX / N, # smoothing length of the sink potential [kpc]

)
sim.add_baryons(sink)
sim.initialize_simulation()
wave_function = sim.wave_functions[0]
wave_function.psi = cp.full(wave_function.psi.shape, np.sqrt(RHO_BACKGROUND), dtype=cp.complex128)   # uniform, at rest


# =====================================================================================================
# DIMENSIONLESS NUMBERS AND LINEAR THEORY FOR THIS EXACT SET-UP
# =====================================================================================================
G = float(sim.G)
hbar = float(sim.h_bar_tilde)                     # hbar / m [kpc^2/Gyr]
eps = float(sink.softening_bh)                    # Gaussian width of the sink potential [kpc]
volume = L_BOX ** 3
lam = hbar / V_SINK                               # lambda = hbar / (m v) of the paper [kpc]
mach_q = hbar * V_SINK / (G * M_SINK)             # quantum Mach number M_Q = v lambda / (G M)  (>> 1: linear theory works)
f_rel = 4.0 * np.pi * RHO_BACKGROUND * (G * M_SINK / V_SINK) ** 2     # F_rel of the paper [Msun kpc/Gyr^2]
dx = min(sim.dx)
h_max = 4.0 / (3.0 * np.pi) * dx ** 2 / hbar * 0.5                    # the stability limit used by Simulation_Class

print(f"background {RHO_BACKGROUND:.2e} Msun/kpc^3, sink {M_SINK:.2e} Msun at {V_SINK} kpc/Gyr, hbar/m = {hbar:.3f} kpc^2/Gyr")
print(f"lambda = hbar/(m v) = {lam:.3f} kpc (cell {dx:.3f}, sink softening {eps:.3f} = {eps / lam:.2f} lambda), "
      f"quantum Mach number M_Q = {mach_q:.1f}")
print(f"F_rel = {f_rel:.3e} Msun kpc/Gyr^2, box {L_BOX} kpc, the sink travels {TRAVEL_FRACTION * L_BOX:.1f} kpc in {TOTAL_TIME:.3f} Gyr "
      f"({round(TOTAL_TIME / H)} steps of {H} Gyr; stability limit {h_max:.2e})")
if H > h_max:
    print("WARNING: H is above the stability limit of the split-step scheme")

kx, ky, kz = (cp.asarray(k) for k in sim.k_space)
k2 = kx * kx + ky * ky + kz * kz
nonzero = k2 > 0
k2_safe = cp.where(nonzero, k2, 1.0)
gauss = cp.exp(-0.5 * k2 * eps ** 2)                                  # Fourier transform of the Gaussian sink
omega_sq = (hbar ** 2 * k2 ** 2 / 4.0 - 4.0 * np.pi * G * RHO_BACKGROUND).astype(cp.complex128)   # Bogoliubov + Jeans
omega_k = cp.sqrt(omega_sq)                                           # imaginary for the Jeans-unstable box modes
kappa = kx * V_SINK                                                   # k . v
denominator = omega_sq - kappa ** 2
near_resonance = cp.abs(denominator) < 1.0e-9 * (cp.abs(omega_sq) + kappa ** 2 + 1.0)
kappa = cp.where(near_resonance, kappa * (1.0 + 1.0e-6) + 1.0e-9, kappa)    # removable singularity, nudge it
denominator = omega_sq - kappa ** 2
weight = cp.where(nonzero, 1j * kx * gauss / k2_safe, 0.0)


def linear_theory_force(t):
    """Force on the sink along x at time t [Msun kpc/Gyr^2] from the linear response of psi (sink at constant speed).

    u_k'' + Omega_k^2 u_k = -psi0 k^2 Phi_k exp(-i k.v t) for the density contrast u_k = delta rho_k / psi0, solved with
    u = u' = 0 at t = 0; the force is the gradient of the potential of delta rho at the sink."""
    response = ((1j * kappa * cp.sin(omega_k * t) - omega_k * cp.cos(omega_k * t)) * cp.exp(1j * kappa * t)
                + omega_k) / (denominator * omega_k)
    total = cp.sum(weight * response)
    return float(16.0 * np.pi ** 2 * G ** 2 * M_SINK ** 2 * RHO_BACKGROUND / volume * total.real)


def literature_coefficient(b):
    """C_rel of Lancaster et al. 2020, Eq. 4.26 (point source, steady state, cutoff b in units of lambda)."""
    si, ci = sici(2.0 * b)
    return (np.euler_gamma + np.log(2.0 * b) - ci) + np.sin(2.0 * b) / (2.0 * b) - 1.0


# =====================================================================================================
# RUN (after every step the velocity of the sink is stored)
# =====================================================================================================
n_steps = sim.num_steps
velocity = np.zeros(n_steps + 1)
position = np.zeros(n_steps + 1)
velocity[0], position[0] = V_SINK, x_start
original_step = sim.evolution._perform_evolution_step


def step_and_record(wave_functions, total_density, step, save_step):
    result = original_step(wave_functions, total_density, step, save_step)
    velocity[step + 1] = float(cp.asnumpy(sink.velocities)[0, 0])
    position[step + 1] = float(cp.asnumpy(sink.positions)[0, 0])
    return result


sim.evolution._perform_evolution_step = step_and_record
sim.evolution.evolve(sim.wave_functions, save_every=400, diagnostics_every=400)

time = np.arange(n_steps + 1) * H
print(f"\nthe sink moved from x = {position[0]:.2f} to {position[-1]:.2f} kpc, "
      f"speed {velocity[0]:.3f} -> {velocity[-1]:.4f} kpc/Gyr ({100 * (velocity[-1] / velocity[0] - 1):+.3f} %)")


# =====================================================================================================
# COMPARISON
# =====================================================================================================
half = RECORD_EVERY                                                # force = M dv/dt over +-RECORD_EVERY steps
index = np.arange(half, n_steps - half + 1, RECORD_EVERY)
t_sim = time[index]
c_sim = -M_SINK * (velocity[index + half] - velocity[index - half]) / (2.0 * half * H) / f_rel     # drag, positive

t_theory = time[::RECORD_EVERY][1:]
force_theory = np.array([linear_theory_force(t) for t in t_theory])
c_theory = -force_theory / f_rel
# speed change from the theory: integral of F / M (trapezoid, the force at t = 0 is zero)
t_cum = np.concatenate(([0.0], t_theory))
f_cum = np.concatenate(([0.0], force_theory))
delta_v_theory = np.concatenate(([0.0], np.cumsum(0.5 * (f_cum[1:] + f_cum[:-1]) * np.diff(t_cum)))) / M_SINK
delta_v_sim = velocity - V_SINK

b_tilde = V_SINK * t_sim / lam
c_literature = literature_coefficient(np.maximum(b_tilde, 1.0e-3))
c_theory_at_sim = np.interp(t_sim, t_theory, c_theory)
window = b_tilde >= FIT_FROM_B

print(f"\n{'t [Gyr]':>9} {'b = vt/lambda':>14} {'C sim':>9} {'C linear theory':>16} {'C literature':>13}")
for i in np.linspace(0, len(t_sim) - 1, 9).astype(int):
    print(f"{t_sim[i]:>9.3f} {b_tilde[i]:>14.1f} {c_sim[i]:>9.3f} {c_theory_at_sim[i]:>16.3f} {c_literature[i]:>13.3f}")

delta_v_theory_end = float(np.interp(time[-1], t_cum, delta_v_theory))
print(f"\nspeed change over the run: simulation {delta_v_sim[-1]:.4e} kpc/Gyr, linear theory {delta_v_theory_end:.4e} kpc/Gyr "
      f"({100 * (delta_v_sim[-1] / delta_v_theory_end - 1):+.1f} % difference), "
      f"relative to the speed {100 * abs(delta_v_sim[-1]) / V_SINK:.2f} % (theory assumes this is small)")
if np.any(window):
    mean_c = float(np.mean(c_theory_at_sim[window]))
    rms = float(np.sqrt(np.mean((c_sim[window] - c_theory_at_sim[window]) ** 2)))
    mean_sim = float(np.mean(c_sim[window]))
    print(f"for b > {FIT_FROM_B}: mean C simulation {mean_sim:.3f}, linear theory {mean_c:.3f} ({100 * (mean_sim / mean_c - 1):+.1f} %), "
          f"rms difference {rms:.3f} = {100 * rms / mean_c:.1f} % of the theory; literature (point source, steady state) "
          f"{float(np.mean(c_literature[window])):.3f}")


# =====================================================================================================
# PLOTS (saved into the simulation directory)
# =====================================================================================================
output_dir = sim.evolution.scribe.snapshot_directory
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

axes[0].plot(t_sim, c_sim, label="simulation")
axes[0].plot(t_theory, c_theory, "--", color="black", label="linear theory (this box, Gaussian sink)")
axes[0].plot(t_sim, c_literature, ":", color="gray", label="literature: infinite medium, point source, b = v t")
axes[0].set_xlabel("time [Gyr]")
axes[0].set_ylabel("drag force / F_rel  (C_rel)")
axes[0].set_title(f"dynamical friction coefficient (M_Q = {mach_q:.1f})")
axes[0].legend()

axes[1].plot(time, delta_v_sim, label="simulation")
axes[1].plot(t_cum, delta_v_theory, "--", color="black", label="linear theory")
axes[1].set_xlabel("time [Gyr]")
axes[1].set_ylabel("speed change of the sink [kpc/Gyr]")
axes[1].set_title("speed of the sink")
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "sink_friction.png"), dpi=150)
print(f"Plot saved in: {output_dir}")
