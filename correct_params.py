"""Fill in the numerical parameters of a soliton + black hole simulation from four physical numbers (units: kpc, Msun, Gyr, eV).

Set N, M_SOL, M_S, M_BH below and run `python correct_params.py` (CPU, scipy and astropy only, ~1 min with the 1D solver).
It prints box size, softening, time step, run time, the exact DressedSoliton.py command and a block to paste into your simulation script, plus
a list of checks (OK / WARN). The rules are the ones we measured or derived, and each line says which:
    resolution   dx <= r_min / 4 with r_min = min(Bohr radius of the black hole, half-mass radius of the dressed soliton). A black hole whose
                 Bohr radius is ~1 cell (the N = 256 / 320 runs with M_BH = 5e7) does not stay in equilibrium (not a measured limit, see
                 plans.md: r_B / dx = 18 and r_half / dx = 5 were stationary to 0.14 %).
    softening    eps = 1 cell (the unsmoothed sink needs eps / lambda <~ 0.15, tests/sink_friction*.py; eps = 0.36 cell ran away in testing.py).
    time step    0.9 of the stability limit of Simulation_Class: 4 / (3 pi) dx^2 / hbar_tilde * 0.5. The code only WARNS above it.
    box          at least 4 x the radius that holds 99.9 % of the soliton mass (periodic images, rule of thumb) plus the room the black hole may use.
    potential    H * |V_BH(0)| / hbar_tilde, the phase the black hole potential gives psi in one step; 0.009 in the validated run, 0.35 in the one
                 that ran away. The limit 0.1 is my guess, not measured."""

from pathlib import Path
import math
import sys

import numpy as np

# ---------------------------------------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------------------------------------
N = 320                      # grid points per axis
M_SOL = 1.0e8                # soliton mass [Msun]
M_S = 1.0e-21                # ULDM particle mass [eV]
M_BH = 5.0e7                 # black hole mass [Msun] (0 = no black hole)
BH_ROOM = 0.0                # [kpc] how far from the centre the black hole is allowed to travel (0 = it stays at the centre)
USE_1D_SOLVER = True         # use DressedSoliton.py for the true profile (r_half, r_99, mass); False = estimates for the free soliton
TARGET_CELLS = 6.0           # wanted cells per r_min (comfortable)
MIN_CELLS = 4.0              # least cells per r_min (limit)
RUN_TIME_IN_DYN_TIMES = 10   # suggested run time in dynamical times of the soliton
GPU_MEMORY_GB = 16.0         # P100 on Koios

sys.path.insert(0, str(Path(__file__).resolve().parent))
from DressedSoliton import G, hbar_tilde, solve_physical          # same G and hbar_tilde as Simulation_Class

ht = hbar_tilde(M_S)
r_b = ht ** 2 / (G * M_BH) if M_BH > 0 else math.inf               # gravitational Bohr radius of the black hole [kpc]
r_half = 4.0 * ht ** 2 / (G * M_SOL)                                # half-mass radius of the free soliton (estimate) [kpc]
r_99 = 3.0 * r_half


def radii_of(sol):
    r, rho = sol["r"], ht ** 2 / G * sol["phi"] ** 2
    m = np.concatenate(([0.0], np.cumsum(0.5 * (rho[1:] * r[1:] ** 2 + rho[:-1] * r[:-1] ** 2) * np.diff(r)))) * 4.0 * np.pi
    m /= m[-1]
    return float(np.interp(0.5, m, r)), float(np.interp(0.999, m, r))


def choose(r_half, r_99):
    r_min = min(r_b, r_half)
    dx_comfort, dx_limit = r_min / TARGET_CELLS, r_min / MIN_CELLS
    box_min = max(4.0 * r_99, BH_ROOM + 3.0 * r_99)
    box_comfort, box_max = 0.5 * N * dx_comfort, 0.5 * N * dx_limit
    box = float(f"{min(max(box_comfort, box_min), box_max):.3g}")
    eps = float(f"{2.0 * box / N:.3g}")
    return r_min, box_min, box_max, box, eps


sol = None
for _ in range(4):
    r_min, box_min, box_max, box, eps = choose(r_half, r_99)
    if not USE_1D_SOLVER:
        break
    sol = solve_physical(M_S, M_BH, M_SOL, eps)
    new_half, new_99 = radii_of(sol)
    done = abs(new_half / r_half - 1.0) < 0.02 and abs(new_99 / r_99 - 1.0) < 0.02
    r_half, r_99 = new_half, new_99
    if done:
        break
r_min, box_min, box_max, box, eps = choose(r_half, r_99)
dx = 2.0 * box / N
h_max = 4.0 / (3.0 * math.pi) * dx ** 2 / ht * 0.5
h = float(f"{0.9 * h_max:.2g}")
t_dyn = 2.0 * math.pi * math.sqrt(r_half ** 3 / (G * (M_SOL + M_BH)))
total_time = float(f"{RUN_TIME_IN_DYN_TIMES * t_dyn:.2g}")
v_circ = math.sqrt(G * (M_SOL + M_BH) / r_half)
v_b = ht / r_b if M_BH > 0 else 0.0
m_q = ht * v_circ / (G * M_BH) if M_BH > 0 else math.inf
v_max = ht / (3.0 * dx)                                              # speed whose de Broglie wavelength is 3 cells
phase = h * G * M_BH * math.sqrt(2.0 / math.pi) / eps / ht if M_BH > 0 else 0.0
memory_gb = 300.0 * N ** 3 / 1e9                                     # ~300 B per cell, upper estimate from the out-of-memory at N = 384
mass_file = sol["mass"] if sol is not None else M_SOL
fft_friendly = all(c in (2, 3, 5) for c in [p for p in range(2, N + 1) if N % p == 0 and all(p % q for q in range(2, int(p ** 0.5) + 1))])
profile_name = f"Dressed_m{M_S:.0e}_Msol{M_SOL:.3e}_Mbh{M_BH:.2e}_eps{eps}.dat"


def check(ok, text, hint=""):
    print(f"  [{'OK  ' if ok else 'WARN'}] {text}" + ("" if ok else f"   -> {hint}"))


print(f"particle mass {M_S:.2e} eV: hbar/m = {ht:.5f} kpc^2/Gyr;  soliton {M_SOL:.3e} Msun, black hole {M_BH:.3e} Msun")
print(f"length scales: Bohr radius of the BH r_B = {r_b * 1000:.2f} pc, half-mass radius of the soliton r_1/2 = {r_half * 1000:.2f} pc"
      f"{' (dressed, from DressedSoliton)' if sol is not None else ' (free-soliton estimate)'}, r_99.9% = {r_99 * 1000:.1f} pc")
print(f"speeds: circular at r_1/2 = {v_circ:.1f} kpc/Gyr ({v_circ * 0.978:.0f} km/s), at r_B = {v_b:.1f} kpc/Gyr; quantum Mach number M_Q at r_1/2 = {m_q:.2f}")
print(f"dynamical time of the soliton {t_dyn * 1000:.2f} Myr\n")
print("RECOMMENDED")
print(f"  box   : [-{box}, {box}] kpc   (allowed {box_min:.3f} ... {box_max:.3f})")
print(f"  N     : {N}  -> cell dx = {dx * 1000:.2f} pc  ({r_min / dx:.1f} cells per r_min, {r_b / dx:.1f} per r_B, {r_half / dx:.1f} per r_1/2)")
print(f"  eps   : {eps} kpc (= {eps / dx:.2f} cells)   softening_bh and DressedSoliton --eps")
print(f"  H     : {h} Gyr   (stability limit {h_max:.3e})")
print(f"  time  : {total_time} Gyr = {total_time / h:.0f} steps ({RUN_TIME_IN_DYN_TIMES} dynamical times)")
print(f"  memory: ~{memory_gb:.1f} GB of {GPU_MEMORY_GB:.0f} GB (upper estimate)\n")
print("CHECKS")
check(r_b / dx >= MIN_CELLS, f"r_B = {r_b / dx:.1f} cells", f"M_BH is too heavy for this N: needs N >= {2 * box_min / (r_min / MIN_CELLS):.0f} (or a smaller box and a smaller step)")
check(r_half / dx >= MIN_CELLS, f"r_1/2 = {r_half / dx:.1f} cells", "increase N or shrink the box")
check(box_min <= box_max, f"box fits between {box_min:.3f} and {box_max:.3f} kpc", f"N is too small: needs N >= {math.ceil(2 * box_min / (r_min / MIN_CELLS))}")
check(abs(eps / dx - 1.0) < 0.15, f"eps = {eps / dx:.2f} cells", "keep eps close to one cell")
check(h <= h_max, f"H = {h} below the stability limit {h_max:.3e}", "reduce H")
check(phase < 0.1, f"BH potential phase per step {phase:.3f}", "reduce H (validated 0.009, runaway 0.35; the limit 0.1 is a guess)")
check(memory_gb < GPU_MEMORY_GB, f"memory ~{memory_gb:.1f} GB", "N is too large for this GPU")
check(fft_friendly, f"N = {N} has only the prime factors 2, 3, 5", "use N = 2^a 3^b 5^c for a fast FFT")
check(v_circ < v_max, f"circular speed {v_circ:.0f} < {v_max:.0f} kpc/Gyr (lambda >= 3 cells)", "the de Broglie wavelength of the fast part is under 3 cells: increase N")
if M_BH > 0 and BH_ROOM == 0.0:
    print("  [INFO] BH_ROOM = 0: the box only fits a black hole near the centre; set BH_ROOM to the distance it may travel")
print(f"\nSTEPS\n  python DressedSoliton.py --m {M_S} --M_sol {M_SOL} --M_bh {M_BH} --eps {eps}")
print(f"  (3D version on the same grid, optional)  python -u DressedSoliton3D.py --N {N} --half_box {box} --m {M_S} --M_sol {M_SOL} --M_bh {M_BH} --eps {eps}")
print("\nPASTE INTO THE SIMULATION SCRIPT")
print(f'M_S = {M_S}\nM_BH = {M_BH}\nPROFILE = "resources/solitons/{profile_name}"\nDESIRED_MASS = {mass_file!r}   # the mass DressedSoliton.py prints\n'
      f"N = {N}\nBOX = {box}\nEPS = {eps}\nH = {h}\nTOTAL_TIME = {total_time}\n# SinkNBody(..., softening_bh=EPS, smooth_force=True)   # momentum and energy exact between psi and the sink")
