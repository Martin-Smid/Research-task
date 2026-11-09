from pathlib import Path
import csv
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp

from DressedSoliton import solve_physical, write_profile
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody


# =====================================================================================================
# SOLITON WITH A CENTRAL BLACK HOLE THAT ABSORBS ULDM (units: kpc, Msun, Gyr)
#
#   The start is the "dressed" soliton: the ground state of Schroedinger-Poisson with the black hole in the centre
#   (DressedSoliton.py, 1D solver), so the run starts in equilibrium. The black hole then swallows psi through
#   SinkNBody.accrete_from_uldm (rate 16 pi (G M / c^2)^2 c rho, Bucciotti & Trincherini 2309.02482).
#   What to look at (all in the output folder, see the printout at the end):
#     mass_log.csv   time, mass of psi, mass of the black hole (+ reservoir), and the difference
#                    (psi + sink) - (their sum at the start): it must stay ~0, the mass only moves from psi to the sink
#     mass_log.png   the same as a plot
#     energy.csv     energy components (energy conservation); the E_total jump from the moved mass is expected, see the log
#     max_values.csv maximum of the density |psi|^2 in time (max values are switched on by save_max_vals=True)
#
#   N, BOX, EPS, H, TOTAL_TIME come from correct_params.py (set N, M_SOL, M_S, M_BH there and paste its output here).
# =====================================================================================================

# ---------------------------------------------------- physics
M_S = 1.0e-21                     # ULDM particle mass [eV]
M_SOL = 1.0e8                     # soliton mass [Msun]
BH_FRACTION = 0.25                # black hole mass as a fraction of the soliton mass
M_BH = BH_FRACTION * M_SOL        # black hole mass [Msun]
ULDM_SCALE = 1.0e8                # multiplies the absorption cross section. 1 = physical: only ~90 Msun/Gyr here, i.e. ~0.7 Msun in
                                  # the whole run, invisible next to 1e8. 1e8 (NOT physical) absorbs of the order of 10 % of the soliton in TOTAL_TIME.

# ---------------------------------------------------- numerics (output of correct_params.py for N = 256, M_BH = 2.5e7)
N = 256                           # grid points per axis
BOX = 0.432                       # half box [kpc]
EPS = 0.00337                     # softening of the black hole = 1 cell [kpc]
H = 1.1e-6                        # time step [Gyr]
TOTAL_TIME = 0.0011               # run time [Gyr], ~1000 steps, 1.4 dynamical times (correct_params suggests 0.0076 = 10)
LOG_EVERY = 20                    # steps between lines of mass_log.csv and of the energy / max values logs

# ---------------------------------------------------- the dressed soliton profile (1D, CPU, ~1 min)
sol = solve_physical(M_S, M_BH, M_SOL, EPS)
PROFILE = PROJECT_ROOT / "resources" / "solitons" / f"Dressed_m{M_S:.0e}_Msol{M_SOL:.3e}_Mbh{M_BH:.2e}_eps{EPS}.dat"
write_profile(str(PROFILE), sol)
DESIRED_MASS = float(sol["mass"])

# ---------------------------------------------------- simulation
sim = Simulation_Class(
    dim=3,
    boundaries=[(-BOX, BOX)] * 3,
    N=N,
    total_time=TOTAL_TIME,
    h=H,
    order_of_evolution=2,
    use_gravity=True,
    use_units=True,
    use_sponge=False,
    self_int=False,
    save_max_vals=True,           # max values of the density
    m_s=M_S,
)
sim.strict_time_step = True
sim.sink_accretion = True         # the sinks accrete ...
sim.sink_accretion_uldm = True    # ... also psi

wave_vector = Wave_vector_class(
    packet_type=str(PROFILE),
    means=[0, 0, 0],
    st_deviations=[0.5, 0.5, 0.5],      # unused for a profile read from a file
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0, 0, 0],
    spin=0,
    desired_soliton_mass=DESIRED_MASS,
)
sim.add_wave_vector(wave_vector)

sink = SinkNBody(
    simulation=sim,
    N_sinks=1,
    initial_masses=[M_BH],
    initial_positions=[[0.0, 0.0, 0.0]],
    initial_velocities=[[0.0, 0.0, 0.0]],
    softening_bh=EPS,
    smooth_force=True,
)
sink.uldm_cross_section_scale = ULDM_SCALE
sim.add_baryons(sink)
sim.initialize_simulation()

# ---------------------------------------------------- mass log: wraps one evolution step, writes every LOG_EVERY steps
mass_log = {"path": None, "psi0": None, "sink0": None}
evolution_step = sim.evolution._perform_evolution_step


def psi_mass():
    return float(sum(cp.sum(wf.calculate_density()) for wf in sim.wave_functions)) * sim.dV


def logged_step(wave_functions, total_density, step, save_step):
    result = evolution_step(wave_functions, total_density, step, save_step)
    if mass_log["path"] is None:                                    # first call: the output folder exists now
        mass_log["path"] = os.path.join(sim.evolution.scribe.snapshot_directory, "mass_log.csv")
        mass_log["psi0"], mass_log["sink0"] = psi_mass(), float(sink.masses.sum())
        with open(mass_log["path"], "w") as f:
            f.write("time_Gyr,mass_psi,mass_sink,mass_bh,mass_reservoir,difference\n")
    if (step + 1) % LOG_EVERY == 0:
        m_psi, m_sink = psi_mass(), float(sink.masses.sum())
        difference = (m_psi + m_sink) - (mass_log["psi0"] + mass_log["sink0"])
        with open(mass_log["path"], "a") as f:
            f.write(f"{(step + 1) * H:.9g},{m_psi:.12e},{m_sink:.12e},{float(sink.mass_bh.sum()):.12e},"
                    f"{float(sink.mass_res.sum()):.12e},{difference:.6e}\n")
    return result


sim.evolution._perform_evolution_step = logged_step

# ---------------------------------------------------- run
print(f"soliton {DESIRED_MASS:.4e} Msun, black hole {M_BH:.4e} Msun, N {N}, box +-{BOX} kpc, h {H}, {round(TOTAL_TIME / H)} steps")
sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=LOG_EVERY)
folder = sim.evolution.scribe.snapshot_directory
print("output folder:", folder)

# ---------------------------------------------------- plot of mass_log.csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(mass_log["path"], delimiter=",", names=True)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
ax1.plot(data["time_Gyr"] * 1e3, data["mass_psi"], label="soliton (psi)")
ax1.plot(data["time_Gyr"] * 1e3, data["mass_sink"], label="black hole (+ reservoir)")
ax1.set_ylabel("mass [Msun]")
ax1.legend()
ax2.plot(data["time_Gyr"] * 1e3, data["difference"])
ax2.set_ylabel("(psi + sink) - start [Msun]")
ax2.set_xlabel("time [Myr]")
fig.savefig(os.path.join(folder, "mass_log.png"), dpi=120)
