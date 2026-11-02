from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from DressedSoliton import hbar_tilde
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody


# =====================================================================================================
# T5: black hole released at rest inside an undisturbed soliton, the set-up of arXiv:2403.09038 (PyUltraLight; units kpc, Msun, Gyr).
#   soliton 5.57e7 Msun (m = 1e-21 eV) at the origin, black hole 4e6 Msun at (0.125, 0.125, 0) kpc at rest, box [-1, 1]^3, N = 192,
#   softening 0.5 cell. The soliton is the FREE ground state (GroundState(1).dat rescaled to the mass; the rescaling is valid without a
#   black hole), not the dressed state: the paper starts from the undisturbed soliton.
# Everything is set by environment variables (defaults = the set-up above):
#   N, BOX (half box, kpc), EPS_CELLS (softening in cells), SMOOTH (1 = smooth_force), T_MYR (run time), H (time step, Gyr; default 0.9 of the
#   stability limit), M_BH, M_SOL, SAVE_MYR (snapshot interval), TAG.
# Analysis (CPU): tests/t5_analyze.py <simulation folder>.
# =====================================================================================================

M_S = 1.0e-21
M_SOL = float(os.environ.get("M_SOL", 5.57e7))
M_BH = float(os.environ.get("M_BH", 4.0e6))
BH_START = [0.125, 0.125, 0.0]
N = int(os.environ.get("N", 192))
BOX = float(os.environ.get("BOX", 1.0))
EPS_CELLS = float(os.environ.get("EPS_CELLS", 0.5))
SMOOTH = os.environ.get("SMOOTH", "1") == "1"
T_MYR = float(os.environ.get("T_MYR", 60.0))
SAVE_MYR = float(os.environ.get("SAVE_MYR", 5.0))
PROFILE = "resources/solitons/GroundState(1).dat"

dx = 2.0 * BOX / N
H_MAX = 0.5 * 4.0 / (3.0 * np.pi) * dx ** 2 / hbar_tilde(M_S)         # the limit of Simulation_Class.check_time_step_restriction
H = float(os.environ.get("H", 0.9 * H_MAX))
EPS = EPS_CELLS * dx
TOTAL_TIME = T_MYR * 1.0e-3
SAVE_EVERY = max(1, round(SAVE_MYR * 1.0e-3 / H))
print(f"N {N}  box +-{BOX}  dx {dx * 1e3:.3f} pc  eps {EPS * 1e3:.3f} pc  h {H:.3e} (limit {H_MAX:.3e})  steps {round(TOTAL_TIME / H)}  smooth_force {SMOOTH}")

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
    save_max_vals=True,
    m_s=M_S,
)
sim.strict_time_step = True

wave_vector = Wave_vector_class(
    packet_type=PROFILE,
    means=[0, 0, 0],
    st_deviations=[0.5, 0.5, 0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0, 0.0, 0],
    spin=0,
    desired_soliton_mass=M_SOL,
)
sim.add_wave_vector(wave_vector)

sink = SinkNBody(
    simulation=sim,
    N_sinks=1,
    initial_masses=[M_BH],
    initial_positions=[BH_START],
    initial_velocities=[[0.0, 0.0, 0.0]],
    softening_bh=EPS,
    smooth_force=SMOOTH,
)
sim.add_baryons(sink)
sim.initialize_simulation()

sim.evolution.evolve(sim.wave_functions, save_every=SAVE_EVERY, diagnostics_every=max(1, SAVE_EVERY // 5))
print("output folder:", sim.evolution.scribe.snapshot_directory)
