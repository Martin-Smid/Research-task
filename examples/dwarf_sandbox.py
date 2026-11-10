import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas


# =====================================================================================================
# DWARF-GALAXY SANDBOX (units: kpc, Msun, Gyr). Everything starts as a Gaussian blob, nothing is relaxed:
#   ULDM : Gaussian envelope (cold, zero phase) with a small smooth random perturbation, m_s = 2.5e-22 eV
#   stars: Gaussian cloud of N-body particles, isotropic Gaussian velocities from the measured circular speed
#   gas  : Gaussian density, sound speed CS, slow rotation (fraction of v_c), thin background
# A black hole is NOT put in by hand: sink formation (gas + stars, no ULDM) may create one if the baryons collapse.
# Masses are of a Fornax-like dwarf scaled to what the 20 kpc box and ULDM core size allow.
# Environment overrides for smoke tests: SBX_TIME [Gyr], SBX_SAVE (steps between snapshots), SBX_N.
# =====================================================================================================

# ---------------------------------------------------- masses [Msun]
M_ULDM = 1.0e8
M_STARS = 1.0e7                   # 10 % of ULDM
M_GAS = 1.0e7                     # gas-rich dwarf, gas : stars = 1 : 1

# ---------------------------------------------------- Gaussian widths (1-sigma per axis) [kpc]
S_ULDM = 1.2
S_STARS = 0.8
S_GAS = 1.5
N_STARS = 100_000
CS = 8.0                          # gas sound speed [kpc/Gyr] (7.8 km/s)
RHO_BG = 2.0                      # background gas density [Msun/kpc^3]
GAS_ROT = 0.3                     # gas rotation speed as a fraction of v_c
STAR_SIGMA_FRAC = 0.7             # star 1D velocity dispersion as a fraction of v_c(r)
PERTURB = 0.2                     # relative amplitude of the random ULDM perturbation
PERTURB_SCALE = 0.5               # smoothing length of the perturbation [kpc]
SEED = 1

# ---------------------------------------------------- numerics
N = int(os.environ.get("SBX_N", 256))
BOX = 10.0                        # half box [kpc]
H = 1e-4                        # time step [Gyr] (stability limit at N=256 is 1.65e-4)
TOTAL_TIME = float(os.environ.get("SBX_TIME", 3))
SAVE_EVERY = int(os.environ.get("SBX_SAVE", 250))
DIAGNOSTICS_EVERY = 100

rng = np.random.default_rng(SEED)

sim = Simulation_Class(
    dim=3,
    boundaries=[(-BOX, BOX)] * 3,
    N=N,
    total_time=TOTAL_TIME,
    h=H,
    order_of_evolution=2,
    use_gravity=True,
    static_potential=None,
    save_max_vals=True,           # max values of the density in time
    m_s=2.5e-22,
    use_sponge=True,              # absorbs ULDM leaving the box (gravitational cooling)
    output_root=str(PROJECT_ROOT / "examples" / "sandbox_dwarf_output"),
    self_int=False,
    sink_formation=dict(source="both", include_uldm=False, federrath_checks=True, dynamic_threshold=True,
                        consecutive_steps=3, check_interval=10),
)
sim.sink_accretion = True         # sinks that form eat gas and stars
dx = min(sim.dx)

# ---------------------------------------------------- ULDM: Gaussian + smooth random field
wave_vector = Wave_vector_class(
    packet_type="resources/solitons/GroundState(1).dat",      # only to build the object, psi is replaced below
    means=[0, 0, 0], st_deviations=[0.5, 0.5, 0.5], simulation=sim, mass=1, omega=1, momenta=[0, 0, 0], spin=0,
    desired_soliton_mass=M_ULDM,
)
gx, gy, gz = (cp.asnumpy(g) for g in sim.grids)
R2 = gx ** 2 + gy ** 2 + gz ** 2
envelope = np.exp(-R2 / (4.0 * S_ULDM ** 2))                   # |psi|^2 ~ exp(-r^2 / 2 sigma^2)
noise = rng.normal(size=R2.shape) + 1j * rng.normal(size=R2.shape)
k2 = sum(k ** 2 for k in np.meshgrid(*[2 * np.pi * np.fft.fftfreq(N, d=dx)] * 3, indexing="ij"))
noise = np.fft.ifftn(np.fft.fftn(noise) * np.exp(-0.5 * k2 * PERTURB_SCALE ** 2))
noise *= PERTURB / np.std(noise)
psi = envelope * (1.0 + noise)
psi *= np.sqrt(M_ULDM / (np.sum(np.abs(psi) ** 2) * sim.dV))
wave_vector.wave_vector[0].psi = psi.astype(np.complex128)
sim.add_wave_vector(wave_vector)

# ---------------------------------------------------- stars: Gaussian positions (velocities set after gravity is known)
stars = Baryons(simulation=sim, N_particles=N_STARS, total_mass=M_STARS, init_profile="hernquist", center=(0.0, 0.0, 0.0))
stars.positions = cp.asarray(rng.normal(size=(N_STARS, 3)) * S_STARS)
stars.invalidate_density_cache()
sim.add_baryons(stars)

# ---------------------------------------------------- gas: Gaussian density (velocities set below)
rho_gas = np.exp(-R2 / (2.0 * S_GAS ** 2))
rho_gas *= M_GAS / (rho_gas.sum() * sim.dV)
rho_gas += RHO_BG
zeros = np.zeros_like(rho_gas)
gas = NBodyGas(simulation=sim, rho=rho_gas, vx=zeros, vy=zeros, vz=zeros, cs=CS, gamma=5.0 / 3.0, cfl=0.4,
               max_substeps=50, name="gas")
sim.add_baryons(gas)

sim.initialize_simulation()
evo = sim.evolution
for wf in sim.wave_functions:
    wf.psi = cp.asarray(wf.psi)

# ---------------------------------------------------- circular speed from the gravity on the grid (all components)
phi_env, phi_sink = evo._compute_gravity_fields(evo._compute_total_density(sim.wave_functions))
fx, fy, fz = (cp.asnumpy(f) for f in sim.propagator.compute_force_grids_from_potential(phi_env + phi_sink))
Rs = np.sqrt(R2)
a_in = -(gx * fx + gy * fy + gz * fz) / np.maximum(Rs, 1e-9)
edges = np.linspace(0.0, 9.0, 37)
centres = 0.5 * (edges[1:] + edges[:-1])
shell = np.digitize(Rs, edges) - 1
v_c = np.array([np.sqrt(max(c * a_in[shell == i].mean(), 0.0)) if np.any(shell == i) else 0.0 for i, c in enumerate(centres)])

# stars: isotropic Gaussian velocities, sigma_1D = STAR_SIGMA_FRAC * v_c(r)
pos = cp.asnumpy(stars.positions)
sig = STAR_SIGMA_FRAC * np.interp(np.linalg.norm(pos, axis=1), centres, v_c)
stars.velocities = cp.asarray(rng.normal(size=(N_STARS, 3)) * sig[:, None])

# gas: rotation about z with GAS_ROT * v_c
Rc = np.hypot(gx, gy)
vrot = GAS_ROT * np.interp(Rs, centres, v_c)
gas.vx = cp.asarray(-vrot * gy / np.maximum(Rc, 1e-9) * (Rc / np.maximum(Rs, 1e-9)))
gas.vy = cp.asarray(vrot * gx / np.maximum(Rc, 1e-9) * (Rc / np.maximum(Rs, 1e-9)))
gas.vz = cp.zeros_like(gas.rho)
gas.vel = [gas.vx, gas.vy, gas.vz]
gas.E = gas.rho * CS ** 2 / (gas.gamma * (gas.gamma - 1.0)) + 0.5 * gas.rho * (gas.vx ** 2 + gas.vy ** 2)

print(f"ULDM {M_ULDM:.2e} | stars {M_STARS:.2e} | gas {M_GAS:.2e} Msun, N {N}, dx {dx:.4f} kpc, h {H} Gyr, "
      f"{round(TOTAL_TIME / H)} steps, snapshot every {SAVE_EVERY}")
print("v_c [kpc/Gyr] at r = 0.5, 1, 2, 4, 8 kpc:", [round(float(np.interp(r, centres, v_c)), 1) for r in (0.5, 1, 2, 4, 8)])
sim.evolution.evolve(sim.wave_functions, save_every=SAVE_EVERY, diagnostics_every=DIAGNOSTICS_EVERY)
print("output folder:", sim.evolution.scribe.snapshot_directory)
