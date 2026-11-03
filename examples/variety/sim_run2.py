import numpy as np
import matplotlib.pyplot as plt

from resources.Classes.Wave_function_class import *
from resources.Functions.system_fucntions import *
from matplotlib.colors import LogNorm
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody

# -------------------------
# Unit helper
# -------------------------
KM_S_TO_KPC_GYR = 1.0227121650537077  # 1 km/s in kpc/Gyr

# -------------------------
# Simulation
boundaries = [(-30,30)]*3
# -------------------------
sim = Simulation_Class(
    dim=3,
    boundaries=boundaries,
    N=128,
    total_time=2,     # 0.5 Gyr
    h=5e-4,             # 1 Myr
    order_of_evolution=2,
    use_gravity=True,
    static_potential=None,
    save_max_vals=True,
    self_int=False,
    use_sponge=False,
)

# -------------------------
# SMBH sink from t=0
# -------------------------
MBH = 5e5  # Msun  (low-mass SMBH/IMBH for a small galaxy)
sink = SinkNBody(
    simulation=sim,
    N_sinks=1,
    initial_masses=[MBH],
    initial_positions=[[0.0, 0.0, 0.0]],
    initial_velocities=[[0.0, 0.0, 0.0]],
    # keep these fairly "soft" to avoid immediate numerical vacuum-cleaner behavior:
    capture_radius=3.0 * min(sim.dx),
    softening_length=2.0 * min(sim.dx),
)
sim.add_baryons(sink)

# -------------------------
# Stellar component: Hernquist "bulge" particles
# -------------------------
Mstar = 3e8       # Msun
a = 1.0           # kpc (scale radius)
r_trunc = 12.0    # kpc
sigma_kms = 25.0  # km/s velocity dispersion scale
sigma = sigma_kms * KM_S_TO_KPC_GYR

bulge = Baryons(
    simulation=sim,
    N_particles=int(2e5),
    total_mass=Mstar,
    init_profile="hernquist",
    scale_radius=a,
    truncation_radius=r_trunc,
    center=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    vel_sigma=sigma,
)
#sim.add_baryons(bulge)

# -------------------------
# Gas on grid: rotating exponential-ish disk + small floor
# -------------------------
N = sim.N
x, y, z = sim.grids
x = np.array(x); y = np.array(y); z = np.array(z)

R = np.sqrt(x**2 + y**2)          # cylindrical radius
Z = np.abs(z)

Mgas = 1e8     # Msun
R0 = 3.0       # kpc (radial scale)
Z0 = 0.4       # kpc (vertical scale)


rho_shape = np.exp(-R / R0) * np.exp(-Z / Z0)

rho_floor = 1e-12
rho = rho_shape + rho_floor
rho = np.maximum(rho, rho_floor)

M_current = rho.sum() * sim.dV
rho *= (Mgas / M_current)

# rotation curve (simple, smooth): vphi(R) -> v0 for large R
v0_kms = 35.0
v0 = v0_kms * KM_S_TO_KPC_GYR
Rcore = 0.8  # kpc
vphi = v0 * R / np.sqrt(R**2 + Rcore**2)

# convert vphi to vx, vy (avoid divide by zero at center)
eps = 1e-12
vx = -vphi * (y / (R + eps))
vy =  vphi * (x / (R + eps))
vz = np.zeros_like(vx)

# gas temperature/sound speed
cs_kms = 10.0
cs = cs_kms * KM_S_TO_KPC_GYR

gas = NBodyGas(
    simulation=sim,
    rho=rho,
    vx=vx,
    vy=vy,
    vz=vz,
    cs=cs,
    gamma=5/3,
    tcool=None,        # start without cooling for stability; add later
    e_floor=1e-10,
    rho_floor=rho_floor,
    cfl=0.4,
    max_substeps=200
)
#sim.add_baryons(gas)

# -------------------------
# ULDM soliton (centered)
# -------------------------
wave_vector = Wave_vector_class(
    packet_type="resources/solitons/ModoBH.dat",
    means=[0.0, 0.0, 0.0],
    st_deviations=[0.8, 0.8, 0.8],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0.0, 0.0, 0.0],
    spin=0,
    desired_soliton_mass=18684320,  # Msun
)
sim.add_wave_vector(wave_vector)

# -------------------------
# Run
# -------------------------
sim.evolve(save_every=50)
