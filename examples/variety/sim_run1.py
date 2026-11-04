import matplotlib.pyplot as plt

from resources.Classes.Wave_function_class import *
from resources.Functions.system_fucntions import *
from matplotlib.colors import LogNorm
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody

import numpy as np

# ---- helpers ---------------------------------------------------------------

KM_S_TO_KPC_GYR = 1.0227121650537077  # 1 km/s ≈ 1.0227 kpc/Gyr

import numpy as np

FILE = "resources/solitons/JAXION.bin"
Nmesh = 32

raw = np.fromfile(FILE, dtype=np.float64, count=2*Nmesh**3)
raw = raw.reshape((Nmesh, Nmesh, Nmesh, 2))

psi = raw[...,0] + 1j*raw[...,1]
psi = psi / np.sqrt(1e9)
Bs = 6.0
boundaries = [(-Bs/2, Bs/2)]*3
total_time = 10.0  # Gyr

import cupy as cp
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_function_class import Wave_function

m_scalar_eV = 1e-22
HBAR_TILDE = 19.60695038367394

MBH = 1e6
'''
boundaries = [[-25, 25], [-25, 25], [-25, 25]]

sim = Simulation_Class(
    dim=3,
    boundaries=boundaries,
    N=320,
    total_time=5.0,
    h=1e-3,
    order_of_evolution=2,
    use_gravity=True,
    static_potential=None,
    save_max_vals=True,
    self_int=False,
    use_sponge=True,
    m_s=m_scalar_eV,
)
wave_vector = Wave_vector_class(
    packet_type="resources/solitons/GroundState(1).dat",
    means=[0.0, 0.0, 0.0],
    st_deviations=[1.2, 1.2, 1.2],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0.0, 0.0, 0.0],
    spin=0,
    desired_soliton_mass=331812960,
)


print(wave_vector.wave_vector[0].soliton_radius)
print(wave_vector.wave_vector[0].soliton_mass_radius)
Msol = wave_vector.wave_vector[0].soliton_mass
Msol_dyn0 = wave_vector.wave_vector[0].soliton_dynamic_mass
sim.add_wave_vector(wave_vector)


sim.evolve(save_every=250)

'''
import matplotlib.pyplot as plt
import numpy as np

# Data
benchmarks = ["ULDM only", "N-body only", "Gas only"]
phoebe = [60.24, 17.21, 12.74]
rtx_3070 = [14.79, 5.83, 3.75]
quadro_p4000 = [9.01, 2.59, 1.22]

x = np.arange(len(benchmarks))
width = 0.25

# Vytvoření grafu s jemnějším pozadím
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
#fig.patch.set_facecolor('#FAFAFA')
#ax.set_facecolor('#FAFAFA')

# Moderní barevná paleta
colors = ['#839705', '#008BFF', '#FA8112']

# Vykreslení sloupců s jemným bílým ohraničením pro čistší vzhled
bars1 = ax.bar(x - width, phoebe, width=width, color=colors[0], label="Phoebe", edgecolor='white', linewidth=1.2)
bars2 = ax.bar(x, rtx_3070, width=width, color=colors[1], label="RTX 3070", edgecolor='white', linewidth=1.2)
bars3 = ax.bar(x + width, quadro_p4000, width=width, color=colors[2], label="Quadro P4000", edgecolor='white', linewidth=1.2)

# Odstranění zbytečných os (Chart junk)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.spines['bottom'].set_color('#CCCCCC')

# Odstranění "čárek" u os pro čistší design
ax.tick_params(bottom=False, left=False)

# Přidání přesných hodnot přímo nad sloupce
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 body vertikální posun
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, color='#333333', fontweight='500')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Nastavení os a popisků
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Performance [Steps/s]", fontsize=14, fontweight='bold', labelpad=15, )
plt.yticks(fontsize=13)

ax.set_axisbelow(True)
#ax.yaxis.grid(True, color='#E0E0E0', linestyle='-', linewidth=1)

# Přidání nadpisu
#plt.title("Porovnání výpočetního výkonu", fontsize=16, fontweight='bold', color='#1A1A1A', pad=30, loc='left')

# Vylepšení legendy (vodorovně nad grafem bez rámečku)
ax.legend(loc='upper right', frameon=False, fontsize=13, ncol=3, bbox_to_anchor=(1, 1.1))

plt.tight_layout()
plt.savefig("plots/performance_overview.png", dpi=300, bbox_inches='tight')


import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data
# -------------------------
gpus = ["Phoebe", "RTX 3070", "Quadro P4000"]
practical = [512, 320, 256]
absolute = [640, 320, 320]

x = np.arange(len(gpus))
width = 0.25  # Stejná šířka jako v prvním grafu

# Vytvoření grafu se stejným pozadím a rozměry
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
#fig.patch.set_facecolor('#FAFAFA')
#ax.set_facecolor('#FAFAFA')

# Barvy - z původního kódu
colors = ['#839705', '#008BFF']

# Vykreslení sloupců s jemným bílým ohraničením (stejně jako v grafu 1)
bars1 = ax.bar(x - width/2, practical, width=width, color=colors[0], label="Practical limit", edgecolor='white', linewidth=1.2)
bars2 = ax.bar(x + width/2, absolute, width=width, color=colors[1], label="Absolute limit", edgecolor='white', linewidth=1.2)

# Odstranění zbytečných os (Chart junk)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Odstranění "čárek" u os pro čistší design
ax.tick_params(bottom=False, left=False)

# Přidání přesných hodnot přímo nad sloupce
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', # Odstraněno .2f, protože to jsou celá čísla
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 body vertikální posun
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, color='#333333', fontweight='500')

add_labels(bars1)
add_labels(bars2)

# Nastavení os a popisků
ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=14, fontweight='bold', color='#333333')

# Použití LaTeXu pro N
ax.set_ylabel("Maximum grid resolution $N$", fontsize=14, fontweight='bold', labelpad=15)
plt.yticks(fontsize=13)

# Úprava rozsahu osy Y, aby se vešly popisky nad nejvyšší sloupec
ax.set_ylim(0, 720)
ax.set_axisbelow(True)

# Vylepšení legendy (vodorovně nad grafem bez rámečku, stejná pozice)
ax.legend(loc='upper right', frameon=False, fontsize=13, ncol=2, bbox_to_anchor=(1, 1.1))

plt.tight_layout()
plt.savefig("plots/resolution_limits.png", dpi=300, bbox_inches='tight')
plt.show()

"""
def make_gaussian_gas_density(sim, Mgas_target, sigma_kpc=1.5, floor=1e-30, center=(0.0, 0.0, 0.0),
                             background_frac=1e-6):

    x, y, z = sim.grids
    x = np.array(x); y = np.array(y); z = np.array(z)

    cx, cy, cz = center
    r2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2

    # Shape: small background + Gaussian bump
    # We'll normalize afterwards to target mass anyway.
    rho_shape = background_frac + np.exp(-0.5 * r2 / (sigma_kpc**2))
    rho_shape = np.maximum(rho_shape, floor)

    # Normalize to target mass
    M_current = rho_shape.sum() * sim.dV
    rho = rho_shape * (Mgas_target / M_current)
    return rho
#TODO: přepiš to na gpu

def add_solid_body_rotation(sim, rho, vphi0=30.0, r0=3.0, center=(0.0, 0.0, 0.0)):

    x, y, z = sim.grids
    x = np.array(x); y = np.array(y); z = np.array(z)

    cx, cy, cz = center
    X = x - cx
    Y = y - cy
    r = np.sqrt(X**2 + Y**2) + 1e-30

    vphi = vphi0 * (r / np.sqrt(r**2 + r0**2))
    vx = -vphi * (Y / r)
    vy =  vphi * (X / r)
    vz = np.zeros_like(vx)
    return vx, vy, vz

def center_cell_mass(sim, rho):

    return float(rho.max() * sim.dV)



sim = Simulation_Class(
    dim=3,
    boundaries=[(-3, 3), (-3, 3), (-3, 3)],   # kpc
    N=32,
    total_time=3.0,                                  # Gyr
    h=9e-4,                                          # Gyr  (~0.5 Myr)
    order_of_evolution=2,
    use_gravity=True,
    static_potential=None,
    save_max_vals=True,
    self_int=False,
    use_sponge=False,

)

# --- ULDM (soliton core) ---
wave_vector = Wave_vector_class(
    packet_type="resources/solitons/GroundState(1).dat",
    means=[0.0, 0.0, 0.0],
    st_deviations=[1.0, 1.0, 1.0],                   # kpc
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0.0, 0.0, 0.0],
    spin=0,
    desired_soliton_mass=5.0e7                       # Msun
)

# --- Stars (bulge) ---
bulge = Baryons(
    simulation=sim,
    N_particles=int(2e5),
    total_mass=5.0e8,                                # Msun
    init_profile="hernquist",
    scale_radius=2.0,                                # kpc
    truncation_radius=10.0,                           # kpc
    center=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    vel_sigma=50.0 * KM_S_TO_KPC_GYR                  # ~50 km/s
)
#sim.add_baryons(bulge)

seed_bh = SinkNBody(
    simulation=sim,
    initial_positions=[(0.0, 0.0, 0.0)],
    initial_velocities=[(0.0, 0.0, 0.0)],
    initial_masses=[5.0e5],                           # Msun (seed)
    N_sinks=1,

)
#sim.add_baryons(seed_bh)

# --- Gas: Gaussian + mild rotation ---
Mgas_target = 2.0e7                                  # Msun (dost na kolaps, ale ne “explosive”)
rho = make_gaussian_gas_density(sim, Mgas_target=Mgas_target, sigma_kpc=2.5, background_frac=1e-8)

vx, vy, vz = add_solid_body_rotation(
    sim, rho,
    vphi0=60.0 * KM_S_TO_KPC_GYR,                     # ~60 km/s max (mírná rotace)
    r0=3.0,                                           # kpc (růst + saturace)
    center=(0.0, 0.0, 0.0)
)

psi = np.fromfile("resources/solitons/JAXION.bin", dtype=np.complex128).reshape((32,32,32))
rho = np.abs(psi)**2

rho = rho + 1e-8 * rho.mean()
rho = np.maximum(rho, 1e-30)

M_target = 2e15  # Msun (nebo podle kolegy)

rho *= (M_target / (rho.sum() * sim.dV))

# background: třeba 1e-8 z průměru (laditelné)
rho_mean = rho.mean()
rho = rho + 1e-8 * rho_mean

# tvrdý floor
rho = np.maximum(rho, 1e-30)

print("dx [kpc]:", sim.dx)
print("dV [kpc^3]:", sim.dV)
print("Gas total mass [Msun]:", float(rho.sum() * sim.dV))
print("Max gas cell mass [Msun/cell]:", center_cell_mass(sim, rho))
print("Min gas density [Msun/kpc^3]:", float(rho.min()))
print("Max gas density [Msun/kpc^3]:", float(rho.max()))
print("Density field finite:", np.all(np.isfinite(rho)))


gas = NBodyGas(
    simulation=sim,
    rho=rho,

    cs=1 * KM_S_TO_KPC_GYR,
    gamma=5/3,
    tcool=None,
    e_floor=1e-30
,
    cfl=0.25,
    max_substeps=40000
)
sim.add_baryons(gas)


# --- quick sanity prints ---
print("dx [kpc]:", sim.dx)
print("dV [kpc^3]:", sim.dV)
print("Gas total mass [Msun]:", float(gas.rho.sum() * sim.dV))
print("Max gas cell mass [Msun/cell]:", center_cell_mass(sim, gas.rho))


#sim.add_wave_vector(wave_vector)

sim.evolve(save_every=100)



   sink_formation={
        "source": "gas",
        "consecutive_steps": 5,
        "check_interval": 5,

        "enable_gas_sinks": True,
        "gas_mode": "truelove",
        "gas_NJ": 4,

        "gas_cs_floor": 0.03,
        "gas_cs_truelove": 0.5, # change this to change trulove value

        "gas_consecutive_steps": 5,
        "gas_check_interval": 5,
        "gas_r_acc_cells": 2,
        "merge_r_cells": 2,
        "merge_interval": 20,
    }"""