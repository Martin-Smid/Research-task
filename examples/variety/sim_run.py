import matplotlib.pyplot as plt

from resources.Classes.Wave_function_class import *
from resources.Functions.system_fucntions import *
from matplotlib.colors import LogNorm
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas

import numpy as np

# ---- helpers ---------------------------------------------------------------

KM_S_TO_KPC_GYR = 1.0227121650537077  # 1 km/s ≈ 1.0227 kpc/Gyr

def make_gaussian_gas_density(sim, Mgas_target, sigma_kpc=1.5, floor=1e-30, center=(0.0, 0.0, 0.0),
                             background_frac=1e-6):
    """
    Build a smooth, centrally-peaked gas density field on the sim grid and normalize to Mgas_target.
    - background_frac controls a tiny uniform background mass fraction (avoid true vacuum).
    """
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
    """
    Optional: add a simple rotating velocity field around z-axis.
    v_phi(r) = vphi0 * (r / sqrt(r^2 + r0^2))  (rises then flattens).
    Returns vx, vy, vz arrays in kpc/Gyr.
    """
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
    """Mass in the densest cell (useful sanity check vs Truelove threshold in Msun/cell)."""
    return float(rho.max() * sim.dV)

sim = Simulation_Class(
    dim=1,
    boundaries=[(-20.0, 20.0)],
    N=256,
    total_time=0.1,
    h=1e-3,
    use_gravity=True,
)


gas = NBodyGas(sim, total_mass=1.0, cs=0.1, gamma=5/3)

sim.add_baryons(gas)
sim.evolve(save_every=10)
