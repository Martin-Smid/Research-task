import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from resources.Classes.Wave_function_class import Wave_function
from resources.Functions.Schrodinger_eq_functions import energy_nd, quadratic_potential
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Classes.Wave_function_class import *
from resources.system_fucntions import *

from resources.Classes.Simulation_Class import Simulation_class

sim = Simulation_class(
    dim=3,                             # 2D simulation
    boundaries=[(-20, 20),(-20, 20),(-20, 20)], # Spatial boundaries
    N=256,                             # Grid resolution
    total_time=10.0,                   # Total simulation time
    h=0.001,                            # Time step
    use_gravity=True , # Enable gravitational effects
    static_potential=None,
)

vlna = Wave_function(
    packet_type="/home/martin/Downloads/GroundState(1).dat",
    means=[0,0,0],
    st_deviations=[0.5,0.5,0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0,0,0],
)

sim.add_wave_function(vlna)

sim.evolve(save_every=2500)

'''1D
plt.figure()
for time in sim.accessible_times:
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time))**2)
    plt.plot(wave_values, label=f"Time: {time}")
plt.legend()
plt.show()
'''


# Choose a slice in the Z direction (middle of the grid)
z_index = sim.grids[2].shape[0] // 2  # Middle z-plane

plt.figure(figsize=(8, 6))
for time in sim.accessible_times:
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time)) ** 2)

    # Take the middle z-slice
    wave_slice = wave_values[:, :, z_index]

    plt.imshow(cp.asnumpy(wave_slice), extent=[sim.grids[0].min().get(), sim.grids[0].max().get(),
                                               sim.grids[1].min().get(), sim.grids[1].max().get()],
               origin="lower", cmap="inferno")
    plt.colorbar(label="|ψ|²")
    plt.title(f"Wavefunction Probability Density at Time {time}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

'''
rho = cp.abs(sim.get_wave_function_at_time(0)) ** 2


rho_xy = cp.asnumpy(rho.sum(axis=2))  # Summing over Z → XY plane
rho_zx = cp.asnumpy(rho.sum(axis=1))  # Summing over Y → ZX plane
rho_yz = cp.asnumpy(rho.sum(axis=0))  # Summing over X → YZ plane


x_min, x_max = sim.grids[0].min().get(), sim.grids[0].max().get()
y_min, y_max = sim.grids[1].min().get(), sim.grids[1].max().get()
z_min, z_max = sim.grids[2].min().get(), sim.grids[2].max().get()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))


im1 = axes[0].imshow(rho_xy, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="inferno")
axes[0].set_title("XY Projection ")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
plt.colorbar(im1, ax=axes[0])

# ZX Projection
im2 = axes[1].imshow(rho_zx, extent=[z_min, z_max, x_min, x_max], origin="lower", cmap="inferno")
axes[1].set_title("ZX Projection ")
axes[1].set_xlabel("Z")
axes[1].set_ylabel("X")
plt.colorbar(im2, ax=axes[1])

# YZ Projection
im3 = axes[2].imshow(rho_yz, extent=[z_min, z_max, y_min, y_max], origin="lower", cmap="inferno")
axes[2].set_title("YZ Projection")
axes[2].set_xlabel("Z")
axes[2].set_ylabel("Y")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()
'''

#TODO make it so that you can create different dim wave from the simulation, maybe make dim a wave_function class attribute and if not given take it from sim