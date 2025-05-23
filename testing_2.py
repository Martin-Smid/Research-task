import matplotlib.pyplot as plt
from resources.Classes.Wave_function_class import *
from resources.Functions.system_fucntions import *
from matplotlib.colors import LogNorm
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from datetime import datetime
import os
import numpy as np

np.random.seed(1)

sim = Simulation_Class(
    dim=3,                             # 2D simulation
    boundaries=[(-10, 10),(-10, 10),(-10,10)], # Spatial boundaries
    N=32 ,                             # Grid resolution
    total_time=100,                   # Total simulation time
    h=0.01,                            # Time step
    order_of_evolution=2,
    use_gravity=True , # Enable gravitational effects
    static_potential=None,
    save_max_vals=True,
)

wave = Wave_function(
    packet_type="/home/martin/Downloads/Modo-1e-80.dat",
    #packet_type="/home/martin/Downloads/GroundState(1).dat",
    means=[0,0,0],
    st_deviations=[0.5, 0.5, 0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0, 0, 0],
)

w_vect = Wave_vector_class(wave_function=wave,spin=2)

sim.add_wave_function(w_vect.wave_vector)
sim.evolve(save_every=500)

'''
centers = np.zeros((10, 3))
centers[:, 0] = np.random.uniform(-35, 35, 10)
centers[:, 1] = np.random.uniform(-35, 35, 10)
centers_list = [list(row) for row in centers]

waves = []
for i in range(5):
    vlna = Wave_function(
        packet_type="/home/martin/Downloads/GroundState(1).dat",
        means=centers_list[i],
        st_deviations=[0.5, 0.5, 0.5],
        simulation=sim,
        mass=1,
        omega=1,
        momenta=[0, 0, 0],
    )
    Wave_vector = Wave_vector_class([vlna], spin=2)
    waves.append(Wave_vector)


print(waves)

for wave in waves:
    sim.add_wave_function(wave_vector=wave.wave_vector)
'''


current_date = datetime.now().strftime("%y_%d_%H")
x_index = (sim.grids[0].shape[0] // 2)
y_index = (sim.grids[1].shape[0] //2)
z_index = (sim.grids[2].shape[0] //2)
# For the YZ plane plotting
y_mesh_2d, z_mesh_2d = np.meshgrid(sim.grids[1][0,:,0].get(), sim.grids[2][0,0,:].get())
x_mesh_2d, z_mesh_2d = np.meshgrid(sim.grids[0][:,0,0].get(), sim.grids[2][0,0,:].get())
plt.figure(figsize=(8, 6))
x_mesh_2d, y_mesh_2d = np.meshgrid(sim.grids[0][:,0,0].get(), sim.grids[1][0,:,0].get())
save_dir = f"resources/data/sim_N{sim.N}_{current_date}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # This will create all intermediate directories as needed

for time in sim.accessible_times:
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time)) ** 2)

    # Take the middle x-slice
    wave_slice = wave_values[:, :, z_index]
    levels = np.logspace(np.log10(wave_values[wave_values > 0].min()), np.log10(wave_values.max()), 128)
    plt.contourf(x_mesh_2d, y_mesh_2d, cp.asnumpy(wave_slice).T,
                 origin="lower", levels=levels, cmap="inferno",norm=LogNorm())
    plt.colorbar(label="|ψ|²", format="%.2e")
    plt.title(f"Wavefunction Probability Density at Time {time}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig(f"{save_dir}/timestep_{time}.jpg")
    plt.close()