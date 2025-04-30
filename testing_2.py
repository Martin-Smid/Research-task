import matplotlib.pyplot as plt

from resources.Classes.Wave_function_class import *
from resources.Functions.system_fucntions import *
from matplotlib.colors import LogNorm
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class

sim = Simulation_Class(
    dim=3,                             # 2D simulation
    boundaries=[(-20, 20),(-20, 20),(-20, 20)], # Spatial boundaries
    N=128,                             # Grid resolution
    total_time=1,                   # Total simulation time
    h=0.005,                            # Time step
    order_of_evolution=4,
    use_gravity=True , # Enable gravitational effects
    static_potential=None,
    save_max_vals=False,
)

vlna = Wave_function(
    packet_type="/home/martin/Downloads/GroundState(1).dat",
    means=[-0,5,0],
    st_deviations=[0.5,0.5,0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[-5,0,0],
)



vlna2 = Wave_function(
    packet_type="/home/martin/Downloads/GroundState(1).dat",
    means=[-0,-5,0],
    st_deviations=[0.5,0.5,0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[4,0,0],
)
Wave_vector1 = Wave_vector_class([vlna], spin=1)



sim.add_wave_function(wave_vector=Wave_vector1.wave_vector)


sim.evolve(save_every=25)

x_index = (sim.grids[0].shape[0] // 2)
y_index = (sim.grids[1].shape[0] //2)
z_index = (sim.grids[2].shape[0] //2)
# For the YZ plane plotting
y_mesh_2d, z_mesh_2d = np.meshgrid(sim.grids[1][0,:,0].get(), sim.grids[2][0,0,:].get())
x_mesh_2d, z_mesh_2d = np.meshgrid(sim.grids[0][:,0,0].get(), sim.grids[2][0,0,:].get())
plt.figure(figsize=(8, 6))
x_mesh_2d, y_mesh_2d = np.meshgrid(sim.grids[0][:,0,0].get(), sim.grids[1][0,:,0].get())
for time in sim.accessible_times:
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time)) ** 2)

    # Take the middle x-slice
    wave_slice = wave_values[:, :, z_index]
    levels = np.logspace(np.log10(wave_values[wave_values > 0].min()), np.log10(wave_values.max()), 128)
    plt.contourf(x_mesh_2d, y_mesh_2d, cp.asnumpy(wave_slice).T,
                 origin="lower", levels=levels, cmap="inferno", norm=LogNorm())
    plt.colorbar(label="|ψ|²", format="%.2e")
    plt.title(f"Wavefunction Probability Density at Time {time}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()