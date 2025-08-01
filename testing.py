import matplotlib.pyplot as plt

from resources.Classes.Wave_function_class import *
from resources.Functions.system_fucntions import *
from matplotlib.colors import LogNorm
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class

sim = Simulation_Class(
    dim=3,                             # 2D simulation
    boundaries=[(-20, 20),(-20, 20),(-20, 20)], # Spatial boundaries
    N=64,                             # Grid resolution
    total_time=33,                   # Total simulation time
    h=0.01,                            # Time step
    order_of_evolution=2,
    use_gravity=True , # Enable gravitational effects
    static_potential=gravity_potential,
    save_max_vals=False,
    self_int=False,
)



wave_vector = Wave_vector_class(
    packet_type="resources/solitons/GroundState(1).dat",
    means=[5, 0, 0],
    st_deviations=[0.5, 0.5, 0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0, 0.947, 0],
    spin=0,
    desired_soliton_mass=53090068

)


sim.add_wave_vector(wave_vector)
#sim.add_wave_function(vlna2)
#sim.add_wave_function(vlna3)


sim.evolve(save_every=100 )

'''1D
plt.figure()
for time in sim.accessible_times:
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time))**2)
    plt.plot(wave_values, label=f"Time: {time}")
plt.legend()
plt.show()
'''

x_mesh = cp.asnumpy(sim.grids[0][:,:,0])
y_mesh = cp.asnumpy(sim.grids[1][:,:,0])
z_mesh = cp.asnumpy(sim.grids[2][0,0,:])



'''
# Choose a slice in the Z direction (middle of the grid)
z_index = (sim.grids[2].shape[0] // 2 )  + 10# Middle z-plane

plt.figure(figsize=(8, 6))
for time in sim.accessible_times:
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time)) ** 2)

    # Take the middle z-slice
    wave_slice = wave_values[:, :, z_index]
    levels = np.logspace(np.log10(wave_values[wave_values > 0].min()),np.log10(wave_values.max()), 128)
    plt.contourf(x_mesh,y_mesh,cp.asnumpy(wave_slice),
               origin="lower", levels=levels,cmap="inferno", norm=LogNorm())
    plt.colorbar(label="|ψ|²",format = "%.2e")
    plt.title(f"Wavefunction Probability Density at Time {time}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

'''
x_index = (sim.grids[0].shape[0] // 2)
y_index = (sim.grids[1].shape[0] //2)
z_index = (sim.grids[2].shape[0] //2)
# For the YZ plane plotting
y_mesh_2d, z_mesh_2d = np.meshgrid(sim.grids[1][0,:,0], sim.grids[2][0,0,:])
x_mesh_2d, z_mesh_2d = np.meshgrid(sim.grids[0][:,0,0], sim.grids[2][0,0,:])
plt.figure(figsize=(8, 6))
x_mesh_2d, y_mesh_2d = np.meshgrid(sim.grids[0][:,0,0], sim.grids[1][0,:,0])
for time in sim.accessible_times:

    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time)) ** 2)

    # Take the middle x-slice
    wave_slice = wave_values[:, :, z_index]
    levels = np.logspace(np.log10(wave_values[wave_values > 0].min()), np.log10(wave_values.max()), 128)
    plt.contourf(x_mesh_2d, y_mesh_2d, cp.asnumpy(wave_slice).T,
                 origin="lower", levels=levels, cmap="viridis", norm=LogNorm())
    plt.colorbar(label="|ψ|²", format="%.2e")
    plt.title(f"Wavefunction Probability Density at Time {time}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()



'''

def plot_wave_slice(sim, time, axis="z", index=None):
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time)) ** 2)
    levels = np.logspace(np.log10(wave_values[wave_values > 0].min()), np.log10(wave_values.max()), 128)

    if axis == "z":
        index = index or sim.grids[2].shape[0] // 2
        x_mesh = cp.asnumpy(sim.grids[0][:, :, index])
        y_mesh = cp.asnumpy(sim.grids[1][:, :, index])
        wave_slice = wave_values[:, :, index]
        xlabel, ylabel = "X", "Y"

    elif axis == "x":
        index = index or sim.grids[0].shape[0] // 2
        y_mesh = cp.asnumpy(sim.grids[1][index, :, :])
        z_mesh = cp.asnumpy(sim.grids[2][index, :, :])
        wave_slice = wave_values[index, :, :]
        x_mesh, y_mesh = y_mesh, z_mesh
        xlabel, ylabel = "Y", "Z"

    elif axis == "y":
        index = index or sim.grids[1].shape[1] // 2
        x_mesh = cp.asnumpy(sim.grids[0][:, index, :])
        z_mesh = cp.asnumpy(sim.grids[2][:, index, :])
        wave_slice = wave_values[:, index, :]
        x_mesh, y_mesh = x_mesh, z_mesh
        xlabel, ylabel = "X", "Z"

    else:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'")

    plt.figure(figsize=(8, 6))
    plt.contourf(x_mesh, y_mesh, wave_slice, origin="lower", levels=levels, cmap="inferno", norm=LogNorm())
    plt.colorbar(label="|ψ|²", format="%.2e")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Wavefunction Probability Density at Time {time} (Slice along {axis.upper()}={index})")
    plt.grid()
    plt.show()

for time in sim.accessible_times:
    plot_wave_slice(sim, time=time, axis="x")
'''
#TODO make it so that you can create different dim wave from the simulation, maybe make dim a wave_function class attribute and if not given take it from sim