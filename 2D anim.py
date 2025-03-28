import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.constants import gravitational_constant

from resources.Classes.Wave_function_class import Wave_function
from resources.Functions.Schrodinger_eq_functions import energy_nd, quadratic_potential
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Classes.Simulation_Class import Simulation_class

N = 1024
# Initialize the 2D system


sim = Simulation_class(
    dim=2,
    boundaries=[(-10, 10), (-10, 10)],
    N=N,
    total_time=200,
    h=0.01,
    static_potential=None,
    use_gravity=True,
)

st_dev = 0.5

vlna = Wave_function(
    simulation=sim,
    mass=1,
    packet_type="gaussian",
    means=[3, 3],
    st_deviations=[st_dev,st_dev],
    momenta=[0, 0],

)
'''
vlna3 = Wave_function(
    simulation=sim,
    mass=50,
    packet_type="gaussian",
    means=[1, -1],
    st_deviations=[st_dev,st_dev],
    momenta=[0, 0],

)

vlna4 = Wave_function(
    simulation=sim,
    mass=50,
    packet_type="gaussian",
    means=[-1, 1],
    st_deviations=[st_dev,st_dev],
    momenta=[0, 0],

)
'''
vlna2 = Wave_function(
    simulation=sim,
    mass=10,
    packet_type="gaussian",
    means=[-3, -3],
    st_deviations=[st_dev,st_dev],
    momenta=[0, 0],

)

sim.add_wave_function(vlna)
sim.add_wave_function(vlna2)

sim.evolve(save_every=2000)


'''
x_vals = np.linspace(-10, 10, N)
y_vals = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x_vals, y_vals)
# Create the plots
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title('X-axis')
plt.legend(['Analytical real', 'Numerical real', 'Analytical imag', 'Numerical imag'], loc='upper right')

plt.subplot(1, 2, 2)
plt.title('Y-axis')
plt.legend(['Analytical real', 'Numerical real', 'Analytical imag', 'Numerical imag'], loc='upper right')

for time in sim.accessible_times:

    an_psi = cp.asnumpy(cp.abs(vlna.psi  * cp.exp(-1j * energy_nd([0, 0], omega=1, hbar=1) * time))**2)
    an_psi_real = cp.asnumpy(cp.real(vlna.psi  * cp.exp(-1j * energy_nd([0, 0], omega=1, hbar=1) * time)))
    an_psi_imag = cp.asnumpy(cp.imag(vlna.psi  * cp.exp(-1j * energy_nd([0, 0], omega=1, hbar=1) * time)))

    num_psi = sim.get_wave_function_at_time(time)
    num_psi = cp.asnumpy(num_psi)
    num_psi_real = cp.asnumpy(cp.real(num_psi))
    num_psi_imag = cp.asnumpy(cp.imag(num_psi))

    # Plot x-axis
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, an_psi_real[:, x_vals.shape[0]//2], color='red', alpha=0.5)
    plt.plot(x_vals, num_psi_real[:, x_vals.shape[0]//2], color='green', linestyle='--', alpha=0.5)
    plt.plot(x_vals, an_psi_imag[:, x_vals.shape[0]//2], color='blue', alpha=0.5)
    plt.plot(x_vals, num_psi_imag[:, x_vals.shape[0]//2], color='#FFC0CB', linestyle='--', alpha=0.5)

    # Plot y-axis
    plt.subplot(1, 2, 2)
    plt.plot(y_vals, an_psi_real[x_vals.shape[0]//2, :], color='red', alpha=0.5)
    plt.plot(y_vals, num_psi_real[x_vals.shape[0]//2, :], color='green', linestyle='--', alpha=0.5)
    plt.plot(y_vals, an_psi_imag[x_vals.shape[0]//2, :], color='blue', alpha=0.5)
    plt.plot(y_vals, num_psi_imag[x_vals.shape[0]//2, :], color='#FFC0CB', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
'''


#TODO is implement the minimal time step requirement    eq 21 in Volker paper with a = 1


#uncomment this section for plotting wavefunction snapshots when not treated individually

wave_snapshots = [cp.asnumpy(sim.get_wave_function_at_time(time)) for time in sim.accessible_times]

# Directly create x and y using vlna.boundaries and vlna.N
x = np.linspace(vlna.boundaries[0][0], vlna.boundaries[0][1], vlna.N)
y = np.linspace(vlna.boundaries[1][0], vlna.boundaries[1][1], vlna.N)
xy = np.meshgrid(x,y, indexing='ij')

# Calculate the number of rows and columns for the subplots
n_plots = len(wave_snapshots)
n_cols = int(np.ceil(np.sqrt(n_plots)))
n_rows = int(np.ceil(n_plots / n_cols))

# Plot the wavefunction at the selected time steps
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*6))
titles = [f"Wavefunction at Time {time}" for time in sim.accessible_times]

for ax, wave, title in zip(axes.flat, wave_snapshots, titles):

    im = ax.contourf(xy[0],xy[1], np.abs(wave), shading='auto', cmap='viridis', levels = 100)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, orientation="vertical", label="|ψ|")

# Hide any unused subplots
for ax in axes.flat[len(wave_snapshots):]:
    ax.axis('off')

plt.tight_layout()
plt.show()


'''
def plot_combined_wave_functions(sim, combine_method='sum', cmap='viridis'):
    """
    Plot all wave functions combined into a single plot for each timestep.

    Parameters:
        sim: Simulation object with evolved wave functions
        combine_method: How to combine the wave functions ('sum' or 'density_sum')
                       'sum': Add the wave functions directly (ψ₁ + ψ₂ + ...)
                       'density_sum': Add the densities (|ψ₁|² + |ψ₂|² + ...)
        cmap: Colormap to use for the plots
    """
    # Get the number of time steps to plot
    n_times = len(sim.accessible_times)

    # Calculate the number of rows and columns for the subplots
    n_cols = min(3, n_times)  # Max 3 columns
    n_rows = int(np.ceil(n_times / n_cols))

    # Get grid information from the simulation
    x = np.linspace(sim.boundaries[0][0], sim.boundaries[0][1], sim.N)
    y = np.linspace(sim.boundaries[1][0], sim.boundaries[1][1], sim.N)
    xy = np.meshgrid(x, y, indexing='ij')

    # Create figure and axes for the subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False)

    # Maximum value for consistent color scale
    max_val = 0

    # Get the combined wave functions at all times
    combined_waves = []
    for time in sim.accessible_times:
        # Get all wave functions at this time
        all_waves = sim.get_wave_function_at_time(time)

        if combine_method == 'sum':
            # Sum the wave functions directly
            combined = sum(cp.asnumpy(wave) for wave in all_waves)
            combined_waves.append(combined)
            # Update max value based on density
            max_val = max(max_val, np.max(np.abs(combined) ** 2))
        else:  # density_sum
            # Sum the densities
            combined_density = sum(cp.asnumpy(np.abs(wave) ** 2) for wave in all_waves)
            combined_waves.append(combined_density)
            # Update max value
            max_val = max(max_val, np.max(combined_density))

    # Plot each timestep
    for idx, (time, wave) in enumerate(zip(sim.accessible_times, combined_waves)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if combine_method == 'sum':
            # Plot the density of the combined wave function
            density = np.abs(wave) ** 2
            title = f"Combined Wave Functions (t = {time:.2f})"
            label = "|ψ₁ + ψ₂ + ...|²"
        else:
            # Plot the sum of densities
            density = wave  # already the sum of densities
            title = f"Sum of Densities (t = {time:.2f})"
            label = "|ψ₁|² + |ψ₂|² + ..."

        im = ax.contourf(xy[0], xy[1], density,
                         shading='auto', cmap=cmap, levels=100,
                         vmin=0, vmax=max_val)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, orientation="vertical", label=label)

    # Hide any unused subplots
    for ax in axes.flat[n_times:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return fig



plot_combined_wave_functions(sim, combine_method='sum')  # Plot sum of wave functions
'''



'''
fig, ax = plt.subplots()

# Convert the initial wavefunction to a NumPy array and take its magnitude
initial_wave = np.abs(vlna.wave_values[0][0].get())

im = ax.pcolormesh(X, Y, initial_wave, shading='auto', cmap='viridis')


def animate(i):
    # Convert the current wavefunction to a NumPy array and take its magnitude
    wave = np.abs(vlna.wave_values[i][0].get())

    im.set_array(wave.ravel())
    return im,


ani = FuncAnimation(fig, animate, frames=len(vlna.wave_values), interval=50, blit=False)
ani.save("2devolution.mp4")
plt.show()



'''
