import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from resources.Classes.Wave_function_class import Wave_function
from resources.Functions.Schrodinger_eq_functions import energy_nd, quadratic_potential
from resources.Classes.Simulation_Class import Simulation_Class

# Initialize constants
N = 256  # Reduced for 3D computation
# Initialize the 3D system
x_vals = np.linspace(-10, 10, N)
y_vals = np.linspace(-10, 10, N)
z_vals = np.linspace(-10, 10, N)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

sim = Simulation_Class(
    dim=3,
    boundaries=[(-10, 10), (-10, 10),(-10, 10)],
    N=N,
    total_time=5,
    h=0.01,
    static_potential=quadratic_potential,
    use_gravity=False,
)

vlna = Wave_function(
    simulation=sim,
    mass=1,
    packet_type="LHO",
    means=[0, 0.0,0],
    st_deviations=[0.1,0.1,0.1],
    momenta=[0, 0,0],
      # Quadratic potential for harmonic evolution
)
# Iterate over reduced time steps for better visualization

sim.add_wave_function(vlna)


sim.evolve(save_every=50)


x_vals = np.linspace(-10, 10, N)
y_vals = np.linspace(-10, 10, N)
z_vals = np.linspace(-10, 10, N)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

# Create the plots
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.title('X-axis')
plt.legend(['Analytical real', 'Numerical real', 'Analytical imag', 'Numerical imag'], loc='upper right')

plt.subplot(1, 3, 2)
plt.title('Y-axis')
plt.legend(['Analytical real', 'Numerical real', 'Analytical imag', 'Numerical imag'], loc='upper right')

plt.subplot(1, 3, 3)
plt.title('Z-axis')
plt.legend(['Analytical real', 'Numerical real', 'Analytical imag', 'Numerical imag'], loc='upper right')

for time in sim.accessible_times:

    an_psi = cp.asnumpy(cp.abs(vlna.psi  * cp.exp(-1j * energy_nd([0, 0, 0], omega=1, hbar=1) * time))**2)
    an_psi_real = cp.asnumpy(cp.real(vlna.psi  * cp.exp(-1j * energy_nd([0, 0, 0], omega=1, hbar=1) * time)))
    an_psi_imag = cp.asnumpy(cp.imag(vlna.psi  * cp.exp(-1j * energy_nd([0, 0, 0], omega=1, hbar=1) * time)))

    num_psi = sim.get_wave_function_at_time(time)
    num_psi = cp.asnumpy(num_psi)
    num_psi_real = cp.asnumpy(cp.real(num_psi))
    num_psi_imag = cp.asnumpy(cp.imag(num_psi))

    # Plot x-axis
    plt.subplot(1, 3, 1)
    plt.plot(x_vals, an_psi_real[:, x_vals.shape[0]//2, z_vals.shape[0]//2], color='red', alpha=0.5)
    plt.plot(x_vals, num_psi_real[:, x_vals.shape[0]//2, z_vals.shape[0]//2], color='green', linestyle='--', alpha=0.5)
    plt.plot(x_vals, an_psi_imag[:, x_vals.shape[0]//2, z_vals.shape[0]//2], color='blue', alpha=0.5)
    plt.plot(x_vals, num_psi_imag[:, x_vals.shape[0]//2, z_vals.shape[0]//2], color='#FFC0CB', linestyle='--', alpha=0.5)

    # Plot y-axis
    plt.subplot(1, 3, 2)
    plt.plot(y_vals, an_psi_real[x_vals.shape[0]//2, :, z_vals.shape[0]//2], color='red', alpha=0.5)
    plt.plot(y_vals, num_psi_real[x_vals.shape[0]//2, :, z_vals.shape[0]//2], color='green', linestyle='--', alpha=0.5)
    plt.plot(y_vals, an_psi_imag[x_vals.shape[0]//2, :, z_vals.shape[0]//2], color='blue', alpha=0.5)
    plt.plot(y_vals, num_psi_imag[x_vals.shape[0]//2, :, z_vals.shape[0]//2], color='#FFC0CB', linestyle='--', alpha=0.5)

    # Plot z-axis
    plt.subplot(1, 3, 3)
    plt.plot(z_vals, an_psi_real[x_vals.shape[0]//2, y_vals.shape[0]//2, :], color='red', alpha=0.5)
    plt.plot(z_vals, num_psi_real[x_vals.shape[0]//2, y_vals.shape[0]//2, :], color='green', linestyle='--', alpha=0.5)
    plt.plot(z_vals, an_psi_imag[x_vals.shape[0]//2, y_vals.shape[0]//2, :], color='blue', alpha=0.5)
    plt.plot(z_vals, num_psi_imag[x_vals.shape[0]//2, y_vals.shape[0]//2, :], color='#FFC0CB', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


'''
wave_begin = cp.asnumpy(cp.abs(vlna.wave_values[0]))  # Initial snapshot
wave_middle = cp.asnumpy(cp.abs(vlna.wave_values[len(vlna.wave_values) // 2]))  # Middle snapshot
wave_end = cp.asnumpy(cp.abs(vlna.wave_values[-1]))  # Final snapshot

# Take a central slice for 2D visualization (slice at z = middle of the grid)
z_idx = vlna.N // 2  # Index for z=0 plane
wave_begin_slice = wave_begin[:, :, z_idx]
wave_middle_slice = wave_middle[:, :, z_idx]
wave_end_slice = wave_end[:, :, z_idx]

# Grids for plotting (create a uniform grid based on boundaries)
x, y = np.linspace(vlna.boundaries[0][0], vlna.boundaries[0][1], vlna.N), \
    np.linspace(vlna.boundaries[1][0], vlna.boundaries[1][1], vlna.N)
X, Y = np.meshgrid(x, y)

# Plot the wavefunction slices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["Wavefunction at Beginning", "Wavefunction at Middle", "Wavefunction at End"]
waves = [wave_begin_slice, wave_middle_slice, wave_end_slice]

for ax, wave, title in zip(axes, waves, titles):
    im = ax.pcolormesh(X, Y, wave, shading='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, orientation="vertical", label="|ψ|")

plt.tight_layout()
plt.show()
'''