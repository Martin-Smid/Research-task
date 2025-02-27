import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from resources.Classes.Wave_function_class import Wave_function
from resources.Functions.Schrodinger_eq_functions import energy_nd, quadratic_potential

# Initialize constants
N = 128  # Reduced for 3D computation
# Initialize the 3D system
x_vals = np.linspace(-10, 10, N)
y_vals = np.linspace(-10, 10, N)
z_vals = np.linspace(-10, 10, N)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

vlna = Wave_function(
    dim=3,  # Changed to 3D
    boundaries=[(-10, 10), (-10, 10), (-10, 10)],  # 3D boundaries
    N=N,
    total_time=2,
    h=0.01,
    mass=1,
    packet_type="gaussian",
    means=[0.0, 0.0, 0.0],  # 3D means
    st_deviations=[0.1, 0.1, 0.1],  # 3D standard deviations
    gravity_potential=True,
    momenta=[0, 0, 0],  # 3D momenta
    potential=None  # Quadratic potential
)

# Iterate over reduced time steps for better visualization


times = []
real_x_an_values, imag_x_an_values = [], []
real_x_num_values, imag_x_num_values = [], []

real_y_an_values, imag_y_an_values = [], []
real_y_num_values, imag_y_num_values = [], []

real_z_an_values, imag_z_an_values = [], []
real_z_num_values, imag_z_num_values = [], []

time_steps = int(vlna.total_time / vlna.h)

for step in np.arange(0, time_steps, time_steps // 25):
    t = step * vlna.h
    times.append(t)

    psi_3d_evolved = cp.asnumpy(vlna.psi_0 * cp.exp(-1j * energy_nd([0, 0, 0], omega=1, hbar=1) * t))
    wave_at_time_t = cp.asnumpy(vlna.wave_function_at_time(t))

    real_x_an = np.real(psi_3d_evolved[:, N // 2, N // 2])
    imag_x_an = np.imag(psi_3d_evolved[:, N // 2, N // 2])

    real_x_num = np.real(wave_at_time_t[:, N // 2, N // 2])
    imag_x_num = np.imag(wave_at_time_t[:, N // 2, N // 2])

    real_x_an_values.append(real_x_an)
    imag_x_an_values.append(imag_x_an)
    real_x_num_values.append(real_x_num)
    imag_x_num_values.append(imag_x_num)

    real_y_an = np.real(psi_3d_evolved[N // 2, :, N // 2])
    imag_y_an = np.imag(psi_3d_evolved[N // 2, :, N // 2])

    real_y_num = np.real(wave_at_time_t[N // 2, :, N // 2])
    imag_y_num = np.imag(wave_at_time_t[N // 2, :, N // 2])

    real_y_an_values.append(real_y_an)
    imag_y_an_values.append(imag_y_an)
    real_y_num_values.append(real_y_num)
    imag_y_num_values.append(imag_y_num)

    real_z_an = np.real(psi_3d_evolved[N // 2, N // 2, :])
    imag_z_an = np.imag(psi_3d_evolved[N // 2, N // 2, :])

    real_z_num = np.real(wave_at_time_t[N // 2, N // 2, :])
    imag_z_num = np.imag(wave_at_time_t[N // 2, N // 2, :])

    real_z_an_values.append(real_z_an)
    imag_z_an_values.append(imag_z_an)
    real_z_num_values.append(real_z_num)
    imag_z_num_values.append(imag_z_num)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
fig.suptitle('Comparison of Real and Imaginary Parts of Wave Functions', fontsize=14)

for i, t in enumerate(times):
    ax1.plot(x_vals, real_x_an_values[i], color='blue', alpha=0.6, label=r'Re $\psi_{An}$' if i == 0 else "")
    ax1.plot(x_vals, imag_x_an_values[i], color='red', alpha=0.6, label=r'Im $\psi_{An}$' if i == 0 else "")
    ax1.plot(x_vals, real_x_num_values[i], color='green', alpha=0.4, linestyle='--',
             label=r'Re $\psi_{Num}$' if i == 0 else "")
    ax1.plot(x_vals, imag_x_num_values[i], color='orange', alpha=0.4, linestyle='--',
             label=r'Im $\psi_{Num}$' if i == 0 else "")

ax1.set_title('Real and Imaginary Parts Along x-axis (central column)')
ax1.set_xlabel('x-axis')
ax1.set_ylabel('ψ')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid()

for i, t in enumerate(times):
    ax2.plot(y_vals, real_y_an_values[i], color='blue', alpha=0.6, label=r'Re $\psi_{An}$' if i == 0 else "")
    ax2.plot(y_vals, imag_y_an_values[i], color='red', alpha=0.6, label=r'Im $\psi_{An}$' if i == 0 else "")
    ax2.plot(y_vals, real_y_num_values[i], color='green', alpha=0.4, linestyle='--',
             label=r'Re $\psi_{Num}$' if i == 0 else "")
    ax2.plot(y_vals, imag_y_num_values[i], color='orange', alpha=0.4, linestyle='--',
             label=r'Im $\psi_{Num}$' if i == 0 else "")

ax2.set_title('Real and Imaginary Parts Along y-axis (central column)')
ax2.set_xlabel('y-axis')
ax2.set_ylabel('ψ')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid()

for i, t in enumerate(times):
    ax3.plot(z_vals, real_z_an_values[i], color='blue', alpha=0.6, label=r'Re $\psi_{An}$' if i == 0 else "")
    ax3.plot(z_vals, imag_z_an_values[i], color='red', alpha=0.6, label=r'Im $\psi_{An}$' if i == 0 else "")
    ax3.plot(z_vals, real_z_num_values[i], color='green', alpha=0.4, linestyle='--',
             label=r'Re $\psi_{Num}$' if i == 0 else "")
    ax3.plot(z_vals, imag_z_num_values[i], color='orange', alpha=0.4, linestyle='--',
             label=r'Im $\psi_{Num}$' if i == 0 else "")

ax3.set_title('Real and Imaginary Parts Along z-axis (central column)')
ax3.set_xlabel('z-axis')
ax3.set_ylabel('ψ')
ax3.legend(fontsize=10, loc='upper right')
ax3.grid()

plt.tight_layout()
plt.subplots_adjust(top=0.9)
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