import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from resources.Classes.Wave_function_class import Wave_function
from resources.Functions.Schrodinger_eq_functions import energy_nd, quadratic_potential
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize constants
N = 256
# Initialize the 2D system
x_vals = np.linspace(-10, 10, N)
y_vals = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x_vals, y_vals)

vlna = Wave_function(
    dim=2,
    boundaries=[(-10, 10), (-10, 10)],
    N=N,
    total_time=2000,  # Total simulation time
    h=10,  # Time interval
    mass=1000000,
    packet_type="gaussian",
    means=[0.0, 0.0],
    st_deviations=[5,5],
    gravity_potential=True,
    momenta=[0, 0],
    potential=None,  # Quadratic potential for harmonic evolution
)

time_steps = [0, len(vlna.wave_values) // 4, len(vlna.wave_values) // 2,
              (3 * len(vlna.wave_values)) // 4, len(vlna.wave_values) - 1]
wave_snapshots = [cp.asnumpy(cp.abs(vlna.wave_values[step])) for step in time_steps]

# Directly create x and y using vlna.boundaries and vlna.N
x = np.linspace(vlna.boundaries[0][0], vlna.boundaries[0][1], vlna.N)
y = np.linspace(vlna.boundaries[1][0], vlna.boundaries[1][1], vlna.N)

# Plot the wavefunction at the selected time steps
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
titles = ["Wavefunction at Start", "Wavefunction at 1/4 Time",
          "Wavefunction at Half Time", "Wavefunction at 3/4 Time",
          "Wavefunction at End"]

for ax, wave, title in zip(axes, wave_snapshots, titles):

    im = ax.pcolormesh(x, y, np.abs(wave[0]**2 + wave[1]**2), shading='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, orientation="vertical", label="|ψ|")

plt.tight_layout()
plt.show()

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
'''
#plotování rozdílu 2D LHO

# Arrays to store results
times = []  # Times corresponding to each calculation
real_x_an_values, imag_x_an_values = [], []  # Real and imaginary parts of analytical wave function along x-axis
real_x_num_values, imag_x_num_values = [], []  # Real and imaginary parts of numerical wave function along x-axis

real_y_an_values, imag_y_an_values = [], []  # Real and imaginary parts of analytical wave function along y-axis
real_y_num_values, imag_y_num_values = [], []  # Real and imaginary parts of numerical wave function along y-axis

# Time steps
time_steps = int(vlna.total_time / vlna.h)

# Iterate over reduced time steps for better visualization
for step in np.arange(0, time_steps, time_steps // 25):  # We don't need all time steps
    t = step * vlna.h  # Equivalent time for the given step
    times.append(t)

    # Analytical solution (complex array for current time step)
    psi_2d_evolved = cp.asnumpy(vlna.psi_0 * cp.exp(-1j * energy_nd([0, 0], omega=1, hbar=1) * t))

    # Numerical wave function evolved at the current time
    wave_at_time_t = cp.asnumpy(vlna.wave_function_at_time(t))

    # Calculate real and imaginary parts along the middle slice (x-axis and y-axis)
    # Real and imaginary parts along the central column (x-axis)
    real_x_an = np.real(psi_2d_evolved[:, N // 2])  # Analytical real part along x-axis
    imag_x_an = np.imag(psi_2d_evolved[:, N // 2])  # Analytical imaginary part along x-axis

    real_x_num = np.real(wave_at_time_t[:, N // 2])  # Numerical real part along x-axis
    imag_x_num = np.imag(wave_at_time_t[:, N // 2])  # Numerical imaginary part along x-axis

    # Real and imaginary parts along the central row (y-axis)
    real_y_an = np.real(psi_2d_evolved[N // 2, :])  # Analytical real part along y-axis
    imag_y_an = np.imag(psi_2d_evolved[N // 2, :])  # Analytical imaginary part along y-axis

    real_y_num = np.real(wave_at_time_t[N // 2, :])  # Numerical real part along y-axis
    imag_y_num = np.imag(wave_at_time_t[N // 2, :])  # Numerical imaginary part along y-axis

    # Store the calculated values for plotting
    real_x_an_values.append(real_x_an)
    imag_x_an_values.append(imag_x_an)
    real_x_num_values.append(real_x_num)
    imag_x_num_values.append(imag_x_num)

    real_y_an_values.append(real_y_an)
    imag_y_an_values.append(imag_y_an)
    real_y_num_values.append(real_y_num)
    imag_y_num_values.append(imag_y_num)

# Plotting the Comparison Along x-axis and y-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle('Comparison of Real and Imaginary Parts of Wave Functions', fontsize=14)

# Iterate over the stored values to plot
for i, t in enumerate(times):
    # Plot analytical and numerical real/imaginary parts for the x-axis
    ax1.plot(x_vals, real_x_an_values[i], color='blue', alpha=0.6, label=r'Re $\psi_{An}$' if i == 0 else "")
    ax1.plot(x_vals, imag_x_an_values[i], color='red', alpha=0.6, label=r'Im $\psi_{An}$' if i == 0 else "")
    ax1.plot(x_vals, real_x_num_values[i], color='green', alpha=0.4, linestyle='--',
             label=r'Re $\psi_{Num}$' if i == 0 else "")
    ax1.plot(x_vals, imag_x_num_values[i], color='orange', alpha=0.4, linestyle='--',
             label=r'Im $\psi_{Num}$' if i == 0 else "")

    # Plot analytical and numerical real/imaginary parts for the y-axis
    ax2.plot(y_vals, real_y_an_values[i], color='blue', alpha=0.6, label=r'Re $\psi_{An}$' if i == 0 else "")
    ax2.plot(y_vals, imag_y_an_values[i], color='red', alpha=0.6, label=r'Im $\psi_{An}$' if i == 0 else "")
    ax2.plot(y_vals, real_y_num_values[i], color='green', alpha=0.4, linestyle='--',
             label=r'Re $\psi_{Num}$' if i == 0 else "")
    ax2.plot(y_vals, imag_y_num_values[i], color='orange', alpha=0.4, linestyle='--',
             label=r'Im $\psi_{Num}$' if i == 0 else "")

# Customize plots for x-axis comparison
ax1.set_title('Real and Imaginary Parts Along x-axis (central column)')
ax1.set_xlabel('x-axis')
ax1.set_ylabel('ψ')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid()

# Customize plots for y-axis comparison
ax2.set_title('Real and Imaginary Parts Along y-axis (central row)')
ax2.set_xlabel('y-axis')
ax2.set_ylabel('ψ')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid()

# Adjust layout and display plots
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Space for the title
plt.show()



'''