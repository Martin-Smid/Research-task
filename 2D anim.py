import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from resources.Wave_function_class import Wave_function
from resources.Schrodinger_eq_functions import energy_nd, quadratic_potential

# Initialize constants
N = 512

# Initialize the 2D system
x_vals = np.linspace(-10, 10, N)
y_vals = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x_vals, y_vals)

vlna = Wave_function(
    dim=2,
    boundaries=[(-10, 10), (-10, 10)],
    N=N,
    total_time=10,  # Total simulation time
    h=0.1,  # Time interval
    packet_type="LHO",
    means=[0.0, 0.0],
    st_deviations=[0.1, 0.1],
    momenta=[0, 0],
    potential=quadratic_potential,  # Quadratic potential for harmonic evolution
)

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
