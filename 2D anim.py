import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.constants import gravitational_constant

from resources.Classes.Wave_function_class import Wave_function
from resources.Functions.Schrodinger_eq_functions import energy_nd, quadratic_potential
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Classes.Simulation_Class import Simulation_class

N = 256
# Initialize the 2D system
x_vals = np.linspace(-10, 10, N)
y_vals = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x_vals, y_vals)

sim = Simulation_class(
    dim=2,
    boundaries=[(-10, 10), (-10, 10)],
    N=N,
    total_time=5,
    h=0.001,
    static_potential=quadratic_potential,
    use_gravity=False,
)

vlna = Wave_function(
    simulation=sim,
    mass=1.5,
    packet_type="LHO",
    means=[0, 0.0],
    st_deviations=[0.1,0.1],
    momenta=[0, 0],
      # Quadratic potential for harmonic evolution
)

vlna2 = Wave_function(
    simulation=sim,
    mass=1.5,
    packet_type="LHO",
    means=[-5, 5.0],
    st_deviations=[0.1,0.1],
    momenta=[0, 0],
      # Quadratic potential for harmonic evolution
)

sim.add_wave_function(vlna)


sim.evolve(save_every=500)


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


#TODO add second wave function and make them interact - sum operator to the wave function - maybe add momentum to force collapse
#TODO is implement the minimal time step requirement    eq 21 in Volker paper with a = 1

'''
time_steps = [0, len(vlna.wave_values) // 4, len(vlna.wave_values) // 2,
              (3 * len(vlna.wave_values)) // 4, len(vlna.wave_values) - 1]
wave_snapshots = [cp.asnumpy((vlna.wave_values[step])) for step in time_steps]

# Directly create x and y using vlna.boundaries and vlna.N
x = np.linspace(vlna.boundaries[0][0], vlna.boundaries[0][1], vlna.N)
y = np.linspace(vlna.boundaries[1][0], vlna.boundaries[1][1], vlna.N)
xy = np.meshgrid(x,y, indexing='ij')
# Plot the wavefunction at the selected time steps
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
titles = ["Wavefunction at Start", "Wavefunction at 1/4 Time",
          "Wavefunction at Half Time", "Wavefunction at 3/4 Time",
          "Wavefunction at End"]

for ax, wave, title in zip(axes, wave_snapshots, titles):
    print(wave)
    im = ax.contourf(xy[0],xy[1], np.abs(wave), shading='auto', cmap='viridis', levels = 100)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, orientation="vertical", label="|ψ|")

plt.tight_layout()
plt.show()



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
    real_x_an = np.real(psi_2d_evolved[:, vlna.N // 2])  # Analytical real part along x-axis
    imag_x_an = np.imag(psi_2d_evolved[:, vlna.N // 2])  # Analytical imaginary part along x-axis

    real_x_num = np.real(wave_at_time_t[:, vlna.N // 2])  # Numerical real part along x-axis
    imag_x_num = np.imag(wave_at_time_t[:, vlna.N// 2])  # Numerical imaginary part along x-axis

    # Real and imaginary parts along the central row (y-axis)
    real_y_an = np.real(psi_2d_evolved[vlna.N// 2, :])  # Analytical real part along y-axis
    imag_y_an = np.imag(psi_2d_evolved[vlna.N // 2, :])  # Analytical imaginary part along y-axis

    real_y_num = np.real(wave_at_time_t[vlna.N // 2, :])  # Numerical real part along y-axis
    imag_y_num = np.imag(wave_at_time_t[vlna.N// 2, :])  # Numerical imaginary part along y-axis

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
#plt.savefig('plots/wave_function_comparison.png', dpi=300)

'''
