import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Wave_function_class import *
from resources.Schrodinger_eq_functions import *


# Setup parameters for the domain
a, b = -10, 10  # Domain boundaries
N = 1024  # Number of spatial points

# Initialize the Wave_function instance
vlna = Wave_function(
    packet_type="gaussian",
    momenta=[0],
    means=[0],
    st_deviations=[0.1],
    dim=1,  # 1D wave function
    boundaries=[(a, b)],
    N=N,
    h=0.01,
    total_time=0.5,
    potential=quadratic_potential
)

print(vlna.psi_0)

print("#--------------------------------------------------------------------------#")


# Generate and show the animation
#anim = plot_1D_wavefunction_evolution(vlna, interval=10, save_file="1D_wave_potential.mp4")
plt.show()

trojdvlna = Wave_function(
    packet_type="LHO",
    momenta=[20,20,20],
    means=[0,0.5,0.8],
    st_deviations=[0.1,0.1,0.1],
    dim=3,  # 1D wave function
    boundaries=[(-10, 10), (-10, 10), (-10, 10)],
    N=64,
    h=0.01,
    total_time=0.5,
    potential=quadratic_potential
)




# Assuming the rest of Wave_function setup is already complete

# Initialize x_vals for spatial domain
x_vals = np.linspace(-10, 10, 1024)

# Arrays to store results
differences = []  # Mean magnitude differences
real_imag_1 = {'real': [], 'imag': []}  # Real/Imaginary values of first method
real_imag_2 = {'real': [], 'imag': []}  # Real/Imaginary values of second method
times = []  # Times corresponding to each step

# Time steps
time_steps = int(vlna.total_time / vlna.h)

# Iterate over each time step
for step in range(time_steps):
    t = step * vlna.h  # Equivalent time for the current step
    times.append(t)

    # Calculate the two methods of evolution
    psi_2d_evolved = cp.asnumpy(vlna.psi_0 * cp.exp(-1j * energy_nd([0, 0], omega=1, hbar=1) * t))
    wave_at_time_t = cp.asnumpy(vlna.wave_function_at_time(t))

    # Compute the absolute squared difference |Δψ|²
    difference = np.abs(psi_2d_evolved - wave_at_time_t) ** 2
    differences.append(np.mean(difference))

    # Store the real and imaginary parts of each method
    real_imag_1['real'].append(np.mean(np.real(psi_2d_evolved)))
    real_imag_1['imag'].append(np.mean(np.imag(psi_2d_evolved)))
    real_imag_2['real'].append(np.mean(np.real(wave_at_time_t)))
    print(f"a přidal jsem do listu {(np.real(wave_at_time_t))} s mean value {np.mean(np.real(wave_at_time_t))}")
    real_imag_2['imag'].append(np.mean(np.imag(wave_at_time_t)))

# Create the first plot: Differences in magnitude
plt.figure(figsize=(12, 6))

plt.scatter(times, differences, c='blue', label='Mean Difference |Δψ|²', alpha=0.7)
plt.title('Differences in Magnitudes Between Two Evolution Methods')
plt.xlabel('Time')
plt.ylabel('Mean Difference |Δψ|²')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # Reference line at 0
plt.legend()
plt.grid()
plt.show()

# Create the second plot: Real/Imaginary parts of each method
plt.figure(figsize=(12, 6))

# Plot for the first method
plt.scatter(times, real_imag_1['real'], c='lightgreen', label='Real Part (First Method)', alpha=0.7)
plt.scatter(times, real_imag_1['imag'], c='darkgreen', label='Imaginary Part (First Method)', alpha=0.7)

# Plot for the second method
plt.scatter(times, real_imag_2['real'], c='lightblue', label='Real Part (Second Method)', alpha=0.7)
plt.scatter(times, real_imag_2['imag'], c='darkblue', label='Imaginary Part (Second Method)', alpha=0.7)

plt.title('Real and Imaginary Parts of Wave Functions Over Time')
plt.xlabel('Time')
plt.ylabel('Mean Value')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # Reference line at 0
plt.legend()
plt.grid()
plt.show()

