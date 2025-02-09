import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Wave_function_class import *
from resources.Schrodinger_eq_functions import *
from resources.system_fucntions import *


# Setup parameters for the domain
a, b = -5, 5  # Domain boundaries
N = 1024 # Number of spatial points

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
    total_time=10,
    potential=None
)


print("#--------------------------------------------------------------------------#")


# Generate and show the animation
anim = plot_1D_wavefunction_evolution(vlna, interval=10, save_file="1D_wave_potential.mp4")
#plt.show()


'''

x_vals = np.linspace(a,b,N,endpoint=False)

# Arrays to store results
differences_A, differences_N = [] , []  # Mean magnitude differences
real_imag_1 = {'real': [], 'imag': []}  # Real/Imaginary values of first method
real_imag_2 = {'real': [], 'imag': []}  # Real/Imaginary values of second method
times = []  # Times corresponding to each step

# Time steps
time_steps = int(vlna.total_time / vlna.h)
plt.figure(figsize=(12, 6))


# Iterate over each time step
for step in np.arange(0,time_steps,time_steps//25): #We don't need all time_steps
    t = step * vlna.h  # Equivalent time for the current step
    times.append(t)

    # Calculate the two methods of evolution
    psi_2d_evolved = cp.asnumpy(vlna.psi_0 * cp.exp(-1j * energy_nd([0], omega=1, hbar=1) * t))
    wave_at_time_t = cp.asnumpy(vlna.wave_function_at_time(t))

    plt.plot(x_vals, psi_2d_evolved.real, c='blue', alpha=1)
    plt.plot(x_vals, psi_2d_evolved.imag, c='red', alpha=1)
    plt.plot(x_vals, wave_at_time_t.real, c='green', alpha=0.7,ls='--')
    plt.plot(x_vals, wave_at_time_t.imag, c='magenta', alpha=0.7,ls='--')
    plt.plot(x_vals, np.abs(wave_at_time_t)**2, c='k', alpha=1)


    # Compute the absolute squared difference |Δψ|²
    differences_N.append((np.abs(wave_at_time_t)**2)[N//2])
    differences_A.append((np.abs(psi_2d_evolved)**2)[N//2])

    # Store the real and imaginary parts of each method
    real_imag_1['real'].append((np.real(psi_2d_evolved))[N//2])
    real_imag_1['imag'].append((np.imag(psi_2d_evolved))[N//2])
    real_imag_2['real'].append((np.real(wave_at_time_t))[N//2])
    real_imag_2['imag'].append((np.imag(wave_at_time_t))[N//2])

plt.plot([],[], c='blue', label=r'Re $\psi_{An}$', alpha=1)
plt.plot([],[], c='red', label=r'Im $\psi_{An}$', alpha=1)
plt.plot([],[], c='green', label=r'Re $\psi_{Num}$', alpha=0.7,ls='--')
plt.plot([],[], c='magenta', label=r'Im $\psi_{Num}$', alpha=0.7,ls='--')
plt.plot([],[], c='k', label=r'$|\psi_{Num}|^2$', alpha=1)

plt.title('Wafunction evolution in time')
plt.xlabel('Time')
plt.ylabel('ψ')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # Reference line at 0
plt.legend()
plt.grid()
plt.show()

# Create the first plot: Differences in magnitude
plt.figure(figsize=(12, 6))

plt.plot(times, differences_N , c='k', label='Numerical |ψ|²', alpha=0.7)
plt.plot(times, differences_A, c='magenta', label='Analytical |ψ|²', alpha=0.7)

# Plot for the first method
plt.scatter(times, real_imag_1['real'], c='lightgreen', label='Real Part (Analytical)', alpha=0.7)
plt.scatter(times, real_imag_1['imag'], c='darkgreen', label='Imaginary Part (Analytical)', alpha=0.7)

# Plot for the second method
plt.plot(times, real_imag_2['real'], c='lightblue', label='Real Part (Numerical)', alpha=0.7)
plt.plot(times, real_imag_2['imag'], c='darkblue', label='Imaginary Part (Numerical)', alpha=0.7)

plt.title('Absolute value, Real and Imaginary Parts of Wave Functions Over Time at x=0.0')
plt.xlabel('Time')
plt.ylabel('Mean Value')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # Reference line at 0
plt.legend()
plt.grid()
plt.show()
'''

