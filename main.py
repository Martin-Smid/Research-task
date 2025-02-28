import matplotlib.pyplot as plt
from cupy import asnumpy

from resources.Classes.Wave_function_class import *
from resources.system_fucntions import *


# Setup parameters for the domain
a, b = -10, 10  # Domain boundaries
N = 512 # Number of spatial points

# Initialize the Wave_function instance


sim = Simulation_class(
    dim=1,                             # 2D simulation
    boundaries=[(-10, 10)], # Spatial boundaries
    N=512,                             # Grid resolution
    total_time=10.0,                   # Total simulation time
    h=0.01,                            # Time step
    use_gravity=False,  # Enable gravitational effects
    static_potential=quadratic_potential,
)

vlna = Wave_function(
    packet_type="LHO",
    means=[-5],
    st_deviations=[0.1],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0],
)
vlna2 = Wave_function(
    packet_type="LHO",
    means=[5],
    st_deviations=[0.1],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0],
)
sim.add_wave_function(vlna)



sim.evolve(save_every=50)



x_vals = np.linspace(a, b, N, endpoint=False)

controlled_times = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
for time, index in zip(controlled_times, range(len(controlled_times))):
    print(f"Time: {time}")
    print(f"Psi: {vlna.psi  * cp.exp(-1j * energy_nd([0], omega=1, hbar=1) * time)}")
    an_psi = asnumpy(cp.abs(vlna.psi  * cp.exp(-1j * energy_nd([0], omega=1, hbar=1) * time))**2)
    an_psi_real = asnumpy(cp.real(vlna.psi  * cp.exp(-1j * energy_nd([0], omega=1, hbar=1) * time)))
    an_psi_imag = asnumpy(cp.imag(vlna.psi  * cp.exp(-1j * energy_nd([0], omega=1, hbar=1) * time)))
    print(f"Wave values: {sim.wave_values[index]}")
    num_psi = asnumpy(cp.abs(sim.wave_values[index]) ** 2)
    num_psi_real = asnumpy(cp.real(sim.wave_values[index]))
    num_psi_imag = asnumpy(cp.imag(sim.wave_values[index]))
    plt.plot(x_vals, an_psi_real, color='red')
    plt.plot(x_vals, num_psi_real, color='green', linestyle='--')
    plt.plot(x_vals, an_psi_imag, color='blue')
    plt.plot(x_vals, num_psi_imag, color='#FFC0CB', linestyle='--')

plt.legend(['Analytical real', 'Numerical real', 'Analytical imag', 'Numerical imag'], loc='upper right')
plt.savefig(r'plots/LHO_comparison_evolution_in_simulation.png')

print("#--------------------------------------------------------------------------#")

'''
time_steps = [0, len(vlna.wave_values) // 4, len(vlna.wave_values) // 2,
              (3 * len(vlna.wave_values)) // 4, len(vlna.wave_values) - 1]
wave_snapshots = [cp.asnumpy(cp.abs(vlna.wave_values[step])) for step in time_steps]

# Directly create x using vlna.boundaries and vlna.N
x = np.linspace(vlna.boundaries[0][0], vlna.boundaries[0][1], vlna.N)

# Plot the wavefunction at the selected time steps
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
titles = ["Wavefunction at Start", "Wavefunction at 1/4 Time",
          "Wavefunction at Half Time", "Wavefunction at 3/4 Time",
          "Wavefunction at End"]

for ax, wave, title in zip(axes, wave_snapshots, titles):
    ax.plot(x, np.abs(wave)**2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("|ψ|^2")

plt.tight_layout()
plt.show()


'''


# Generate and show the animation
#anim = plot_1D_wavefunction_evolution(vlna, interval=10, save_file="quad1D_wave_potential.mp4")
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
    psi_2d_evolved = cp.asnumpy(vlna.psi * cp.exp(-1j * energy_nd([0], omega=1, hbar=1) * t))
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


