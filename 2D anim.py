import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Schrodinger_eq_functions import *
from resources.Wave_function_class import *
import numpy as np





# Example Usage
x_vals = np.linspace(-10, 10, 1024)
y_vals = np.linspace(-10, 10, 1024)
t = 0.5  # Time

X, Y = np.meshgrid(x_vals, y_vals)
psi_2d = Wave_function(
    dim=2,
    boundaries=[(-10, 10), (-10, 10)],
    N=1024,
    total_time=1,  # Total simulation time
    h=0.01,  # Time step
    packet_type="LHO",
    means=[0.0, 0.0],
    st_deviations=[0.1, 0.1],
    momenta=[0, 0],
    potential=quadratic_potential # Free particle
)
T =0.1

psi_2d_evolved = psi_2d.psi_0 * np.exp(-1j * energy_nd([0,0],1,1)*T)



# Example Usage
twoD_wave_function = Wave_function(
    dim=2,
    boundaries=[(-10, 10), (-10, 10)],
    N=1024,
    total_time=1.0,  # Total simulation time
    h=0.01,  # Time step
    packet_type="LHO",
    means=[0.0, 0.0],
    st_deviations=[0.1, 0.1],
    momenta=[0, 0],
    potential=quadratic_potential
)


# Get the wavefunction value at time t = 0.01
wave_t_001 = twoD_wave_function.wave_function_at_time(0.1)
print(wave_t_001)
# Print or inspect the result using CuPy arrays



psi_2d_evolved = cp.asnumpy(psi_2d_evolved)
wave_t_001 = cp.asnumpy(wave_t_001)

X, Y = np.meshgrid(x_vals, y_vals)
#analytical_psi_t = psi_2d

plt.pcolor(X, Y, np.abs(psi_2d_evolved - wave_t_001)**2)

plt.show()

'''
analytical_psi_t = cp.array(analytical_psi_t)
for _ in range(len(twoD_wave_function.wave_function_at_time(0))):
    # Ensure both operands are `cupy.ndarray` for GPU-accelerated subtraction
    diff = cp.abs(twoD_wave_function.wave_function_at_time(0)[_] - wave_t_001[_])
    print(diff)
'''
 
anim = plot_wave_equation_evolution(twoD_wave_function, interval=20, save_file="anim_videos/pot_wave_equation_evolution.mp4", N=1024)
print(anim)
