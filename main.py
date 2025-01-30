import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Wave_function_class import *
from resources.Schrodinger_eq_functions import *


# Setup parameters for the domain
a, b = -1, 1  # Domain boundaries
N = 1024  # Number of spatial points

# Initialize the Wave_function instance
vlna = Wave_function(
    packet_type="gaussian",
    momenta=[20],
    means=[0],
    st_deviations=[0.1],
    dim=1,  # 1D wave function
    boundaries=[(a, b)],
    N=N,
    h=0.01,
    total_time=10,
    potential=quadratic_potential
)

print(vlna.psi_0)




# Generate and show the animation
anim = plot_1D_wavefunction_evolution(vlna, interval=20, save_file="1D_wave_potential.mp4")
plt.show()

