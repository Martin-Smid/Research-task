import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Wave_function_class import *
from Schrodinger_eq_functions import *

# Setup parameters for the domain
a, b = -1, 1  # Domain boundaries
N = 1024  # Number of spatial points

# Initialize the Wave_function instance
vlna = Wave_function(
    packet_type="gaussian",
    means=[0],
    st_deviations=[0.1],
    dim=1,  # 1D wave function
    boundaries=[(a, b)],
    N=N,
    h=0.1,
    total_time=100
)



# Animate the wavefunction evolution
anim = plot_1D_wavefunction_evolution(
    x=vlna.grids[0],
    vlna=vlna,
    num_steps=vlna.num_steps,
    interval=20,  # Frame interval in milliseconds
    save_file="wavefunction_evolution.mp4",  # Optional: save the animation
    dx=vlna.dx[0]
)

# Display the animation
plt.show()
