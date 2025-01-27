import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Schrodinger_eq_functions import *
from Wave_function_class import *


boundaries = [(-1,1), (-1,1)]
N = 1024

twoD_wavefunction = Wave_function(
    packet_type="gaussian",
    means=[0,0.0],
    st_deviations=[0.1,0.1],
    dim=2,  # 2D wave function
    boundaries=boundaries,
    N=N,
    h=0.1,
    total_time=10
)

# Create the animation
anim = plot_2D_wavefunction_evolution(
    x=twoD_wavefunction.grids[0],
    y=twoD_wavefunction.grids[1],
    psi_0=twoD_wavefunction.psi_0,
    propagator_x=twoD_wavefunction.propagator,
    propagator_y=twoD_wavefunction.propagator,
    dx=twoD_wavefunction.dx[0],
    num_steps=twoD_wavefunction.num_steps,
    interval=20,  # Frame interval in milliseconds
    save_file="2D_wave_function_evolution2.mp4",  # Optional: save the animation
    N=N
)

# Display the animation
plt.show()
