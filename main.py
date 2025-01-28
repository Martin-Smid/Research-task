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
    momenta=[20],
    means=[0],
    st_deviations=[0.1],
    dim=1,  # 1D wave function
    boundaries=[(a, b)],
    N=N,
    h=0.1,
    total_time=10
)




anim = plot_wave_equation_evolution(vlna, interval=20, save_file="wave_equation_evolution.mp4", N=N)
print(anim)

