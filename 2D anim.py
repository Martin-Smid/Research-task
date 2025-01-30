import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Schrodinger_eq_functions import *
from resources.Wave_function_class import *


boundaries = [(-1,1), (-1,1)]



twoD_wave_function = Wave_function(
    dim=2,  # 2D simulation
    boundaries=[(-1, 1), (-1, 1)],
    N=128,  # Number of grid points per dimension
    total_time=10.0,  # Total simulation time
    h=0.01,  # Time step
    packet_type="gaussian",
    means=[0.0, 0.0],
    st_deviations=[0.1, 0.1],
    momenta=[0, 0],  # Zero initial momentum
    potential=quadratic_potential
)





anim = plot_wave_equation_evolution(twoD_wave_function, interval=20, save_file="anim_videos/pot_wave_equation_evolution.mp4", N=1024)
print(anim)
