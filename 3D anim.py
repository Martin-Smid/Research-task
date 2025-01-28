import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Schrodinger_eq_functions import *
from Wave_function_class import *



boundaries3D = [(-1,1), (-1,1), (-1,1)]
N = 128


threeD_wavefunction = Wave_function(
    packet_type="gaussian",
    momenta=[10,10,10],
    means=[0, 0.0,0],
    st_deviations=[0.1, 0.1,0.3],
    dim=3,  # 3D wave function
    boundaries=boundaries3D,
    N=N,
    h=0.1,
    total_time=1
)

anim = plot_wave_equation_evolution(threeD_wavefunction, interval=2000, save_file="3Dwave_equation_evolution.mp4", N=N)