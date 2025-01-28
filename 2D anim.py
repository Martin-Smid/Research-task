import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Schrodinger_eq_functions import *
from Wave_function_class import *


boundaries = [(-1,1), (-1,1)]
N = 256

boundaries4D = [(-1,1), (-1,1), (-6,2), (-8,3)]

twoD_wavefunction = Wave_function(
    packet_type="gaussian",
    means=[0,0.0],
    momenta=[50,0],
    st_deviations=[0.1,0.1],
    dim=2,  # 2D wave function
    boundaries=boundaries,
    N=N,
    h=0.1,
    total_time=10
)





anim = plot_wave_equation_evolution(twoD_wavefunction, interval=20, save_file="Awave_equation_evolution.mp4", N=N)
print(anim)
