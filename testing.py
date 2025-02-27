import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from resources.Classes.Wave_function_class import Wave_function
from resources.Functions.Schrodinger_eq_functions import energy_nd, quadratic_potential
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Classes.Wave_function_class import *
from resources.system_fucntions import *

from resources.Classes.Simulation_Class import Simulation_class

sim = Simulation_class(
    dim=4,
    boundaries=[(-10, 10), (-10, 10),(-10, 10), (-10, 10)],
    N=4,
    total_time=2,
    h=0.01,
)

print(f"sim dim je {sim.dim}")

wave = Wave_function(
    packet_type="gaussian",
    means=[0],
    st_deviations=[0.1, 0.1],
    simulation=sim,
    mass=1,
    omega=1,
)

