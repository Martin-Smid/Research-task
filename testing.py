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
    dim=1,                             # 2D simulation
    boundaries=[(-10, 10)], # Spatial boundaries
    N=256,                             # Grid resolution
    total_time=10.0,                   # Total simulation time
    h=0.01,                            # Time step
    use_gravity=True , # Enable gravitational effects
    static_potential=quadratic_potential,
)

vlna = Wave_function(
    packet_type="gaussian",
    means=[0],
    st_deviations=[0.1],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[2],
)

#TODO make it so that its possible to include Ground state as a wave_packet option, r is sphereical coordinate and must bee calcllated from means with phi as the real_part of wave function values at radius r
