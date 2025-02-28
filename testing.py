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
    dim=2,                             # 2D simulation
    boundaries=[(-10, 10), (-10, 10)], # Spatial boundaries
    N=256,                             # Grid resolution
    total_time=10.0,                   # Total simulation time
    h=0.01,                            # Time step
    use_gravity=True , # Enable gravitational effects
    static_potential=quadratic_potential,
)

wave1 = Wave_function(
    packet_type="gaussian",
    means=[0,0],
    st_deviations=[0.1, 0.1],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0, 0],
)
wave2 = Wave_function(
    packet_type="gaussian",
    means=[0,0],
    st_deviations=[0.1, 0.1],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0, 0],
)

# Add wave functions to the simulation
sim.add_wave_function(wave1)  # Assuming wave_function1 is already defined
sim.add_wave_function(wave2)  # Assuming wave_function2 is already defined

# Initialize and run the simulation
sim.initialize_simulation()
sim.evolve(save_every=10)  # Save every 10th step

# Access the evolution results
final_state = sim.combined_psi
all_states = sim.wave_values



