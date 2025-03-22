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
    dim=3,                             # 2D simulation
    boundaries=[(-10, 10),(-10, 10),(-10, 10)], # Spatial boundaries
    N=256,                             # Grid resolution
    total_time=10.0,                   # Total simulation time
    h=0.01,                            # Time step
    use_gravity=True , # Enable gravitational effects
    static_potential=None,
)

vlna = Wave_function(
    packet_type="/home/martin/Downloads/GroundState(1).dat",
    means=[0,0,0],
    st_deviations=[0.5,0.5,0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0,0,0],
)

sim.add_wave_function(vlna)

sim.evolve(save_every=250)

'''1D
plt.figure()
for time in sim.accessible_times:
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time))**2)
    plt.plot(wave_values, label=f"Time: {time}")
plt.legend()
plt.show()
'''

import matplotlib.pyplot as plt

# Choose a slice in the Z direction (middle of the grid)
z_index = sim.grids[2].shape[0] // 2  # Middle z-plane

plt.figure(figsize=(8, 6))
for time in sim.accessible_times:
    wave_values = cp.asnumpy(abs(sim.get_wave_function_at_time(time)) ** 2)

    # Take the middle z-slice
    wave_slice = wave_values[:, :, z_index]

    plt.imshow(cp.asnumpy(wave_slice), extent=[sim.grids[0].min().get(), sim.grids[0].max().get(),
                                               sim.grids[1].min().get(), sim.grids[1].max().get()],
               origin="lower", cmap="inferno")
    plt.colorbar(label="|ψ|²")
    plt.title(f"Wavefunction Probability Density at Time {time}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

#TODO make it so that its possible to include Ground state as a wave_packet option, r is sphereical coordinate and must be calcllated from means with phi as the real_part of wave function values at radius r
#TODO make it so that you can create different dim wave from the simulation, maybe make dim a wave_function class attribute and if not given take it from sim