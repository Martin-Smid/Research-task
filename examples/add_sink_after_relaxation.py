"""Add a central sink after the wave/baryon system has relaxed."""

from examples._runtime import configure_cuda_on_windows

configure_cuda_on_windows()

from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


endpoint_directory = "resources/data/simulation_..."

sim = Simulation_Class.from_checkpoint(endpoint_directory)
sim.start_new_segment(total_time=0.5, h=0.001)

central_sink = SinkNBody(
    simulation=sim,
    N_sinks=1,
    initial_masses=[1.0e7],
    initial_positions=[[0, 0, 0]],
    initial_velocities=[[0, 0, 0]],
)
sim.add_baryons(central_sink)
sim.evolve(save_every=50)
