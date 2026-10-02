"""Start a new experiment from a completed simulation endpoint."""

from examples._runtime import configure_cuda_on_windows

configure_cuda_on_windows()

from resources.Classes.Simulation_Class import Simulation_Class


endpoint_directory = "resources/data/simulation_..."

sim = Simulation_Class.from_checkpoint(endpoint_directory)
sim.start_new_segment(
    total_time=0.5,  # Duration of the new segment, not absolute simulation time.
    h=0.001,
    use_gravity=True,
)
sim.evolve(save_every=50)

print(f"New segment saved to: {sim.snapshot_directory}")
