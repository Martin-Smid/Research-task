"""Resume an aborted simulation from its latest complete checkpoint."""

from examples._runtime import configure_cuda_on_windows

configure_cuda_on_windows()

from resources.Classes.Simulation_Class import Simulation_Class


simulation_directory = "resources/data/simulation_..."

sim = Simulation_Class.from_checkpoint(simulation_directory)

# resume() calls evolve() internally; no second evolve() call is needed.
sim.resume(save_every=50)
