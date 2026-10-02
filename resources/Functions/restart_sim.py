"""Backward-compatible entry point for the checkpoint API."""

from resources.Classes.Simulation_Class import Simulation_Class


def restart_simulation(checkpoint_path, static_potential=None, output_directory=None):
    """Load a checkpoint while preserving the historical tuple return value.

    New code should use Simulation_Class.from_checkpoint() followed by
    simulation.resume().
    """
    simulation = Simulation_Class.from_checkpoint(
        checkpoint_path,
        static_potential=static_potential,
        output_directory=output_directory,
    )
    return simulation, simulation.current_step, simulation._restart_manifest
