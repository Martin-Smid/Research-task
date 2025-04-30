import cupy as cp
import numpy as np
import os
import datetime
import pandas as pd
from resources.Functions.system_fucntions import plot_max_values_on_N


class Evolution_Class:
    """
    Class to handle the time evolution of wave functions in the simulation.
    Uses the split-step Fourier method for propagation.
    """

    def __init__(self, simulation, propagator,order=2):
        """
        Initialize the evolution handler.

        Parameters:
            simulation: Simulation_Class instance
            propagator: Propagator_Class instance
        """
        self.simulation = simulation
        self.propagator = propagator
        self.h = simulation.h
        self.num_steps = simulation.num_steps
        self.total_time = simulation.total_time
        self.save_max_vals = simulation.save_max_vals
        self.order = order

        # Evolution data storage
        self.wave_values = []
        self.accessible_times = []
        self.max_wave_vals_during_evolution = {}
        self.snapshot_directory = None



    def evolve(self, combined_psi, save_every=1):
        """
        Perform the full time evolution for the combined wave function,
        saving snapshots to files instead of memory.

        Parameters:
            combined_psi (cp.ndarray): Initial wave function
            save_every (int): Frequency of saving the wave function values

        Returns:
            cp.ndarray: Final evolved wave function
        """
        if save_every <= 0:
            save_every = 1

        # Create directory for saving snapshots
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"resources/data/simulation_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        self.snapshot_directory = save_dir

        # Initialize with the provided wave function
        psi = combined_psi.copy()
        self.accessible_times.append(0)

        # Save the initial state
        initial_path = f"{save_dir}/wave_snapshot_at_time_0.npy"
        np.save(initial_path, cp.asnumpy(psi))
        self.wave_values = [initial_path]

        # Evolve over time steps
        for step in range(self.num_steps):
            # Perform the evolution step
            if self.order == 2:
                psi = self.evolve_wavefunction_split_step_o2(
                    psi, step_index=step, total_steps=self.num_steps
                )
            elif self.order == 4:
                psi = self.evolve_wavefunction_split_step_o4(
                    psi, step_index=step, total_steps=self.num_steps)
            # Save wave function at specified intervals
            if step % save_every == 0 and step > 0:
                print(f"Still working... Step {step} out of {self.num_steps}")

                # Calculate current time
                current_time = step * self.h

                # Create filename and path
                snapshot_path = f"{save_dir}/wave_snapshot_at_time_{current_time:.6f}.npy"

                # Save to disk (convert from cupy to numpy array)
                np.save(snapshot_path, cp.asnumpy(psi))

                # Store the path instead of the array
                self.wave_values.append(snapshot_path)
                self.accessible_times.append(current_time)

            # Free memory
            cp.get_default_memory_pool().free_all_blocks()

        # Ensure the last state is saved if it wasn't already
        if (self.num_steps - 1) % save_every != 0:
            final_time = self.total_time
            final_path = f"{save_dir}/wave_snapshot_at_time_{final_time:.6f}.npy"
            np.save(final_path, cp.asnumpy(psi))
            self.wave_values.append(final_path)
            self.accessible_times.append(final_time)

        # Save metadata
        self._save_metadata(save_dir)

        # Cleanup and finalize
        self._finalize_evolution()

        return psi

    def evolve_wavefunction_split_step_o2(self, psi, step_index, total_steps):
        """
        Evolve the wavefunction using the split-step Fourier method.

        Parameters:
            psi (cp.ndarray): Initial wave function
            step_index (int): Index of the current step (starting at 0)
            total_steps (int): Total number of steps in the simulation

        Returns:
            cp.ndarray: Evolved wave function after one time step
        """
        is_first_step = (step_index == 0)
        is_last_step = (step_index == total_steps - 1)


        psi = self.kick_step(psi, is_first_step, is_last_step)
        psi = self.drift_step(psi)

        if is_last_step:
            psi = self.kick_step(psi, is_first_step, is_last_step)


        # Track maximum values if enabled
        if self.save_max_vals:
            self.max_wave_vals_during_evolution[step_index] = float(abs(psi).max())
        return psi

    def evolve_wavefunction_split_step_o4(self, psi, step_index, total_steps):
        v1 = (121.0 / 3924.0) * (12.0 - cp.sqrt(471.0))
        w = cp.sqrt(3.0 - 12.0 * v1 + 9 * v1 * v1)
        t2 = 0.25 * (1.0 - cp.sqrt((9.0 * v1 - 4.0 + 2 * w) / (3.0 * v1)))
        t1 = 0.5 - t2
        v2 = (1.0 / 6.0) - 4 * v1 * t1 * t1
        v0 = 1.0 - 2.0 * (v1 + v2)


        is_first_step = False
        is_last_step = False



        psi = self.kick_step(psi, is_first_step, is_last_step,v2)
        psi = self.drift_step(psi,t2)
        psi = self.kick_step(psi, is_first_step, is_last_step,v1)
        psi = self.drift_step(psi,t1)
        psi = self.kick_step(psi, is_first_step, is_last_step,v0)
        psi = self.drift_step(psi,t1)
        psi = self.kick_step(psi, is_first_step, is_last_step,v1)
        psi = self.drift_step(psi,t2)
        psi = self.kick_step(psi, is_first_step, is_last_step,v2)


        # Track maximum values if enabled
        if self.save_max_vals:
            self.max_wave_vals_during_evolution[step_index] = float(abs(psi).max())
        return psi
    def get_wave_function_at_time(self, time):
        """
        Retrieve the wave function values at a given time.
        Loads data from disk instead of memory.

        Parameters:
            time (float): The time at which to retrieve the wave function values

        Returns:
            cp.ndarray: The wave function at the given time
        """
        # Check if the input time is within range
        if time < self.accessible_times[0] or time > self.accessible_times[-1]:
            raise ValueError("Input time is outside the range of accessible times")

        # Find the closest time in the accessible times list
        closest_time_index = min(range(len(self.accessible_times)),
                                 key=lambda i: abs(self.accessible_times[i] - time))

        # Get the file path for the closest time
        file_path = self.wave_values[closest_time_index]

        # Load the wave function from disk and convert to cupy array
        wave_function = cp.array(np.load(file_path))

        return wave_function

    def _save_metadata(self, save_dir):
        """
        Save metadata about the simulation to a file.

        Parameters:
            save_dir (str): Directory to save metadata
        """
        with open(f"{save_dir}/metadata.txt", "w") as f:
            f.write(f"Total steps: {self.num_steps}\n")
            f.write(f"Time step: {self.h}\n")
            f.write(f"Total time: {self.total_time}\n")
            f.write("Accessible times:\n")
            f.write(",".join([str(t) for t in self.accessible_times]))

    def _finalize_evolution(self):
        """
        Cleanup and finalize the evolution process.
        Save maximum values if enabled.
        """
        if self.save_max_vals:
            self._save_max_values()
            plot_y_or_n = input("Should I plot these values? (y/n/del): ")
            if plot_y_or_n == "y":
                plot_max_values_on_N(self)
            elif plot_y_or_n == "del":
                if os.path.exists(self.max_vals_filename):
                    os.remove(self.max_vals_filename)
                    print(f"File '{self.max_vals_filename}' has been deleted.")
                else:
                    print(f"File '{self.max_vals_filename}' does not exist.")

        print("Evolution completed successfully")
        print(f"Saved times are {self.accessible_times}")
        cp.get_default_memory_pool().free_all_blocks()

    def _save_max_values(self):
        """
        Save the maximum values of wave functions during evolution to a CSV file.
        """
        self.max_vals_filename = "resources/data/max_values.csv"
        header = [
            "# This file contains the maximum values of wave functions at specific steps along the time evolution.",
            "# The first row contains the corresponding N - the resolution of the saved simulation.",
            "# The first column is the time step, and subsequent columns are max values for a given simulation.",
        ]

        # Convert dictionary to DataFrame (keys as index)
        new_data = pd.DataFrame.from_dict(
            self.max_wave_vals_during_evolution,
            orient="index",
            columns=[f"{int(self.simulation.N)}"]
        )

        # Check if file exists
        if os.path.exists(self.max_vals_filename):
            existing_data = pd.read_csv(self.max_vals_filename, index_col=0, comment="#")

            # Assign new column name dynamically
            new_column_name = f"{int(self.simulation.N)}"
            new_data.columns = [new_column_name]

            # Merge with existing data
            updated_data = pd.concat([existing_data, new_data], axis=1)

            # Write header and updated data back to the file
            with open(self.max_vals_filename, "w") as file:
                file.write("\n".join(header) + "\n")
            updated_data.to_csv(self.max_vals_filename, mode="a")
        else:
            # File doesn't exist, create it with a header
            with open(self.max_vals_filename, "w") as file:
                file.write("\n".join(header) + "\n")
            new_data.to_csv(self.max_vals_filename, mode="a")

        print(f"{self.max_vals_filename} saved")

    def kick_step(self, psi, is_first_step, is_last_step, time_factor=1):
        """
        takes care of the propagation by potential
        """

        if is_first_step or is_last_step:
            gravity_propagator = self.propagator.compute_gravity_propagator(psi, first_step=is_first_step, last_step=is_last_step, time_factor=time_factor)
            full_potential_propagator = self.propagator.static_potential_propagator * gravity_propagator
            return psi * full_potential_propagator
        else:
            gravity_propagator = self.propagator.compute_gravity_propagator(psi,time_factor=time_factor)
            full_potential_propagator = self.propagator.static_potential_propagator * gravity_propagator
            return psi * full_potential_propagator

    def drift_step(self, psi,time_factor=1):
        """
        kinetic part of the evolution
        """
        psi_k = cp.fft.fftn(psi)
        kinetic_propagator = self.propagator.compute_kinetic_propagator(time_factor=time_factor)
        psi_k *= kinetic_propagator
        return cp.fft.ifftn(psi_k)