import cupy as cp
import numpy as np
import os
import datetime
import pandas as pd
from resources.Functions.system_fucntions import plot_max_values_on_N

np.random.seed(1)


class Evolution_Class:
    """
    Class to handle the time evolution of multiple wave functions in the simulation.
    Uses the split-step Fourier method for propagation with orders 2, 4, or 6.
    Each wave function should have a .psi attribute containing the wave data.
    """

    def __init__(self, simulation, propagator, order=2):
        """
        Initialize the evolution handler.

        Parameters:
            simulation: Simulation_Class instance
            propagator: Propagator_Class instance
            order: Order of the split-step method (2, 4, or 6)
        """
        self.simulation = simulation
        self.propagator = propagator
        self.h = simulation.h
        self.num_steps = simulation.num_steps
        self.total_time = simulation.total_time
        self.save_max_vals = simulation.save_max_vals
        self.order = order

        # Evolution data storage - now for multiple wave functions
        self.wave_values = []  # List of lists: [wf_index][time_index] = file_path
        self.accessible_times = []
        self.max_wave_vals_during_evolution = {}  # {wf_index: {step: max_val}}
        self.snapshot_directory = None
        self.num_wave_functions = 0

        # Calculate coefficients and pre-compute propagators based on order
        self._calculate_coefficients_and_propagators()

    def evolve(self, wave_functions, save_every=1):
        """
        Perform the full time evolution for multiple wave functions,
        saving snapshots to files instead of memory.

        Parameters:
            wave_functions (list): List of wave function instances, each with .psi attribute
            save_every (int): Frequency of saving the wave function values

        Returns:
            list: List of wave function instances with evolved .psi attributes
        """
        if save_every <= 0:
            save_every = 1

        # Initialize for multiple wave functions
        self.num_wave_functions = len(wave_functions)

        # Create directory for saving snapshots
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"resources/data/simulation_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        self.snapshot_directory = save_dir

        # Initialize storage for each wave function
        self.wave_values = [[] for _ in range(self.num_wave_functions)]


        self.accessible_times.append(0)

        # Save the initial states for all wave functions
        for wf_idx, wf in enumerate(wave_functions):
            initial_path = f"{save_dir}/wf_{wf_idx}_snapshot_at_time_0.npy"
            np.save(initial_path, cp.asnumpy(wf.psi))
            self.wave_values[wf_idx].append(initial_path)

        # Evolve over time steps
        for step in range(self.num_steps):
            # Perform the evolution step based on the specified order
            if self.order == 2:
                wave_functions = self.evolve_wavefunction_split_step_o2(
                    wave_functions, step_index=step, total_steps=self.num_steps
                )
            elif self.order == 4:
                wave_functions = self.evolve_wavefunction_split_step_o4(
                    wave_functions, step_index=step, total_steps=self.num_steps
                )
            elif self.order == 6:
                wave_functions = self.evolve_wavefunction_split_step_o6(
                    wave_functions, step_index=step, total_steps=self.num_steps
                )
            else:
                raise ValueError(f"Order {self.order} is not supported. Use 2, 4, or 6.")

            # Save wave functions at specified intervals
            if step % save_every == 0 and step > 0:
                print(f"Still working... Step {step} out of {self.num_steps}")

                # Calculate current time
                current_time = step * self.h

                # Save all wave functions at this time
                for wf_idx, wf in enumerate(wave_functions):
                    snapshot_path = f"{save_dir}/wf_{wf_idx}_snapshot_at_time_{current_time:.6f}.npy"
                    np.save(snapshot_path, cp.asnumpy(wf.psi))
                    self.wave_values[wf_idx].append(snapshot_path)

                self.accessible_times.append(current_time)

            # Free memory
            cp.get_default_memory_pool().free_all_blocks()

        # Ensure the last state is saved if it wasn't already
        if (self.num_steps - 1) % save_every != 0:
            final_time = self.total_time
            for wf_idx, wf in enumerate(wave_functions):
                final_path = f"{save_dir}/wf_{wf_idx}_snapshot_at_time_{final_time:.6f}.npy"
                np.save(final_path, cp.asnumpy(wf.psi))
                self.wave_values[wf_idx].append(final_path)
            self.accessible_times.append(final_time)

        # Save metadata
        self._save_metadata(save_dir)

        # Cleanup and finalize
        self._finalize_evolution()

        return wave_functions

    def evolve_wavefunction_split_step_o2(self, wave_functions, step_index, total_steps):
        """
        Evolve all wave functions using the split-step Fourier method (order 2).

        Parameters:
            wave_functions (list): List of wave function instances
            step_index (int): Index of the current step (starting at 0)
            total_steps (int): Total number of steps in the simulation

        Returns:
            list: List of evolved wave function instances
        """
        is_first_step = (step_index == 0)
        is_last_step = (step_index == total_steps - 1)


        #density_at_step = self.compute_density(summed_wf)/len(wave_functions)

        # Kick step for all wave functions
        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step)

        # Drift step for all wave functions
        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi)

        # Final kick step for last iteration
        if is_last_step:
            for wf in wave_functions:
                wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step)

        # Track maximum values if enabled
        if self.save_max_vals:
            # Calculate sum of all wave functions
            summed_wf = wave_functions[0].psi.copy()
            for wf in wave_functions[1:]:
                summed_wf += wf.psi

            self.max_wave_vals_during_evolution[step_index] = float(abs(summed_wf).max())


        return wave_functions

    def evolve_wavefunction_split_step_o4(self, wave_functions, step_index, total_steps):
        """
        Evolve all wave functions using the 4th-order split-step Fourier method.

        Parameters:
            wave_functions (list): List of wave function instances
            step_index (int): Index of the current step (starting at 0)
            total_steps (int): Total number of steps in the simulation

        Returns:
            list: List of evolved wave function instances
        """
        is_first_step = (step_index == 0)
        is_last_step = (step_index == total_steps - 1)

        # First half-step potential with v2 coefficient
        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, False, 'v2')

        # Main sequence of operations using pre-calculated propagators
        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 't2')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, False, 'v1')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 't1')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, False, 'v0')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 't1')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, False, is_last_step, 'v1')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 't2')

        # Last half-step potential with v2 coefficient
        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, False, is_last_step, 'v2')

        # Track maximum values if enabled
        if self.save_max_vals:
            # Calculate sum of all wave functions
            summed_wf = wave_functions[0].psi.copy()
            for wf in wave_functions[1:]:
                summed_wf += wf.psi
            self.max_wave_vals_during_evolution[step_index] = float(abs(summed_wf).max())

        return wave_functions

    def evolve_wavefunction_split_step_o6(self, wave_functions, step_index, total_steps):
        """
        Evolve all wave functions using the 6th-order split-step Fourier method.

        Parameters:
            wave_functions (list): List of wave function instances
            step_index (int): Index of the current step (starting at 0)
            total_steps (int): Total number of steps in the simulation

        Returns:
            list: List of evolved wave function instances
        """
        is_first_step = (step_index == 0)
        is_last_step = (step_index == total_steps - 1)

        # Main sequence of operations using pre-calculated propagators
        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 'v1')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step, 't1')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 'v2')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step, 't2')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 'v3')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step, 't3')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 'v4')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step, 't4')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 'v4')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step, 't3')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 'v3')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step, 't2')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 'v2')

        for wf in wave_functions:
            wf.psi = self.kick_step(wf.psi, is_first_step, is_last_step, 't1')

        for wf in wave_functions:
            wf.psi = self.drift_step(wf.psi, 'v1')

        # Track maximum values if enabled
        if self.save_max_vals:
            # Calculate sum of all wave functions
            summed_wf = wave_functions[0].psi.copy()
            for wf in wave_functions[1:]:
                summed_wf += wf.psi
            self.max_wave_vals_during_evolution[step_index] = float(abs(summed_wf).max())

        return wave_functions

    def get_wave_function_at_time(self, time, wf_index=None):
        """
        Retrieve the wave function values at a given time.
        If wf_index is specified, returns that specific wave function.
        If wf_index is None, returns the sum of all wave functions.
        Loads data from disk instead of memory.

        Parameters:
            time (float): The time at which to retrieve the wave function values
            wf_index (int, optional): Index of the wave function to retrieve.
                                    If None, returns sum of all wave functions.

        Returns:
            cp.ndarray: The wave function (or sum of all wave functions) at the given time
        """
        # Find the closest time in the accessible times list
        closest_time_index = min(range(len(self.accessible_times)),
                                 key=lambda i: abs(self.accessible_times[i] - time))

        if wf_index is not None:
            # Return specific wave function
            if wf_index >= self.num_wave_functions:
                raise ValueError(
                    f"Wave function index {wf_index} is out of range. Only {self.num_wave_functions} wave functions available.")

            file_path = self.wave_values[wf_index][closest_time_index]
            wave_function = cp.array(np.load(file_path))
            return wave_function
        else:
            # Return sum of all wave functions
            summed_wave_function = None
            for wf_idx in range(self.num_wave_functions):
                file_path = self.wave_values[wf_idx][closest_time_index]
                wave_function = cp.array(np.load(file_path))

                if summed_wave_function is None:
                    summed_wave_function = wave_function.copy()
                else:
                    summed_wave_function += wave_function

            return summed_wave_function



    def _calculate_coefficients_and_propagators(self):
        """
        Calculate the coefficients for the requested split-step method order
        and pre-compute all static propagators to optimize performance.
        """
        if self.order == 2:
            # For order 2, just use full time step
            self.coefficients = {'full': 1.0}

            # Pre-calculate standard propagators
            self.static_propagators = {
                'full': self.propagator.compute_static_potential_propagator(
                    self.simulation.static_potential, time_factor=1.0
                )
            }
            self.kinetic_propagators = {
                'full': self.propagator.compute_kinetic_propagator(time_factor=1.0)
            }

        elif self.order == 4:
            # Calculate coefficients for 4th-order method
            v1 = (121.0 / 3924.0) * (12.0 - cp.sqrt(471.0))
            w = cp.sqrt(3.0 - 12.0 * v1 + 9 * v1 * v1)
            t2 = 0.25 * (1.0 - cp.sqrt((9.0 * v1 - 4.0 + 2 * w) / (3.0 * v1)))
            t1 = 0.5 - t2
            v2 = (1.0 / 6.0) - 4 * v1 * t1 * t1
            v0 = 1.0 - 2.0 * (v1 + v2)

            # Store coefficients
            self.coefficients = {
                'v0': v0,
                'v1': v1,
                'v2': v2,
                't1': t1,
                't2': t2
            }

            # Define the set of propagators to pre-calculate
            potential_coeffs = [('v0', v0), ('v1', v1), ('v2', v2)]
            kinetic_coeffs = [('t1', t1), ('t2', t2)]

        elif self.order == 6:
            w1 = -0.117767998417887e1
            w2 = 0.235573213359357e0
            w3 = 0.784513610477560e0
            w0 = 1 - 2 * (w1 + w2 + w3)

            # Potential coefficients
            v1 = w3 / 2
            v2 = (w2 + w3) / 2
            v3 = (w1 + w2) / 2
            v4 = (w0 + w1) / 2

            # Kinetic coefficients
            t1 = w3
            t2 = w2
            t3 = w1
            t4 = w0

            # Store coefficients
            self.coefficients = {
                'v1': v1,
                'v2': v2,
                'v3': v3,
                'v4': v4,
                't1': t1,
                't2': t2,
                't3': t3,
                't4': t4,
            }

            # Define the set of propagators to pre-calculate
            potential_coeffs = [
                ('t1', t1), ('t2', t2),
                ('t3', t3), ('t4', t4),
            ]
            kinetic_coeffs = [
                ('v1', v1), ('v2', v2), ('v3', v3),
                ('v4', v4),
            ]

        else:
            raise ValueError(f"Order {self.order} is not supported. Use 2, 4, or 6.")

        # Pre-calculate all needed propagators if not already done for order 2
        if self.order > 2:
            # Pre-calculate static potential propagators
            self.static_propagators = {}
            for label, factor in potential_coeffs:
                self.static_propagators[label] = self.propagator.compute_static_potential_propagator(
                    self.simulation.static_potential, time_factor=factor
                )

            # Pre-calculate kinetic propagators
            self.kinetic_propagators = {}
            for label, factor in kinetic_coeffs:
                self.kinetic_propagators[label] = self.propagator.compute_kinetic_propagator(
                    time_factor=factor
                )

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
            f.write(f"Method order: {self.order}\n")
            f.write(f"Number of wave functions: {self.num_wave_functions}\n")
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


    def kick_step(self, psi, is_first_step, is_last_step, time_factor_key='full'):
        """
        Takes care of the propagation by potential.

        Parameters:
            psi (cp.ndarray): Wave function
            is_first_step (bool): Whether this is the first step in evolution
            is_last_step (bool): Whether this is the last step in evolution
            time_factor_key (str): Key for pre-calculated propagators

        Returns:
            cp.ndarray: Evolved wave function after potential propagation
        """
        # Get coefficient for this step
        time_factor = self.coefficients[time_factor_key]

        # Calculate gravity propagator (which depends on psi)
        if is_first_step or is_last_step:
            gravity_propagator = self.propagator.compute_gravity_propagator(
                psi, first_step=is_first_step, last_step=is_last_step, time_factor=time_factor
            )
        else:
            gravity_propagator = self.propagator.compute_gravity_propagator(
                psi, time_factor=time_factor
            )

        # Use pre-calculated static potential propagator
        static_propagator = self.static_propagators[time_factor_key]

        # Apply both propagators
        full_potential_propagator = static_propagator * gravity_propagator
        return psi * full_potential_propagator

    def drift_step(self, psi, time_factor_key='full'):
        """
        Kinetic part of the evolution.

        Parameters:
            psi (cp.ndarray): Wave function
            time_factor_key (str): Key for pre-calculated propagators

        Returns:
            cp.ndarray: Evolved wave function after kinetic propagation
        """
        # Transform to momentum space
        psi_k = cp.fft.fftn(psi)

        # Use pre-calculated kinetic propagator
        kinetic_propagator = self.kinetic_propagators[time_factor_key]

        # Apply propagator and transform back
        psi_k *= kinetic_propagator
        return cp.fft.ifftn(psi_k)


