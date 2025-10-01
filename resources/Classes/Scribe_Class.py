import os
import datetime
import numpy as np
import cupy as cp
import pandas as pd


class Scribe:
    """
    Handles all data writing operations for the simulation.
    Separates I/O concerns from evolution logic.
    """

    def __init__(self, simulation):
        """
        Initialize the Scribe.

        Parameters:
            simulation: Simulation_Class instance (for accessing grid info, etc.)
        """
        self.simulation = simulation
        self.snapshot_directory = None
        self.max_locations_path = None

        # Data storage
        self.wave_values = []
        self.accessible_times = []
        self.energy_log = []
        self.max_location_log = []
        self.max_wave_vals_during_evolution = {}

    def setup_directories(self, num_wave_functions):
        """
        Create directory structure for saving data.

        Parameters:
            num_wave_functions: Number of wave functions to track

        Returns:
            str: Path to the snapshot directory
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"resources/data/simulation_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        self.snapshot_directory = save_dir

        self.max_vals_filename = "resources/data/max_values.csv"

        # Initialize storage for each wave function
        self.wave_values = [[] for _ in range(num_wave_functions)]
        self.accessible_times.append(0)

        # Initialize max location logger
        self.max_locations_path = os.path.join(self.snapshot_directory, "max_locations.txt")
        with open(self.max_locations_path, "w") as f:
            f.write("# time, ix, iy, iz, x, y, z\n")

        return save_dir

    def save_initial_states(self, wave_functions):
        """
        Save initial wave function states.

        Parameters:
            wave_functions: List of wave function objects
        """
        for wf_idx, wf in enumerate(wave_functions):
            initial_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_0.npy"
            np.save(initial_path, cp.asnumpy(wf.psi))
            self.wave_values[wf_idx].append(initial_path)

    def save_snapshots(self, wave_functions, step, h):
        """
        Save wave function snapshots at current step.

        Parameters:
            wave_functions: List of wave function objects
            step: Current step number
            h: Time step size
        """
        current_time = step * h
        for wf_idx, wf in enumerate(wave_functions):
            snapshot_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_{current_time:.6f}.npy"
            np.save(snapshot_path, cp.asnumpy(wf.psi))
            self.wave_values[wf_idx].append(snapshot_path)

        self.accessible_times.append(current_time)

    def save_final_state(self, wave_functions, num_steps, save_every, h, total_time):
        """
        Save final state if it wasn't already saved.

        Parameters:
            wave_functions: List of wave function objects
            num_steps: Total number of steps
            save_every: Save frequency
            h: Time step size
            total_time: Total simulation time
        """
        if (num_steps - 1) % save_every != 0:
            final_time = total_time
            for wf_idx, wf in enumerate(wave_functions):
                final_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_{final_time:.6f}.npy"
                np.save(final_path, cp.asnumpy(wf.psi))
                self.wave_values[wf_idx].append(final_path)
            self.accessible_times.append(final_time)

    def save_metadata(self, num_steps, h, total_time, order, num_wave_functions):
        """
        Save simulation metadata.

        Parameters:
            num_steps: Total number of steps
            h: Time step size
            total_time: Total simulation time
            order: Method order
            num_wave_functions: Number of wave functions
        """
        with open(f"{self.snapshot_directory}/metadata.txt", "w") as f:
            f.write(f"Total steps: {num_steps}\n")
            f.write(f"Time step: {h}\n")
            f.write(f"Total time: {total_time}\n")
            f.write(f"Method order: {order}\n")
            f.write(f"Number of wave functions: {num_wave_functions}\n")
            f.write("Accessible times:\n")
            f.write(",".join([str(t) for t in self.accessible_times]))

    def record_max_location(self, ix, iy, iz, time_value):
        """
        Save the grid-index location of the current max density and its coordinates.

        Parameters:
            ix, iy, iz: Indices of the max-density cell
            time_value: Physical time corresponding to this step
        """
        # Get physical coordinates at that index
        gx, gy, gz = self.simulation.grids
        x = float(gx[ix, iy, iz])
        y = float(gy[ix, iy, iz])
        z = float(gz[ix, iy, iz])

        # Cache in memory and append to file
        self.max_location_log.append((float(time_value), int(ix), int(iy), int(iz), x, y, z))
        with open(self.max_locations_path, "a") as f:
            f.write(f"{time_value:.9e}, {ix:d}, {iy:d}, {iz:d}, {x:.9e}, {y:.9e}, {z:.9e}\n")

    def log_energy(self, time, K, W):
        """
        Log energy values at a given time.

        Parameters:
            time: Current time
            K: Kinetic energy
            W: Potential energy
        """
        E = K + W
        W_over_E = W / cp.abs(E)

        self.energy_log.append({
            "time": float(time),
            "K": float(K),
            "W": float(W),
            "E": float(E),
            "W/|E|": float(W_over_E)
        })

    def save_energy_log(self):
        """Save energy log to CSV file."""
        energy_path = os.path.join(self.snapshot_directory, "energy.txt")
        df = pd.DataFrame(self.energy_log)
        df.to_csv(energy_path, index=False, float_format="%.6e")
        print(f"Energy log saved to: {energy_path}")

    def save_radial_density_profile(self, bin_centers, rho_avg, current_time):
        """
        Save radial density profile data to CSV.

        Parameters:
            bin_centers: Array of radial bin centers
            rho_avg: Array of average densities per bin
            current_time: Current simulation time
        """
        save_dir = os.path.join(self.snapshot_directory, "density_profiles")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"density_data_at_time_{current_time:.6f}.csv")
        np.savetxt(save_path, np.column_stack((bin_centers, rho_avg)),
                   delimiter=",", header="radius,density", comments="")

    def track_max_value(self, step, max_value):
        """
        Track maximum value during evolution.

        Parameters:
            step: Current step number
            max_value: Maximum value to track
        """
        self.max_wave_vals_during_evolution[step] = float(max_value)

    def save_max_values(self, resolution, spin):
        """
        Save maximum values during evolution to CSV.

        Parameters:
            resolution: Grid resolution (N)
            spin: Spin value of the wavefunction
        """

        header = [
            "# This file contains the maximum values of wave functions at specific steps along the time evolution.",
            "# The first row contains the corresponding N - the resolution of the saved simulation.",
            "# The second row contains the spin of the wavefunction in the corresponding simulation.",
            "# The first column is the time step, and subsequent columns are max values for a given simulation.",
        ]

        # Prepare multi-column format
        col_tuples = [(f"N{resolution}", f"s={spin}")]
        columns = pd.MultiIndex.from_tuples(col_tuples)

        new_data = pd.DataFrame.from_dict(
            self.max_wave_vals_during_evolution,
            orient="index",
            columns=columns
        )

        if os.path.exists(self.max_vals_filename):
            existing_data = pd.read_csv(self.max_vals_filename, header=[0, 1], index_col=0, comment="#")
            updated_data = pd.concat([existing_data, new_data], axis=1)

            with open(self.max_vals_filename, "w") as file:
                file.write("\n".join(header) + "\n")
            updated_data.to_csv(self.max_vals_filename)
        else:
            with open(self.max_vals_filename, "w") as file:
                file.write("\n".join(header) + "\n")
            new_data.to_csv(self.max_vals_filename)

        print(f"{self.max_vals_filename} saved")

    def get_wave_function_at_time(self, time, wf_index=None, num_wave_functions=None):
        """
        Retrieve wave function(s) at a given time.

        Parameters:
            time: The time at which to retrieve the wave function values
            wf_index: Index of specific wave function. If None, returns sum of all.
            num_wave_functions: Total number of wave functions (needed if wf_index is None)

        Returns:
            cp.ndarray: The wave function (or sum) at the given time
        """
        # Find closest time
        closest_time_index = min(range(len(self.accessible_times)),
                                 key=lambda i: abs(self.accessible_times[i] - time))

        if wf_index is not None:
            file_path = self.wave_values[wf_index][closest_time_index]
            return cp.array(np.load(file_path))
        else:
            # Return sum of all wave functions
            if num_wave_functions is None:
                num_wave_functions = len(self.wave_values)

            summed_wave_function = None
            for wf_idx in range(num_wave_functions):
                file_path = self.wave_values[wf_idx][closest_time_index]
                wave_function = cp.array(np.load(file_path))

                if summed_wave_function is None:
                    summed_wave_function = wave_function.copy()
                else:
                    summed_wave_function += wave_function

            return summed_wave_function