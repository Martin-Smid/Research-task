import cupy as cp
import numpy as np
import os
import datetime
import pandas as pd
from resources.Functions.system_fucntions import plot_max_values_on_N
import matplotlib.pyplot as plt

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

        # Evolution data storage
        self.wave_values = []
        self.accessible_times = []
        self.max_wave_vals_during_evolution = {}
        self.snapshot_directory = None
        self.num_wave_functions = 0

        self.energy_log = []


        # Pre-compute propagators and coefficients
        self._calculate_coefficients_and_propagators()

    def evolve(self, wave_functions, save_every=1):
        """
        Perform the full time evolution for multiple wave functions.

        Parameters:
            wave_functions (list): List of wave function instances, each with .psi attribute
            save_every (int): Frequency of saving the wave function values

        Returns:
            list: List of wave function instances with evolved .psi attributes
        """
        save_every = max(1, save_every)
        self.num_wave_functions = len(wave_functions)


        for wf in wave_functions:
            wf.psi = cp.asarray(wf.psi)

            # Setup save directory and initialize storage
        self._setup_evolution_storage(wave_functions)

        total_density = self._compute_total_density(wave_functions)

        mass_diff = (
                            (abs(total_density).sum()
                             * (self.simulation.dV ** 3)
                             )/ wave_functions[0].soliton_mass) - self.simulation.num_of_w_vects_in_sim
        if (mass_diff > 1e-2):
            print(f"mass diff {mass_diff} is greater than 1e-2, might want to increase the resolution")
        else:
            print(f"mass diff is {mass_diff:.6f} Msun")

        del total_density


        # Main evolution loop
        for step in range(self.num_steps):

            total_density = self._compute_total_density(wave_functions)
            self.total_energy = 0
            save_step=False
            current_time = step * self.h
            self.compute_kinetic_energy(wave_functions)
            self._compute_potential_energy(wave_functions, total_density, current_time)
            # Save snapshots at specified intervals
            if step % save_every == 0 and step > 0:
                save_step = True
            wave_functions = self._perform_evolution_step(wave_functions,total_density, step, save_step)

            if step % save_every == 0 and step > 0:
                self.compute_radial_density_profile(total_density,current_time, Nbins=100)

                self._save_snapshots(wave_functions, step, save_every)


            # Memory cleanup
            cp.get_default_memory_pool().free_all_blocks()

        # Save final state if needed
        self._save_final_state(wave_functions, save_every)

        # Cleanup and finalize
        self._finalize_evolution()

        return wave_functions

    def _setup_evolution_storage(self, wave_functions):
        """Setup directory structure and save initial states."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"resources/data/simulation_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        self.snapshot_directory = save_dir

        # Initialize storage for each wave function
        self.wave_values = [[] for _ in range(self.num_wave_functions)]
        self.accessible_times.append(0)

        # Save initial states
        for wf_idx, wf in enumerate(wave_functions):
            initial_path = f"{save_dir}/wf_{wf_idx}_snapshot_at_time_0.npy"
            np.save(initial_path, cp.asnumpy(wf.psi))
            self.wave_values[wf_idx].append(initial_path)

    def _perform_evolution_step(self, wave_functions,total_density, step,save_step):
        """Perform a single evolution step based on the order."""
        evolution_methods = {
            2: self._evolve_order_2,
            4: self._evolve_order_4,
            6: self._evolve_order_6
        }

        if self.order not in evolution_methods:
            raise ValueError(f"Order {self.order} is not supported. Use 2, 4, or 6.")

        is_first = (step == 0)
        is_last = (step == self.num_steps - 1)

        wave_functions = evolution_methods[self.order](wave_functions,total_density, is_first, is_last,save_step)

        # Track maximum values if enabled
        if self.save_max_vals:
            self._track_max_values(wave_functions, step)



        return wave_functions

    def _evolve_order_2(self, wave_functions,total_density ,is_first, is_last,save_step):
        """Second-order split-step evolution."""


        # Kick step
        self._kick_all_wave_functions(wave_functions, total_density, is_first, is_last)

        # Drift step
        compute_kin_en = save_step

        self._drift_all_wave_functions(wave_functions)




        # Final kick for last step
        if is_last:
            total_density = self._compute_total_density(wave_functions)
            self._kick_all_wave_functions(wave_functions, total_density, is_first, is_last)


        return wave_functions

    def _evolve_order_4(self, wave_functions, is_first, is_last,save_step):
        """Fourth-order split-step evolution."""
        # Step sequence for 4th order
        steps = [
            ('kick', 'v2'), ('drift', 't2'), ('kick', 'v1'), ('drift', 't1'),
            ('kick', 'v0'), ('drift', 't1'), ('kick', 'v1'), ('drift', 't2'),
            ('kick', 'v2')
        ]

        for i, (operation, coeff_key) in enumerate(steps):
            if operation == 'kick':
                total_density = self._compute_total_density(wave_functions)

                first_op = is_first and i == 0
                last_op = is_last and i == len(steps) - 1
                self._kick_all_wave_functions(wave_functions, total_density, first_op, last_op, coeff_key)

            else:  # drift
                is_last_drift = save_step and (i == len(steps) - 2)
                self._drift_all_wave_functions(wave_functions, time_factor_key=coeff_key)


        return wave_functions

    def _evolve_order_6(self, wave_functions, is_first, is_last,save_step):
        """Sixth-order split-step evolution."""
        # Step sequence for 6th order (symmetric)
        steps = [
            ('drift', 'v1'), ('kick', 't1'), ('drift', 'v2'), ('kick', 't2'),
            ('drift', 'v3'), ('kick', 't3'), ('drift', 'v4'), ('kick', 't4'),
            ('drift', 'v4'), ('kick', 't3'), ('drift', 'v3'), ('kick', 't2'),
            ('drift', 'v2'), ('kick', 't1'), ('drift', 'v1')
        ]

        kick_indices = [i for i, (op, _) in enumerate(steps) if op == 'kick']

        for i, (operation, coeff_key) in enumerate(steps):
            if operation == 'kick':
                total_density = self._compute_total_density(wave_functions)
                first_kick = is_first and i == kick_indices[0]
                last_kick = is_last and i == kick_indices[-1]
                self._kick_all_wave_functions(wave_functions, total_density, first_kick, last_kick, coeff_key)

            else:  # drift
                is_last_drift = save_step and (i == len(steps) - 1)


                self._drift_all_wave_functions(wave_functions, time_factor_key=coeff_key)



        return wave_functions

    def _kick_all_wave_functions(self, wave_functions, total_density, is_first_step, is_last_step,
                                 time_factor_key='full'):
        """Apply kick step to all wave functions with shared density."""
        time_factor = self.coefficients[time_factor_key]

        # Pre-calculate shared components
        static_propagator = self.static_propagators[time_factor_key]

        for wf in wave_functions:
            # Calculate gravity propagator for this specific wave function
            if is_first_step or is_last_step:
                gravity_propagator = self.propagator.compute_gravity_propagator(
                    wf.psi, total_density, first_step=is_first_step,
                    last_step=is_last_step, time_factor=time_factor
                )
            else:
                gravity_propagator = self.propagator.compute_gravity_propagator(
                    wf.psi, total_density, time_factor=time_factor
                )

            full_potential_propagator = static_propagator * gravity_propagator
            wf.psi *= full_potential_propagator

    def _drift_all_wave_functions(self, wave_functions, time_factor_key='full'):

        """Apply drift step to all wave functions."""
        kinetic_propagator = self.kinetic_propagators[time_factor_key]

        for wf in wave_functions:
            # Transform to momentum space
            psi_k = cp.fft.fftn(wf.psi)
            # Apply kinetic propagator
            psi_k *= kinetic_propagator
            # Transform back
            wf.psi = cp.fft.ifftn(psi_k)



    def _compute_total_density(self, wave_functions):
        """Calculate the total density ρ = Σ|ψᵢ|² from all wave functions."""
        total_density = cp.zeros_like(wave_functions[0].psi, dtype=cp.float64)
        for wf in wave_functions:
            density_i = wf.calculate_density()
            total_density += density_i

        return total_density

    def _track_max_values(self, wave_functions, step):
        """Track maximum values during evolution."""
        total_density = self._compute_total_density(wave_functions)

        self.max_wave_vals_during_evolution[step] = float(abs(total_density).max())

    def _save_snapshots(self, wave_functions, step, save_every):
        """Save wave function snapshots at current step."""
        print(f"Still working... Step {step} out of {self.num_steps}")

        current_time = step * self.h
        for wf_idx, wf in enumerate(wave_functions):
            snapshot_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_{current_time:.6f}.npy"
            np.save(snapshot_path, cp.asnumpy(wf.psi))
            self.wave_values[wf_idx].append(snapshot_path)

        self.accessible_times.append(current_time)

    def _save_final_state(self, wave_functions, save_every):
        """Save final state if it wasn't already saved."""
        if (self.num_steps - 1) % save_every != 0:
            final_time = self.total_time
            for wf_idx, wf in enumerate(wave_functions):
                final_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_{final_time:.6f}.npy"
                np.save(final_path, cp.asnumpy(wf.psi))
                self.wave_values[wf_idx].append(final_path)
            self.accessible_times.append(final_time)

        # Save metadata
        self._save_metadata()

    def get_wave_function_at_time(self, time, wf_index=None):
        """
        Retrieve wave function(s) at a given time.

        Parameters:
            time (float): The time at which to retrieve the wave function values
            wf_index (int, optional): Index of specific wave function.
                                    If None, returns sum of all wave functions.

        Returns:
            cp.ndarray: The wave function (or sum) at the given time
        """
        # Find closest time
        closest_time_index = min(range(len(self.accessible_times)),
                                 key=lambda i: abs(self.accessible_times[i] - time))

        if wf_index is not None:
            if wf_index >= self.num_wave_functions:
                raise ValueError(f"Wave function index {wf_index} out of range. "
                                 f"Only {self.num_wave_functions} wave functions available.")

            file_path = self.wave_values[wf_index][closest_time_index]
            return cp.array(np.load(file_path))
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
        """Calculate coefficients and pre-compute static propagators."""
        if self.order == 2:
            self.coefficients = {'full': 1.0}
            potential_coeffs = [('full', 1.0)]
            kinetic_coeffs = [('full', 1.0)]

        elif self.order == 4:
            # 4th-order coefficients
            v1 = (121.0 / 3924.0) * (12.0 - cp.sqrt(471.0))
            w = cp.sqrt(3.0 - 12.0 * v1 + 9 * v1 * v1)
            t2 = 0.25 * (1.0 - cp.sqrt((9.0 * v1 - 4.0 + 2 * w) / (3.0 * v1)))
            t1 = 0.5 - t2
            v2 = (1.0 / 6.0) - 4 * v1 * t1 * t1
            v0 = 1.0 - 2.0 * (v1 + v2)

            self.coefficients = {'v0': v0, 'v1': v1, 'v2': v2, 't1': t1, 't2': t2}
            potential_coeffs = [('v0', v0), ('v1', v1), ('v2', v2)]
            kinetic_coeffs = [('t1', t1), ('t2', t2)]

        elif self.order == 6:
            # 6th-order coefficients
            w1, w2, w3 = -0.117767998417887e1, 0.235573213359357e0, 0.784513610477560e0
            w0 = 1 - 2 * (w1 + w2 + w3)

            v1, v2, v3, v4 = w3 / 2, (w2 + w3) / 2, (w1 + w2) / 2, (w0 + w1) / 2
            t1, t2, t3, t4 = w3, w2, w1, w0

            self.coefficients = {
                'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4,
                't1': t1, 't2': t2, 't3': t3, 't4': t4
            }
            potential_coeffs = [('t1', t1), ('t2', t2), ('t3', t3), ('t4', t4)]
            kinetic_coeffs = [('v1', v1), ('v2', v2), ('v3', v3), ('v4', v4)]

        else:
            raise ValueError(f"Order {self.order} not supported. Use 2, 4, or 6.")

        # Pre-calculate static propagators
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

    def _save_metadata(self):
        """Save simulation metadata."""
        with open(f"{self.snapshot_directory}/metadata.txt", "w") as f:
            f.write(f"Total steps: {self.num_steps}\n")
            f.write(f"Time step: {self.h}\n")
            f.write(f"Total time: {self.total_time}\n")
            f.write(f"Method order: {self.order}\n")
            f.write(f"Number of wave functions: {self.num_wave_functions}\n")
            f.write("Accessible times:\n")
            f.write(",".join([str(t) for t in self.accessible_times]))

    def _finalize_evolution(self):
        """Cleanup and finalize evolution process."""
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
        energy_path = os.path.join(self.snapshot_directory, "energy.txt")
        df = pd.DataFrame(self.energy_log)
        df.to_csv(energy_path, index=False, float_format="%.6e")
        print(f"Energy log saved to: {energy_path}")

        print("Evolution completed successfully")
        print(f"Saved times are {self.accessible_times}")
        cp.get_default_memory_pool().free_all_blocks()

    def _save_max_values(self):
        """Save maximum values during evolution to CSV, including resolution and spin."""
        self.max_vals_filename = "resources/data/max_values.csv"
        header = [
            "# This file contains the maximum values of wave functions at specific steps along the time evolution.",
            "# The first row contains the corresponding N - the resolution of the saved simulation.",
            "# The second row contains the spin of the wavefunction in the corresponding simulation.",
            "# The first column is the time step, and subsequent columns are max values for a given simulation.",
        ]

        # Připrav vícenásobné sloupce (MultiIndex)
        resolution = int(self.simulation.N)
        spin = self.simulation.spin  # musí být uložen v Simulation

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


    def _compute_potential_energy(self,wave_functions,total_density,current_time):
        """
        Compute the potential energy:
        W = ∫ V(x) · Tr(ψ†ψ) dV = ∫ V(x) · total_density(x) dV
        """
        dx = np.prod(self.simulation.dx)
        #potential = self.simulation.static_potential(self.simulation) DODĚLAT AŽ NEBUDU LÍNEJ
        total_density = total_density
        self.last_potential_energy = cp.sum(total_density) * dx  # TODO: use real potential later
        K = self.last_kinetic_energy
        W = self.last_potential_energy
        E = K + W
        W_over_E = W / cp.abs(E)

        t = current_time

        self.energy_log.append({
            "time": float(t),
            "K": float(K),
            "W": float(W),
            "E": float(E),
            "W/|E|": float(W_over_E)
        })

    def compute_kinetic_energy(self,wave_functions):
        """
        Compute the kinetic energy of the wavefunction using 'gradient'  method.
        """
        dx = np.prod(self.simulation.dx)
        k_space = self.simulation.k_space
        kinetic_energy = 0
        for wf in wave_functions:
            partials_squared = 0
            for i, k_i in enumerate(k_space):
                grad_i = cp.fft.ifftn(-1j * k_i * cp.fft.fftn(wf.psi))
                partials_squared += cp.abs(grad_i) ** 2
            part_kin_en = 0.5 * self.simulation.h_bar_tilde ** 2 * cp.sum(partials_squared) * dx
            kinetic_energy += part_kin_en

        self.last_kinetic_energy = kinetic_energy

    def compute_radial_density_profile(self, total_density, current_time, Nbins=100):
        """
        Compute, plot, and save the spherically averaged radial density profile.

        Parameters:
            total_density (cp.ndarray): The total density |ψ|² of the system
            current_time (float): The simulation time to use in file naming
            Nbins (int): Number of radial bins
        """

        dx = self.simulation.dx
        BoxSize = [b[1] - b[0] for b in self.simulation.boundaries]

        rho = total_density

        # Find center of mass (maximum density)
        ix, iy, iz = cp.asnumpy(cp.argwhere(rho == rho.max())[0])
        rho_max = cp.asnumpy(rho.max())

        # Get coordinates
        grid_x = cp.asarray(self.simulation.grids[0])
        grid_y = cp.asarray(self.simulation.grids[1])
        grid_z = cp.asarray(self.simulation.grids[2])
        dx_, dy_, dz_ = dx

        # Physical coordinates of the center (from 3D meshgrid)
        center_x = grid_x[ix, iy, iz]
        center_y = grid_y[ix, iy, iz]
        center_z = grid_z[ix, iy, iz]

        # Shifted coordinates with periodic boundary correction
        Delta_x = grid_x - center_x
        Delta_y = grid_y - center_y
        Delta_z = grid_z - center_z

        # Apply minimum image convention for periodic boundaries
        Delta_x = Delta_x - BoxSize[0] * cp.round(Delta_x / BoxSize[0])
        Delta_y = Delta_y - BoxSize[1] * cp.round(Delta_y / BoxSize[1])
        Delta_z = Delta_z - BoxSize[2] * cp.round(Delta_z / BoxSize[2])

        r = cp.sqrt(Delta_x ** 2 + Delta_y ** 2 + Delta_z ** 2)

        # To CPU for binning
        r_cpu = cp.asnumpy(r)
        rho_cpu = cp.asnumpy(rho)

        # Create radial bins
        bins = np.linspace(0, r_cpu.max(), Nbins)

        # Calculate mean density in each radial bin
        counts = []
        for i in range(len(bins) - 1):
            mask = (r_cpu > bins[i]) & (r_cpu <= bins[i + 1])
            if np.any(mask):
                counts.append(rho_cpu[mask].mean())
            else:
                counts.append(0.0)  # Handle empty bins

        rho_avg = np.array(counts)

        # Normalize
        if rho_avg.max() > 0:
            rho_avg /= rho_avg.max()

        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        save_dir = os.path.join(self.snapshot_directory, "density_profiles")
        os.makedirs(save_dir, exist_ok=True)

        # Save data
        filename = f"density_data_at_time_{current_time:.6f}.csv"
        save_path = os.path.join(save_dir, filename)

        np.savetxt(save_path, np.column_stack((bin_centers, rho_avg)),
                   delimiter=",", header="radius,density", comments="")

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(bin_centers, rho_avg, lw=2)
        plt.xlabel("Radius r [kpc]")
        plt.ylabel("Spherically Averaged Density ρ(r)")
        plt.title(f"Radial Density Profile at t = {current_time:.2f}")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
