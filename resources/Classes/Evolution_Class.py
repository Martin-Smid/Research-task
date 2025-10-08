import cupy as cp
import numpy as np
from resources.Functions.system_fucntions import plot_max_values_on_N
from resources.Classes.Scribe_Class import Scribe
import os

#np.random.seed(1)

class Evolution_Class:
    """
    Class to handle the time evolution of multiple wave functions in the simulation.
    Uses the split-step Fourier method for propagation with orders 2, 4, or 6.
    Each wave function should have a .psi attribute containing the wave data.
    """

    def __init__(self, simulation, propagator,  order=2):
        """
        Initialize the evolution handler.

        Parameters:
            simulation: Simulation_Class instance
            propagator: Propagator_Class instance
            scribe: Scribe instance for handling data writing
            order: Order of the split-step method (2, 4, or 6)
        """
        self.simulation = simulation
        self.propagator = propagator
        self.scribe = Scribe(self.simulation)
        self.h = simulation.h
        self.num_steps = simulation.num_steps
        self.total_time = simulation.total_time
        self.save_max_vals = simulation.save_max_vals
        self.order = order
        self.num_wave_functions = 0

        # Energy tracking (computation, not storage)
        self.last_kinetic_energy = 0
        self.last_potential_energy = 0

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

        # Convert to CuPy arrays
        for wf in wave_functions:
            wf.psi = cp.asarray(wf.psi)

        # Setup directories and save initial state
        self.scribe.setup_directories(self.num_wave_functions)
        self.scribe.save_initial_states(wave_functions)

        # Initial diagnostics
        total_density = self._compute_total_density(wave_functions)
        current_time = 0
        ix, iy, iz = cp.asnumpy(cp.argwhere(total_density == total_density.max())[0])

        self.scribe.record_max_location(int(ix), int(iy), int(iz), float(current_time))
        self._compute_and_save_radial_profile(total_density, current_time, ix, iy, iz)

        # Check mass conservation
        mass_diff = (
                            (abs(total_density).sum() * (self.simulation.dV ** 3))
                            / wave_functions[0].soliton_mass
                    ) - self.simulation.num_of_w_vects_in_sim


        if mass_diff > 1e-2:
            print(f"mass diff {mass_diff} is greater than 1e-2, might want to increase the resolution")
        else:
            print(f"mass diff is {mass_diff:.6f} Msun")

        del total_density

        # Main evolution loop
        for step in range(self.num_steps):
            total_density = self._compute_total_density(wave_functions)
            save_step = False
            current_time = step * self.h

            # Compute energies (after first step)
            if step > 0:
                self._compute_kinetic_energy(wave_functions)
                self._compute_potential_energy(wave_functions, total_density, current_time)

            # Track max location
            ix, iy, iz = cp.asnumpy(cp.argwhere(total_density == total_density.max())[0])
            self.scribe.record_max_location(int(ix), int(iy), int(iz), float(current_time))

            if step % save_every == 0 and step > 0:
                save_step = True

            # Perform evolution step
            wave_functions = self._perform_evolution_step(wave_functions, total_density, step, save_step)

            # Save snapshots and profiles
            if step % save_every == 0 and step > 0:
                self._compute_and_save_radial_profile(total_density, current_time, ix, iy, iz)
                self.scribe.save_snapshots(wave_functions, step, self.h)
                print(f"Still working... Step {step} out of {self.num_steps}")

            # Memory cleanup
            cp.get_default_memory_pool().free_all_blocks()

        # Final state
        total_density = self._compute_total_density(wave_functions)
        ix, iy, iz = cp.asnumpy(cp.argwhere(total_density == total_density.max())[0])
        self.scribe.record_max_location(int(ix), int(iy), int(iz), float(current_time))
        self._compute_and_save_radial_profile(total_density, current_time, ix, iy, iz)
        cp.get_default_memory_pool().free_all_blocks()

        # Save final state and finalize
        self.scribe.save_final_state(wave_functions, self.num_steps, save_every, self.h, self.total_time)
        self._finalize_evolution()

        return wave_functions

    def _perform_evolution_step(self, wave_functions, total_density, step, save_step):
        """Perform a single evolution step based on the order."""
        evolution_methods = {
            2: self._evolve_order_2,
            4: self._evolve_order_4,
            6: self._evolve_order_6
        }

        if self.order not in evolution_methods:
            raise ValueError(f"Order {self.order} is not supported. Use 2, 4, or 6.")

        is_first = (step == 0)
        if is_first:
            print("is first")
        is_last = (step == self.num_steps - 1)

        wave_functions = evolution_methods[self.order](wave_functions, total_density, is_first, is_last, save_step)

        # Track maximum values if enabled
        if self.save_max_vals:
            max_val = float(abs(total_density).max())
            self.scribe.track_max_value(step, max_val)

        return wave_functions

    def _evolve_order_2(self, wave_functions, total_density, is_first, is_last, save_step):
        """Second-order split-step evolution."""
        # Kick step
        self._kick_all_wave_functions(wave_functions, total_density, is_first, is_last)

        # Drift step
        self._drift_all_wave_functions(wave_functions)

        # Final kick for last step
        if is_last:
            total_density = self._compute_total_density(wave_functions)
            self._kick_all_wave_functions(wave_functions, total_density, is_first, is_last)

        return wave_functions

    def _evolve_order_4(self, wave_functions, total_density, is_first, is_last, save_step):
        """Fourth-order split-step evolution."""
        steps = [
            ('kick', 'v2'), ('drift', 't2'), ('kick', 'v1'), ('drift', 't1'),
            ('kick', 'v0'), ('drift', 't1'), ('kick', 'v1'), ('drift', 't2'),
            ('kick', 'v2')
        ]

        for i, (operation, coeff_key) in enumerate(steps):
            if operation == 'kick':
                total_density = self._compute_total_density(wave_functions)
                first_op = is_first and i == 0
                if first_op:
                    print(first_op)
                last_op = is_last and i == len(steps) - 1
                self._kick_all_wave_functions(wave_functions, total_density, first_op, last_op, coeff_key)
            else:  # drift
                self._drift_all_wave_functions(wave_functions, time_factor_key=coeff_key)

        return wave_functions

    def _evolve_order_6(self, wave_functions, total_density, is_first, is_last, save_step):
        """Sixth-order split-step evolution."""
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
                self._drift_all_wave_functions(wave_functions, time_factor_key=coeff_key)

        return wave_functions
    '''
    def _kick_all_wave_functions(self, wave_functions, total_density, is_first_step, is_last_step,
                                 time_factor_key='full'):
        """Apply kick step to all wave functions with shared density."""
        time_factor = self.coefficients[time_factor_key]
        static_propagator = self.static_propagators[time_factor_key]

        for wf in wave_functions:
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
            wf.psi *= full_potential_propagator '''


    def _kick_all_wave_functions(self, wave_functions, total_density, is_first_step, is_last_step,
                                 time_factor_key='full'):
        """Apply kick step to all wave functions with shared density."""
        time_factor = self.coefficients[time_factor_key]
        static_propagator = self.static_propagators[time_factor_key]

        for wf in wave_functions:
            # Compute total dynamic propagator (gravity + self-int + sponge)
            dynamic_propagator = self.propagator.compute_total_propagator(
                wf.psi, total_density,
                first_step=is_first_step,
                last_step=is_last_step,
                time_factor=time_factor
            )

            # Combine with static propagator
            full_propagator = static_propagator * dynamic_propagator
            wf.psi *= full_propagator

    def _drift_all_wave_functions(self, wave_functions, time_factor_key='full'):
        """Apply drift step to all wave functions."""
        kinetic_propagator = self.kinetic_propagators[time_factor_key]

        for wf in wave_functions:
            psi_k = cp.fft.fftn(wf.psi)
            psi_k *= kinetic_propagator
            wf.psi = cp.fft.ifftn(psi_k)

    def _compute_total_density(self, wave_functions):
        """Calculate the total density ρ = Σ|ψⁱ|² from all wave functions."""
        total_density = cp.zeros_like(wave_functions[0].psi, dtype=cp.float64)
        for wf in wave_functions:
            density_i = wf.calculate_density()
            total_density += density_i
        return total_density

    def _compute_potential_energy(self, wave_functions, total_density, current_time):
        """Compute the potential energy and log it."""
        dx = np.prod(self.simulation.dx)

        # Compute gravity potential (it will be stored in propagator)
        phi = self.propagator.compute_gravity_potential(total_density)

        rho = total_density
        W = cp.real(0.5 * cp.sum(rho * phi) * dx)
        K = cp.real(self.last_kinetic_energy)

        # Log energy via scribe
        self.scribe.log_energy(current_time, K, W)

    def _compute_kinetic_energy(self, wave_functions):
        """Compute the kinetic energy of the wavefunction."""
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

    def _compute_and_save_radial_profile(self, total_density, current_time, ix, iy, iz, Nbins=250):
        """Compute and save the spherically averaged radial density profile."""
        dx = self.simulation.dx
        BoxSize = [b[1] - b[0] for b in self.simulation.boundaries]
        rho = total_density

        grid_x = cp.asarray(self.simulation.grids[0])
        grid_y = cp.asarray(self.simulation.grids[1])
        grid_z = cp.asarray(self.simulation.grids[2])

        center_x = grid_x[ix, iy, iz]
        center_y = grid_y[ix, iy, iz]
        center_z = grid_z[ix, iy, iz]

        # Compute shifted periodic coordinates
        Delta_x = grid_x - center_x
        Delta_y = grid_y - center_y
        Delta_z = grid_z - center_z

        Delta_x -= cp.copysign(BoxSize[0], Delta_x) * (cp.abs(Delta_x) > BoxSize[0] / 2)
        Delta_y -= cp.copysign(BoxSize[1], Delta_y) * (cp.abs(Delta_y) > BoxSize[1] / 2)
        Delta_z -= cp.copysign(BoxSize[2], Delta_z) * (cp.abs(Delta_z) > BoxSize[2] / 2)

        r = cp.sqrt(Delta_x ** 2 + Delta_y ** 2 + Delta_z ** 2)

        # Transfer to CPU
        r_cpu = cp.asnumpy(r).ravel()
        rho_cpu = cp.asnumpy(rho).ravel()

        # Create radial bins
        max_radius = 0.95 * 0.5 * min(BoxSize)
        bins = np.concatenate(([0.0], np.geomspace(0.003, max_radius, Nbins)))

        # Compute mean density per bin
        mass_in_bin = []
        for i in range(Nbins):
            mask = (r_cpu > bins[i]) & (r_cpu <= bins[i + 1])
            if np.any(mask):
                r_avg = 0.5 * (bins[i] + bins[i + 1])
                rho_mean = rho_cpu[mask].mean()
                mass_in_bin.append((r_avg, rho_mean))

        mass_in_bin = np.array(mass_in_bin)
        bin_centers = mass_in_bin[:, 0]
        rho_avg = mass_in_bin[:, 1]

        # Save via scribe
        self.scribe.save_radial_density_profile(bin_centers, rho_avg, current_time)

    def _calculate_coefficients_and_propagators(self):
        """Calculate coefficients and pre-compute static propagators."""
        if self.order == 2:
            self.coefficients = {'full': 1.0}
            potential_coeffs = [('full', 1.0)]
            kinetic_coeffs = [('full', 1.0)]

        elif self.order == 4:
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

    def _finalize_evolution(self):
        """Cleanup and finalize evolution process."""
        # Save metadata
        self.scribe.save_metadata(
            self.num_steps,
            self.h,
            self.total_time,
            self.order,
            self.num_wave_functions,
        )

        # Save energy log
        self.scribe.save_energy_log()

        # Handle max values if enabled
        if self.save_max_vals:
            self.scribe.save_max_values(
                resolution=int(self.simulation.N),
                spin=self.simulation.spin
            )
            plot_y_or_n = input("Should I plot these values? (y/n/del): ")
            if plot_y_or_n == "y":
                plot_max_values_on_N(self)
            elif plot_y_or_n == "del":
                max_vals_filename = "resources/data/max_values.csv"
                if os.path.exists(max_vals_filename):
                    os.remove(max_vals_filename)
                    print(f"File '{max_vals_filename}' has been deleted.")
                else:
                    print(f"File '{max_vals_filename}' does not exist.")

        print("Evolution completed successfully")
        print(f"Saved times are {self.scribe.accessible_times}")
        cp.get_default_memory_pool().free_all_blocks()

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
        return self.scribe.get_wave_function_at_time(
            time,
            wf_index=wf_index,
            num_wave_functions=self.num_wave_functions
        )

