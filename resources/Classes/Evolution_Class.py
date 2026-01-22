import cupy as cp
import numpy as np
from resources.Functions.system_fucntions import plot_max_values_on_N
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Scribe_Class import Scribe
import os
from tqdm import tqdm

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

        self.track_particle = False

        #sink particle params
        self.enable_sink_formation = False
        self.sink_density_threshold = None
        self.sink_consecutive_steps = 5
        self.sink_tracker = None
        self.sink_system = None
        self.sink_check_interval = 1

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

        if (hasattr(self.simulation, 'baryonic_matter') and
                self.simulation.baryonic_matter and
                len(self.simulation.baryonic_matter) == 1 and
                hasattr(self.simulation.baryonic_matter[0], "N") and
                self.simulation.baryonic_matter[0].N == 1):
            self.track_particle = True
            print("Single particle detected: Trajectory tracking enabled.")

        # Convert to CuPy arrays
        for wf in wave_functions:
            wf.psi = cp.asarray(wf.psi)



        # Setup directories and save initial state
        self.scribe.setup_directories(self.num_wave_functions)
        self.scribe.save_initial_states(wave_functions)



        # Initial diagnostics
        total_density = self._compute_total_density(wave_functions)

        mass_total = cp.sum(total_density) * self.simulation.dV
        print("Mass:", mass_total)
        print("Deviation [%]:", 100 * (mass_total / 1e8 - 1))


        current_time = 0
        if self.simulation.dim == 3:
            # Check if density has any non-zero values
            max_indices = cp.argwhere(total_density == total_density.max())
            if max_indices.size > 0:
                ix, iy, iz = cp.asnumpy(max_indices[0])
                self.scribe.record_max_location(int(ix), int(iy), int(iz), float(current_time))
                self._compute_and_save_radial_profile(total_density, current_time, ix, iy, iz)
            else:
                print("Warning: No maximum density location found (density may be zero everywhere)")
                ix, iy, iz = self.simulation.N // 2, self.simulation.N // 2, self.simulation.N // 2

        self.compute_total_energy(wave_functions, total_density, current_time)

        # Check mass conservation
        if wave_functions:
            try:
                mass_diff = (
                            (abs(total_density).sum() * (self.simulation.dV ** 3))
                            / wave_functions[0].soliton_mass
                    ) - self.simulation.num_of_w_vects_in_sim
            except AttributeError:
                mass_diff = 0
        else:
            mass_diff = 0
            print("no wave functions detected")

        if mass_diff > 1e-2:
            print(f"mass diff {mass_diff} is greater than 1e-2, might want to increase the resolution")
        else:
            print(f"mass diff is {mass_diff:.6f} Msun")



        # Main evolution loop
        for step in tqdm(range(self.num_steps), desc="Simulation Progress", unit="step"):
            if step == 0:
                print("starting evolution")
            else:
                total_density = self._compute_total_density(wave_functions)
            save_step = False
            current_time = step * self.h

            if self.track_particle:
                # Get position from GPU to CPU
                baryons = self.simulation.baryonic_matter[0]
                pos_gpu = baryons.positions[0]
                pos_cpu = cp.asnumpy(pos_gpu)

                # Log to scribe
                self.scribe.log_single_particle(current_time, pos_cpu)

            # Track max location
            if self.simulation.dim == 3:
                max_indices = cp.argwhere(total_density == total_density.max())
                if max_indices.size > 0:
                    ix, iy, iz = cp.asnumpy(max_indices[0])
                    self.scribe.record_max_location(int(ix), int(iy), int(iz), float(current_time))
                else:
                    # Use center of grid as fallback
                    ix, iy, iz = self.simulation.N // 2, self.simulation.N // 2, self.simulation.N // 2

            if step % save_every == 0 and step > 0:
                save_step = True

            #checking if BH appeared
            new_sink = self._check_and_form_sinks(step)
            self._perform_sink_accretion()
            if new_sink:
                print("New sink formed at step", step)
                total_density = self._compute_total_density(wave_functions)

            # Perform evolution step
            wave_functions = self._perform_evolution_step(wave_functions, total_density, step, save_step)

            total_density = self._compute_total_density(wave_functions)
            current_time = (step + 1) * self.h
            self.compute_total_energy(wave_functions, total_density, current_time)

            # Save snapshots and profiles
            if step % save_every == 0:


                if self.simulation.dim == 3:
                    self._compute_and_save_radial_profile(total_density, current_time, ix, iy, iz)
                                
                baryon_density_to_save = None
                gas_density_to_save = None
                total_density_to_save = total_density 

                if hasattr(self.simulation, 'baryonic_matter') and self.simulation.baryonic_matter:
                    shape = (self.simulation.N,) * self.simulation.dim
                    rho_baryons_all = cp.zeros(shape, dtype=cp.float64)
                    rho_sinks_only = cp.zeros(shape, dtype=cp.float64)
                    rho_gas_only = cp.zeros(shape, dtype=cp.float64)

                    for sys in self.simulation.baryonic_matter:
                        dens = sys.deposit_to_grid()
                        rho_baryons_all += dens
                        
                        if self._is_sink_system(sys):
                            rho_sinks_only += dens

                        if self._is_gas_system(sys):
                            rho_gas_only += dens
                            U = sys.internal_energy()
                            Er = sys.E_radiated
                            print(step, self.h*step, U, Er, U + Er)


                    baryon_density_to_save = rho_baryons_all
                    total_density_to_save = total_density + rho_sinks_only + rho_gas_only

                self.scribe.save_snapshots(
                    wave_functions, 
                    step, 
                    self.h, 
                    total_density=total_density_to_save,
                    baryon_density=baryon_density_to_save,
                    gas_density=gas_density_to_save
                )
                
                self.scribe.flush_trajectory_buffer()


            # Memory cleanup
            cp.get_default_memory_pool().free_all_blocks()

        # Final state
        total_density = self._compute_total_density(wave_functions)
        if self.simulation.dim == 3:
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

        if wave_functions:
            self._kick_all_wave_functions(wave_functions, total_density, is_first, is_last)

        # is there baryonic matter in the sim?
        if self.simulation.baryonic_matter:
            # Potential used for baryon drift is computed in Evolution (Poisson + static potential if present)
            if wave_functions:
                total_density_baryons = self._compute_total_density(wave_functions)
            else:
                shape = (self.simulation.N,) * self.simulation.dim
                total_density_baryons = cp.zeros(shape, dtype=cp.float64)
                for baryon_sys in self.simulation.baryonic_matter:
                    total_density_baryons += baryon_sys.deposit_to_grid()

            phi_sink = self._compute_sink_potential_analytic_kspace()
            phi_environment = self.propagator.compute_gravity_potential(total_density)
            phi_total = phi_environment + phi_sink

            forces_environment = self.propagator.compute_force_grids_from_potential(phi_environment)
            forces_total = self.propagator.compute_force_grids_from_potential(phi_total)

            for baryons in self.simulation.baryonic_matter:
                if isinstance(baryons, SinkNBody):
                    baryons.drift(
                        dt=self.h * 1.0,
                        potential_grid=forces_environment,
                        first_step=is_first,
                        last_step=is_last
                    )
                else:
                    baryons.drift(
                        dt=self.h * 1.0,
                        potential_grid=forces_total,
                        first_step=is_first,
                        last_step=is_last
                    )


        # Drift step (for wave functions if present)
        if wave_functions:
            self._drift_all_wave_functions(wave_functions)

        # Final kick for last step
        if is_last and wave_functions:
            total_density = self._compute_total_density(wave_functions)
            self._kick_all_wave_functions(wave_functions, total_density, False, True)
            print("is last")

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
                last_op = is_last and i == len(steps) - 1

                if wave_functions:
                    self._kick_all_wave_functions(wave_functions, total_density, first_op, last_op, coeff_key)

                # Evolve baryons at appropriate kick steps
                if self.simulation.baryonic_matter:
                    phi_sink = self._compute_sink_potential_analytic_kspace()
                    time_factor = self.coefficients[coeff_key]

                    phi_environment = self.propagator.compute_gravity_potential(total_density)
                    phi_total = phi_environment + phi_sink

                    forces_environment = self.propagator.compute_force_grids_from_potential(phi_environment)
                    forces_total = self.propagator.compute_force_grids_from_potential(phi_total)

                    for baryons in self.simulation.baryonic_matter:
                        if isinstance(baryons, SinkNBody):
                            baryons.drift(
                                dt = self.h * time_factor,
                                potential_grid=forces_environment,
                                first_step = first_op,
                                last_step = last_op
                            )
                        else:
                            baryons.drift(
                                dt = self.h * time_factor,
                                potential_grid=forces_total,
                                first_step = first_op,
                                last_step = last_op
                            )
            else:  # drift
                if wave_functions:
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
                first_op = is_first and i == 0
                last_op = is_last and i == len(steps) - 1

                if wave_functions:
                    self._kick_all_wave_functions(wave_functions, total_density, first_op, last_op, coeff_key)

                # Evolve baryons at appropriate kick steps
                if self.simulation.baryonic_matter:
                    phi_sink = self._compute_sink_potential_analytic_kspace()
                    time_factor = self.coefficients[coeff_key]

                    phi_environment = self.propagator.compute_gravity_potential(total_density)
                    phi_total = phi_environment + phi_sink

                    forces_environment = self.propagator.compute_force_grids_from_potential(phi_environment)
                    forces_total = self.propagator.compute_force_grids_from_potential(phi_total)

                    for baryons in self.simulation.baryonic_matter:
                        if isinstance(baryons, SinkNBody):
                            baryons.drift(
                                dt=self.h * time_factor,
                                potential_grid=forces_environment,
                                first_step=first_op,
                                last_step=last_op
                            )
                        else:
                            baryons.drift(
                                dt=self.h * time_factor,
                                potential_grid=forces_total,
                                first_step=first_op,
                                last_step=last_op
                            )
            else:  # drift
                if wave_functions:
                    self._drift_all_wave_functions(wave_functions, time_factor_key=coeff_key)

        return wave_functions



    def _kick_all_wave_functions(self, wave_functions, total_density, is_first_step, is_last_step,
                                 time_factor_key='full'):
        """Apply kick step to all wave functions with shared density."""
        time_factor = self.coefficients[time_factor_key]

        for wf in wave_functions:
            # Compute total dynamic propagator (gravity + self-int + sponge)
            dynamic_propagator = self.propagator.compute_total_propagator(
                density=total_density, psi=wf.psi,
                first_step=is_first_step,
                last_step=is_last_step,
                time_factor=time_factor
            )


            if time_factor_key in self.static_propagators:
                static_propagator = self.static_propagators[time_factor_key]
                full_propagator = dynamic_propagator * static_propagator
            else:
                full_propagator = dynamic_propagator

            wf.psi *= full_propagator



    def _drift_all_wave_functions(self, wave_functions, time_factor_key='full'):
        """Apply drift step to all wave functions."""
        kinetic_propagator = self.kinetic_propagators[time_factor_key]

        for wf in wave_functions:
            wf.drift(kinetic_propagator)
            #wf._dealias_initial_psi(frac=2/3)

    def _compute_total_density(self, wave_functions):
        """Calculate the total density ρ = Σ|ψᵢ|² from all wave functions."""
        shape = (self.simulation.N,) * self.simulation.dim
        total_density = cp.zeros(shape, dtype=cp.float64)

        for wf in wave_functions:
            density_i = wf.calculate_density()
            total_density += density_i

        if hasattr(self.simulation, 'baryonic_matter') and self.simulation.baryonic_matter:

            for baryon_sys in self.simulation.baryonic_matter:
                if self._is_sink_system(baryon_sys):
                 continue  # sinks handled analytically
                total_density += baryon_sys.deposit_to_grid()


        return total_density

    def compute_total_energy(self, wave_functions, total_density, current_time):
        """Compute all energy components (waves + baryons) and log them."""
        K_flow, U_quantum, K_baryons = self._compute_kinetic_energy(wave_functions)
        K_total = K_flow + U_quantum + K_baryons

        W_self, W_static, W_total = self._compute_potential_energy(
            wave_functions, total_density, current_time
        )

        U_iso_total = 0.0
        E_diss_total = 0.0
        #E_rad_total = 0.0   might need later

        if hasattr(self.simulation, 'baryonic_matter') and self.simulation.baryonic_matter:
            for sys in self.simulation.baryonic_matter:
                if self._is_sink_system(sys):
                    # assume these are cumulative totals
                    E_diss_total += float(getattr(sys, 'E_diss_kin_total', 0.0))
                    E_diss_total += float(getattr(sys, 'E_diss_formation_total', 0.0))

                # **If you still use isothermal U_iso**
                if hasattr(sys, "rho") and hasattr(sys, "cs") and hasattr(sys, "rho_ref"):
                    rho = sys.rho
                    rho_floor = getattr(sys, "rho_floor", 1e-12)
                    rho_clamped = cp.maximum(rho, rho_floor)

                    # **use constant rho_ref, do NOT recompute from current rho**
                    rho_ref = float(sys.rho_ref)
                    U_iso_total += float((sys.cs ** 2) *
                                         cp.sum(rho_clamped * cp.log(rho_clamped / rho_ref)) *
                                         self.simulation.dV)

                # **Optional: if you already track radiated energy on gas**
                #if hasattr(sys, "E_radiated"):
                #    E_rad_total += float(sys.E_radiated)

        # Pass energies to scribe for logging
        self.scribe.log_energy_detailed(
            current_time,
            K_total=float(K_total),
            W=float(W_total),
            U_iso=float(U_iso_total),
            K_flow=float(K_flow),
            U_quantum=float(U_quantum),
            K_baryons=float(K_baryons),
            W_self=float(W_self),
            W_static=float(W_static),
            E_diss=float(E_diss_total )  # add + E_rad_total
        )

        return K_total, W_total, K_flow, U_quantum, K_baryons, W_self, W_static

    def _compute_potential_energy(self, wave_functions, total_density, current_time):

        """
        Compute potential energy:
        - W_self = self-gravity from Poisson solver (whatever is in total_density)
        - W_static = coupling of the same density to the external static potential
        - W_total = W_self + W_static

        """

        dx = np.prod(self.simulation.dx)

        rho = total_density


        phi_self = self.propagator.compute_gravity_potential(rho)  # from rho (no sinks)
        phi_sink = self._compute_sink_potential_analytic_kspace()  # analytic sinks

        W_self = 0.5 * cp.sum(rho * phi_self) * self.simulation.dV

        rho_sinks = cp.zeros_like(rho)

        if hasattr(self.simulation, 'baryonic_matter'):
            for sys in self.simulation.baryonic_matter:
                if self._is_sink_system(sys):
                    rho_sinks += sys.deposit_to_grid()

        # 0.5 * integral( rho_sink * phi_sink ) to account for Sink Self-Energy
        W_sink_self = 0.5 * cp.sum(rho_sinks * phi_sink) * dx

        W_self += W_sink_self
        # external static potential energy: ∫ ρ Φ_static dV
        W_static = 0.0

        if self.simulation.static_potential is not None:
            phi_static = self.simulation.static_potential(self.simulation)
            rho_total = rho + rho_sinks
            W_static = cp.sum(rho_total * cp.real(phi_static)) * dx

        W_total = cp.real(W_self + W_static)
        self.W_self = cp.real(W_self)
        self.W_static = cp.real(W_static)

        return self.W_self, self.W_static, W_total

    def _compute_kinetic_energy(self, wave_functions):
        """Compute kinetic energy: flow + quantum (waves) + baryons."""
        dx = np.prod(self.simulation.dx)
        k_space = self.simulation.k_space

        K_flow_total = 0.0
        U_quantum_total = 0.0

        # --- wavefunctions ---
        for wf in wave_functions:
            rho = wf.calculate_density()  # |ψ|²
            sqrt_rho = cp.sqrt(rho + 1e-30)  # Add small value to avoid division by zero

            # Compute gradients in Fourier space
            psi_k = cp.fft.fftn(wf.psi)
            sqrt_rho_k = cp.fft.fftn(sqrt_rho)

            grad_psi_squared = cp.zeros_like(rho, dtype=cp.float64)
            grad_sqrt_rho_squared = cp.zeros_like(rho, dtype=cp.float64)

            # Loop over spatial dimensions to compute gradients
            for dim in range(self.simulation.dim):
                # Gradient of psi: ∇ψ = iFFT(ik * FFT(ψ))
                grad_psi_dim = cp.fft.ifftn(1j * k_space[dim] * psi_k)
                grad_psi_squared += cp.abs(grad_psi_dim) ** 2

                # Gradient of sqrt(rho): ∇√ρ = iFFT(ik * FFT(√ρ))
                grad_sqrt_rho_dim = cp.fft.ifftn(1j * k_space[dim] * sqrt_rho_k)
                grad_sqrt_rho_squared += cp.abs(grad_sqrt_rho_dim) ** 2

            # U_quantum = (ℏ²/2m) ∫ |∇√ρ|² dV
            U_quantum = 0.5 * self.simulation.h_bar_tilde ** 2 * cp.sum(grad_sqrt_rho_squared) * dx

            # K_total_this_wf = (ℏ²/2m) ∫ |∇ψ|² dV
            K_total_this_wf = 0.5 * self.simulation.h_bar_tilde ** 2 * cp.sum(grad_psi_squared) * dx
            K_flow = K_total_this_wf - U_quantum

            K_flow_total += K_flow
            U_quantum_total += U_quantum

        # --- baryons: ½ m v² summed over ALL particles in ALL systems ---
        K_baryons = 0.0
        if getattr(self.simulation, 'baryonic_matter', None):
            for baryons in self.simulation.baryonic_matter:
                if hasattr(baryons, "kinetic_energy"):
                    K_baryons += baryons.kinetic_energy()


        # Store for possible simple logging
        self.K_flow = K_flow_total
        self.U_quantum = U_quantum_total
        self.K_baryons = K_baryons
        self.last_kinetic_energy = K_flow_total + U_quantum_total + K_baryons

        return K_flow_total, U_quantum_total, K_baryons

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
                spin=self.simulation.spin if getattr(self.simulation, "spin", None) else None
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


    def _compose_baryon_potential(self, total_density):
        """
        Compose the total potential felt by baryons: gravity + static potential.

        Parameters:
            total_density: Total density on grid (waves + baryons)

        Returns:
            Total potential field on the grid
        """
        # Gravitational potential from Poisson solver
        V = self.propagator.compute_gravity_potential(total_density)
        # Add static external potential if present
        if self.simulation.static_potential is not None:
            V_stat = self.simulation.static_potential(self.simulation)
            V = V + cp.real(V_stat)

        return V

    def enable_sink_particle_formation(self, density_threshold, consecutive_steps=5,
                                       check_interval=1):
        """
        Enable dynamic sink particle formation during evolution.

        Parameters
        ----------
        density_threshold : float
            Critical baryonic density for sink formation
        consecutive_steps : int
            Number of consecutive steps density must exceed threshold
        check_interval : int
            Check for sink formation every N steps (default: 1)
        """
        from resources.Classes.Nbody_classes.Sink_N_Body import SinkFormationTracker

        self.enable_sink_formation = True
        self.sink_density_threshold = density_threshold
        self.sink_consecutive_steps = consecutive_steps
        self.sink_check_interval = check_interval

        # Initialize tracker
        grid_shape = (self.simulation.N,) * self.simulation.dim
        self.sink_tracker = SinkFormationTracker(
            grid_shape=grid_shape,
            consecutive_steps_required=consecutive_steps
        )

        print(f"Sink formation enabled:")
        print(f"  - Density threshold: {density_threshold}")
        print(f"  - Consecutive steps required: {consecutive_steps}")
        print(f"  - Check interval: {check_interval} steps")

    def _check_and_form_sinks(self, step):
        """
        Check for sink formation conditions and create sinks if needed.
        Called during evolution loop.

        Parameters
        ----------
        step : int
            Current evolution step
        """
        #self.enable_sink_formation = True
        if not self.enable_sink_formation:
            return

        # Only check at specified intervals
        if step % self.sink_check_interval != 0:
            return

        from resources.Classes.Nbody_classes.Sink_N_Body import (
            check_and_create_sinks,
            SinkNBody
        )

        # Check for new sinks
        new_sinks_data = check_and_create_sinks(
            simulation=self.simulation,
            sink_tracker=self.sink_tracker,
            density_threshold=self.sink_density_threshold,
            aggregation_radius=None  # Will use default
        )
        new_sink = False
        if new_sinks_data is not None:
            # Merge with existing or create new
            self.sink_system = SinkNBody.merge_or_create(
                existing_sink_system=self.sink_system,
                new_sinks_data=new_sinks_data,
                simulation=self.simulation
            )

            # Add to simulation's baryonic_matter if not already there
            if self.sink_system not in self.simulation.baryonic_matter:
                self.simulation.baryonic_matter.append(self.sink_system)
            new_sink = True

            print(f"  Total sinks now: {self.sink_system.N}")
            return new_sink

    def _perform_sink_accretion(self):
        """
        Perform sink accretion on all baryonic systems.
        Called during evolution loop.
        """
        if self.sink_system is None or self.sink_system.N == 0:
            return

        # Get non-sink baryonic systems
        regular_baryons = [
            b for b in self.simulation.baryonic_matter
            if not isinstance(b, type(self.sink_system))
        ]

        if len(regular_baryons) > 0:
            n_accreted = self.sink_system.accrete_from_baryons(regular_baryons)
            if n_accreted > 0:
                print(
                        f"  Sinks accreted {n_accreted} particles | "
                        f"ΔE_diss,kin={self.sink_system.E_diss_kin_last:.6e} | "
                        f"E_diss,kin_total={self.sink_system.E_diss_kin_total:.6e}"
                    )

            self.sink_system.drain_reservoir(self.h)

    def _is_sink_system(self, sys):
        return sys.__class__.__name__.lower().startswith("sink")

    def _compute_sink_potential_analytic_kspace(self):
        """
        Periodic analytic sink potential via k-space phase factors.
        Returns phi_sink(x) on the grid (cupy array).
        """
        # collect sink systems
        sink_systems = []
        if getattr(self.simulation, "baryonic_matter", None):
            for sys in self.simulation.baryonic_matter:
                if self._is_sink_system(sys):
                    sink_systems.append(sys)

        if not sink_systems:
            shape = (self.simulation.N,) * self.simulation.dim
            return cp.zeros(shape, dtype=cp.float64)

        # k-space grids from simulation (already used in your FFT Poisson)
        kx, ky, kz = self.propagator.k_space  # cupy arrays from Simulation.create_k_space() 
        k2 = kx*kx + ky*ky + kz*kz
        k = cp.sqrt(k2)

        mask0 = (k2 == 0)

        # cell volume for scaling: your density grid integrates with sum(rho)*dV
        #dV = float(np.prod(self.simulation.dx))

        phi_k_total = cp.zeros_like(k2, dtype=cp.complex128)

        for sink_sys in sink_systems:
            eps_bh = float(getattr(sink_sys, "softening_bh", min(self.simulation.dx)))
            eps_cusp = float(
                getattr(sink_sys, "softening_cusp", getattr(sink_sys, "capture_radius", 3.0 * min(self.simulation.dx))))

            soft_bh = 1
            soft_cusp = 1
            #soft_bh = cp.exp(-0.5 * (k * eps_bh) ** 2)
            #soft_cusp = cp.exp(-0.5 * (k * eps_cusp) ** 2)

            for mbh, mres, pos in zip(cp.asnumpy(sink_sys.mass_bh),
                                      cp.asnumpy(sink_sys.mass_res),
                                      cp.asnumpy(sink_sys.positions)):
                xs, ys, zs = float(pos[0]), float(pos[1]), float(pos[2])
                phase = cp.exp(-1j * (kx * xs + ky * ys + kz * zs))

                # BH component
                if mbh != 0.0:
                    rho_k_bh = (mbh / self.simulation.dV) * phase * soft_bh
                    phi_k_total += (-4.0 * cp.pi * self.propagator.G) * rho_k_bh / k2

                # Cusp/reservoir component (extended)
                if mres != 0.0:
                    rho_k_cusp = (mres / self.simulation.dV) * phase * soft_cusp
                    phi_k_total += (-4.0 * cp.pi * self.propagator.G) * rho_k_cusp / k2

        phi_k_total[mask0] = 0.0 + 0.0j
        phi_sink = cp.fft.ifftn(phi_k_total).real.astype(cp.float64)
        return phi_sink

    def _is_gas_system(self, sys):
        """Check if a baryonic system is a gas system."""
        return sys.__class__.__name__.lower() == "nbodygas"