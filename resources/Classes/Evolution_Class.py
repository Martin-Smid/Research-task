import cupy as cp
import numpy as np
from resources.Functions.system_fucntions import plot_max_values_on_N
from resources.Classes.Nbody_classes.Sink_N_Body import check_and_create_gas_sinks, SinkNBody
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
        self.sink_source = 'both'  #  'baryons', 'gas', or 'both'

        #values for checking conservations within gas
        self._ref_mass = None
        self._ref_mom = None

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

        if self.sink_system is None and getattr(self.simulation, "baryonic_matter", None):
            for sys in self.simulation.baryonic_matter:
                if self._is_sink_system(sys):
                    self.sink_system = sys
                    break

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
            '''
            #off right now
            new_sink = self._check_and_form_sinks(step)
            self._perform_sink_accretion()
            if new_sink:
                print("New sink formed at step", step)
                total_density = self._compute_total_density(wave_functions)
            '''
            #temporary Mbh mass gain
            if self.sink_system is not None and self.sink_system.N > 0:
                M_start = 1e7  
                M_end = 3e7    
                
               
                growth_end_step = 0.5 * self.num_steps
                
                
                if step <= growth_end_step:
                    fraction = step / growth_end_step
                    new_mass = M_start + (M_end - M_start) * fraction
               
                else:
                    new_mass = M_end
                
                self.sink_system.mass_bh[0] = new_mass
                self.sink_system.masses[0] = new_mass
                
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

                # Save rotation curves for baryonic components
                self._compute_and_save_rotation_curves(
                    current_time=current_time,
                    nbins=40,
                    rmax=15.0,
                    zmax_gas=1.0,
                    zmax_stars=2.0
                )
                #checking for conservation params
                gas_systems = [s for s in self.simulation.baryonic_matter if
                               s.__class__.__name__.lower().endswith("gas")]
                if gas_systems:
                    gas = gas_systems[0]
                    M = gas.total_mass()
                    P = gas.total_momentum()

                    if self._ref_mass is None:
                        self._ref_mass = M
                        self._ref_mom = P

                    dM = (M - self._ref_mass) / (self._ref_mass + 1e-30)
                    dPx = (P[0] - self._ref_mom[0]) / (abs(self._ref_mom[0]) + 1e-30)
                    dPy = (P[1] - self._ref_mom[1]) / (abs(self._ref_mom[1]) + 1e-30)
                    dPz = (P[2] - self._ref_mom[2]) / (abs(self._ref_mom[2]) + 1e-30)

                    if step % 50 == 0:
                        print(f"[Gas invariants] t={current_time:.4f}  dM={dM:.12e}  dP=({dPx:.12e},{dPy:.12e},{dPz:.12e})")

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
        for wf in wave_functions:
            wf.psi = cp.asarray(wf.psi)
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



    def _kick_all_wave_functions(self, wave_functions, total_density, is_first_step, is_last_step, time_factor_key='full'):
        """Apply kick step to all wave functions with shared density."""
        time_factor = self.coefficients[time_factor_key]
        
        phi_sink = self._compute_sink_potential_analytic_kspace()
        

        dt_step = self.h * time_factor
        bh_propagator = cp.exp(-1j * phi_sink * dt_step / self.simulation.h_bar_tilde)

        for wf in wave_functions:
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

            wf.psi *= full_propagator * bh_propagator



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
                #if self._is_sink_system(baryon_sys):
                # continue  # sinks handled analytically
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



                for sys in self.simulation.baryonic_matter:
                    if sys.__class__.__name__.lower().endswith("gas"):
                        if hasattr(sys, "internal_energy"):
                            U_iso_total += float(sys.internal_energy())

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
        Compute potential energy including wave self-gravity, sink analytic potential,
        and the missing cross-terms (wave <-> sink).

        W = 1/2 ∫ rho_w * phi_w dV
          + 1/2 ∫ rho_s * phi_s dV
          +     ∫ rho_w * phi_s dV
          +     ∫ rho_s * phi_w dV
          +     ∫ (rho_w + rho_s) * phi_static dV  (if any)
        """
        dV = float(self.simulation.dV)

        rho_w = total_density
        phi_w = self.propagator.compute_gravity_potential(rho_w)  # from wave density only
        phi_s = self._compute_sink_potential_analytic_kspace()  # analytic sinks (all sinks)

        # sinks deposited on grid (for energy integrals)
        rho_s = cp.zeros_like(rho_w)
        if getattr(self.simulation, "baryonic_matter", None):
            for sys in self.simulation.baryonic_matter:
                if self._is_sink_system(sys):
                    rho_s += sys.deposit_to_grid()

        # self terms
        W_w_self = 0.5 * cp.sum(rho_w * phi_w) * dV
        W_s_self = 0.5 * cp.sum(rho_s * phi_s) * dV

        # cross terms (MISSING in your current version)
        W_cross = (cp.sum(rho_w * phi_s) + cp.sum(rho_s * phi_w)) * dV

        W_static = 0.0
        if self.simulation.static_potential is not None:
            phi_static = self.simulation.static_potential(self.simulation)
            W_static = cp.sum((rho_w + rho_s) * cp.real(phi_static)) * dV

        W_total = cp.real(W_w_self + W_s_self + W_cross + W_static)

        self.W_self = cp.real(W_w_self + W_s_self + W_cross)  # pokud chceš mít "self" jako všechno bez static
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

    def enable_sink_particle_formation(
            self,
            density_threshold=None,
            consecutive_steps=1,
            check_interval=10,
            source="baryons",
            enable_gas_sinks=False,
            gas_density_threshold=None,
            gas_consecutive_steps=1,
            gas_check_interval=10,
            gas_r_acc_cells=3,
            **kwargs
    ):
        """
        Enable dynamic sink particle formation during evolution.

        NOTE:
        - This method expects that Simulation_Class already resolved any "AUTO" logic and passed
          numeric thresholds in density_threshold / gas_density_threshold.
        - If a threshold is None, that branch will simply never create sinks (SinkFormationTracker returns None).
        """
        from resources.Classes.Nbody_classes.Sink_N_Body import SinkFormationTracker

        grid_shape = (self.simulation.N,) * self.simulation.dim

        # Validate source parameter
        if source not in ["baryons", "gas", "both"]:
            raise ValueError(f"source must be 'baryons', 'gas', or 'both', got '{source}'")

        # --- PARTICLE/BARYON SINKS (via check_and_create_sinks) ---
        self.sink_density_threshold = density_threshold
        self.sink_tracker = SinkFormationTracker(grid_shape, consecutive_steps_required=consecutive_steps)
        self.enable_sink_formation = True
        self.sink_consecutive_steps = consecutive_steps
        self.sink_check_interval = check_interval
        self.sink_source = source

        # --- GAS SINKS (via check_and_create_gas_sinks) ---
        self.enable_gas_sink_formation = bool(enable_gas_sinks)
        if self.enable_gas_sink_formation:
            self.gas_sink_tracker = SinkFormationTracker(grid_shape, consecutive_steps_required=gas_consecutive_steps)
            self.gas_sink_density_threshold = gas_density_threshold
            self.gas_sink_consecutive_steps = gas_consecutive_steps
            self.gas_sink_check_interval = gas_check_interval
            self.gas_r_acc_cells = gas_r_acc_cells

        print("Sink formation enabled:")
        print(f"  - Source: {source}")
        print(f"  - Density threshold (baryons/particles): {density_threshold}")
        print(f"  - Consecutive steps required: {consecutive_steps}")
        print(f"  - Check interval: {check_interval} steps")
        if self.enable_gas_sink_formation:
            print(f"  - Gas sinks enabled with threshold: {gas_density_threshold}")
            print(f"  - Gas consecutive steps required: {gas_consecutive_steps}")
            print(f"  - Gas check interval: {gas_check_interval} steps")
            print(f"  - Gas accretion radius: {gas_r_acc_cells} cells")

    def _check_and_form_sinks(self, step):
        """
        Check for sink formation conditions and create sinks if needed.

        Logic:
        - Particle (baryon N-body) sinks use check_and_create_sinks(... source='baryons')
        - Gas sinks use check_and_create_gas_sinks(...), with r_acc = gas_r_acc_cells * dx
        - Both branches merge created sinks into a single SinkNBody system via SinkNBody.merge_or_create(...)
        """
        if not getattr(self, "enable_sink_formation", False):
            return False

        # Global cadence for particle sinks (and the function as a whole)
        if step % getattr(self, "sink_check_interval", 1) != 0:
            return False

        from resources.Classes.Nbody_classes.Sink_N_Body import (
            check_and_create_sinks,
            check_and_create_gas_sinks,
            SinkNBody,
        )

        new_sink_created = False

        # ---------------------------
        # (A) PARTICLE/BARYON SINKS
        # ---------------------------
        do_particles = self.sink_source in ("baryons", "both")
        if do_particles:
            # IMPORTANT: if gas sinks are enabled, we avoid double-counting gas here by forcing source='baryons'
            effective_source = "baryons" if getattr(self, "enable_gas_sink_formation", False) else self.sink_source

            new_sinks_data = check_and_create_sinks(
                simulation=self.simulation,
                sink_tracker=self.sink_tracker,
                density_threshold=self.sink_density_threshold,
                aggregation_radius=None,
                source=effective_source,  # 'baryons' or 'both' (if gas sinks disabled)
            )

            if new_sinks_data is not None:
                self.sink_system = SinkNBody.merge_or_create(
                    existing_sink_system=getattr(self, "sink_system", None),
                    new_sinks_data=new_sinks_data,
                    simulation=self.simulation,
                )
                if self.sink_system not in self.simulation.baryonic_matter:
                    self.simulation.baryonic_matter.append(self.sink_system)

                new_sink_created = True
                print(f"  Total sinks now: {self.sink_system.N}")

                # Merge close sinks after accretion (optional)
                merge_cells = self.simulation._sink_cfg.get("merge_r_cells", 2)
                merge_interval = self.simulation._sink_cfg.get("merge_interval", 10)

                if (step % merge_interval) == 0 and self.sink_system is not None and self.sink_system.N > 1:
                    n_merge, dE = self.sink_system.merge_close_sinks(
                        r_merge_cells=merge_cells,
                        bound_check=True,
                        v_factor=1.0
                    )
                    if n_merge > 0:
                        print(f"  [SINK MERGE] merged {n_merge} pairs, dE_diss={dE:.3e}")

        # ---------------------------
        # (B) GAS SINKS
        # ---------------------------
        gas_systems = [b for b in self.simulation.baryonic_matter
                       if b.__class__.__name__.lower().endswith("gas")]

        if gas_systems and getattr(self, "enable_gas_sink_formation", False) and self.sink_source in ("gas", "both"):
            gas = gas_systems[0]

            if step % getattr(self, "gas_sink_check_interval", 1) == 0:
                dx = float(min(self.simulation.dx))
                r_acc = float(getattr(self, "gas_r_acc_cells", 3)) * dx

                new_gas_sinks = check_and_create_gas_sinks(
                    simulation=self.simulation,
                    sink_tracker=self.gas_sink_tracker,
                    gas_system=gas,
                    density_threshold=self.gas_sink_density_threshold,
                    r_acc=r_acc
                )

                if new_gas_sinks is not None:
                    self.sink_system = SinkNBody.merge_or_create(
                        existing_sink_system=getattr(self, "sink_system", None),
                        new_sinks_data=new_gas_sinks,
                        simulation=self.simulation
                    )
                    if self.sink_system not in self.simulation.baryonic_matter:
                        self.simulation.baryonic_matter.append(self.sink_system)

                    new_sink_created = True
                    print(f"  Total sinks now: {self.sink_system.N}")

                    merge_cells = self.simulation._sink_cfg.get("merge_r_cells", 2)
                    merge_interval = self.simulation._sink_cfg.get("merge_interval", 10)

                    if (step % merge_interval) == 0 and getattr(self, "sink_system",
                                                                None) is not None and self.sink_system.N > 1:
                        n_merge, dE = self.sink_system.merge_close_sinks(
                            r_merge_cells=merge_cells,
                            bound_check=True,
                            v_factor=1.0
                        )
                        if n_merge > 0:
                            print(f"  [SINK MERGE] merged {n_merge} pairs, dE_diss={dE:.3e}")

        return new_sink_created

    def _perform_sink_accretion(self):
        """
        Perform sink accretion on all baryonic systems.
        CHANGE: Added detailed logging for both baryon and gas accretion.
        """
        if self.sink_system is None or self.sink_system.N == 0:
            return

        # Separate baryonic and gas systems
        baryon_systems = []
        gas_systems = []

        for b in self.simulation.baryonic_matter:
            if isinstance(b, type(self.sink_system)):  # Skip sinks
                continue
            if self._is_gas_system(b):
                gas_systems.append(b)
            elif hasattr(b, 'N'):  # Particle-based baryons
                baryon_systems.append(b)
        
        # === ACCRETE FROM BARYONS ===
        if baryon_systems:
            # Track initial state
            initial_baryon_count = sum(b.N for b in baryon_systems)
            initial_baryon_mass = sum(b.N * b.m_particle for b in baryon_systems)

            n_accreted = self.sink_system.accrete_from_baryons(baryon_systems)

            if n_accreted > 0:
                final_baryon_count = sum(b.N for b in baryon_systems)
                final_baryon_mass = sum(b.N * b.m_particle for b in baryon_systems)

                mass_accreted = initial_baryon_mass - final_baryon_mass

                '''
                print(f"  [BARYON ACCRETION]")
                print(f"    Particles accreted: {n_accreted}")
                print(f"    Mass accreted: {mass_accreted:.6e} Msun")
                print(f"    ΔE_diss,kin: {self.sink_system.E_diss_kin_last:.6e}")
                print(f"    E_diss,kin_total: {self.sink_system.E_diss_kin_total:.6e}")
                print(f"    Remaining baryons: {final_baryon_count}")
                '''



        # === ACCRETE FROM GAS ===
        if gas_systems:
            for gas in gas_systems:
                # Track initial state

                initial_gas_mass = float(cp.sum(gas.rho) * self.simulation.dV)
                initial_gas_momentum = gas.total_momentum()

                # Perform gas accretion
                mass_accreted_gas = self.sink_system.accrete_from_gas(
                    gas, dt=self.h, sound_speed=getattr(gas, "cs", None)
                )

                if mass_accreted_gas > 0:
                    final_gas_mass = float(cp.sum(gas.rho) * self.simulation.dV)
                    final_gas_momentum = gas.total_momentum()

                    delta_mass = initial_gas_mass - final_gas_mass
                    delta_momentum = tuple(i - f for i, f in zip(initial_gas_momentum, final_gas_momentum))

                    '''
                    print(f"  [GAS ACCRETION]")
                    print(f"    Mass accreted: {delta_mass:.6e} Msun")
                    print(
                        f"    Δ Momentum: ({delta_momentum[0]:.3e}, {delta_momentum[1]:.3e}, {delta_momentum[2]:.3e})")
                    print(f"    Remaining gas mass: {final_gas_mass:.6e} Msun")
                    '''
                    

        # Drain reservoir (common for all sources)
        self.sink_system.drain_reservoir(self.h)

    def _accrete_gas_to_sinks(self, gas_system):
        """
        Accrete gas from grid to sink particles.

        Returns
        -------
        float : Total mass accreted
        """
        if self.sink_system is None or self.sink_system.N == 0:
            return 0.0

        N = self.simulation.N
        dx = min(self.simulation.dx)
        r_acc = self.sink_system.capture_radius
        r_cells = int(np.ceil(r_acc / dx))

        # Precompute sphere offsets
        offsets = []
        for i in range(-r_cells, r_cells + 1):
            for j in range(-r_cells, r_cells + 1):
                for k in range(-r_cells, r_cells + 1):
                    if (i * i + j * j + k * k) <= r_cells * r_cells:
                        offsets.append((i, j, k))

        grids = [cp.asarray(g) for g in self.simulation.grids]
        cellV = self.simulation.dV
        total_mass_accreted = 0.0

        # For each sink
        for sink_idx in range(self.sink_system.N):
            sink_pos = self.sink_system.positions[sink_idx]

            # Find grid cell containing sink
            ix = int(cp.argmin(cp.abs(grids[0][:, 0, 0] - sink_pos[0])))
            iy = int(cp.argmin(cp.abs(grids[1][0, :, 0] - sink_pos[1])))
            iz = int(cp.argmin(cp.abs(grids[2][0, 0, :] - sink_pos[2])))

            sink_mass_gain = 0.0
            sink_momentum_gain = cp.zeros(3, dtype=cp.float64)

            # Accrete from neighborhood
            for oi, oj, ok in offsets:
                i = (ix + oi) % N
                j = (iy + oj) % N
                k = (iz + ok) % N

                rho_cell = gas_system.rho[i, j, k]
                if rho_cell < gas_system.rho_floor * 2:  # Only accrete above 2x floor
                    continue

                # Accrete fraction of cell mass
                accretion_fraction = 0.1 * self.h / self.sink_system.reservoir_tau  # Gentle accretion
                dm = rho_cell * cellV * accretion_fraction

                vx = gas_system.vx[i, j, k]
                vy = gas_system.vy[i, j, k]
                vz = gas_system.vz[i, j, k]

                sink_mass_gain += dm
                sink_momentum_gain += cp.array([dm * vx, dm * vy, dm * vz])

                # Remove mass from gas (preserve specific energy)
                gas_system.rho[i, j, k] -= dm / cellV
                gas_system.rho[i, j, k] = max(gas_system.rho[i, j, k], gas_system.rho_floor)

                # Update energy proportionally
                if hasattr(gas_system, 'E'):
                    v2 = vx ** 2 + vy ** 2 + vz ** 2
                    e_total_old = gas_system.E[i, j, k]
                    rho_new = gas_system.rho[i, j, k]

                    # Specific internal energy (conserved during accretion)
                    e_int_specific = (e_total_old - 0.5 * rho_cell * v2) / rho_cell
                    gas_system.E[i, j, k] = rho_new * e_int_specific + 0.5 * rho_new * v2

            if sink_mass_gain > 0:
                # Update sink (add to reservoir)
                self.sink_system.mass_res[sink_idx] += sink_mass_gain
                self.sink_system.masses[sink_idx] = self.sink_system.mass_bh[sink_idx] + self.sink_system.mass_res[
                    sink_idx]

                # Momentum conservation
                M_old = self.sink_system.masses[sink_idx] - sink_mass_gain
                V_old = self.sink_system.velocities[sink_idx]
                P_old = M_old * V_old
                P_new = P_old + sink_momentum_gain
                self.sink_system.velocities[sink_idx] = P_new / self.sink_system.masses[sink_idx]

                total_mass_accreted += float(sink_mass_gain)

        return total_mass_accreted

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

            #soft_bh = 1e-6
            #soft_cusp = 1
            soft_bh = cp.exp(-0.5 * (k * eps_bh) ** 2)
            soft_cusp = cp.exp(-0.5 * (k * eps_cusp) ** 2)

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

    def _compute_and_save_rotation_curves(self, current_time, nbins=40, rmax=None, zmax_gas=1.0, zmax_stars=2.0):
        """
        Compute and save rotation curves for all baryonic components that implement
        compute_rotation_curve().
        """
        if not getattr(self.simulation, "baryonic_matter", None):
            return
    
        for i, sys in enumerate(self.simulation.baryonic_matter):
            if not hasattr(sys, "compute_rotation_curve"):
                continue
    
            # choose a readable component name
            if self._is_gas_system(sys):
                component_name = "gas"
                zmax = zmax_gas
            elif self._is_sink_system(sys):
                continue   # sink does not have a meaningful rotation curve here
            else:
                # try to distinguish particle components
                component_name = getattr(sys, "name", f"baryons_{i}")
                zmax = zmax_stars
    
            try:
                R, vphi, sigma, weights = sys.compute_rotation_curve(
                    nbins=nbins,
                    center=(0.0, 0.0, 0.0),
                    zmax=zmax,
                    rmax=rmax
                )
                self.scribe.save_rotation_curve(
                    time=current_time,
                    component_name=component_name,
                    R_centers=R,
                    vphi_mean=vphi,
                    vphi_std=sigma,
                    weights=weights
                )
            except Exception as e:
                print(f"[Evolution] Warning: could not compute rotation curve for {component_name}: {e}")

    def _is_gas_system(self, sys):
        """Check if a baryonic system is a gas system."""
        return sys.__class__.__name__.lower() == "nbodygas"