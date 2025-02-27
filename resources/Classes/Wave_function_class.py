import cupy as cp
from scipy.constants import gravitational_constant

from resources.Functions.Schrodinger_eq_functions import *
from resources.Classes.Simulation_Class import Simulation_parameters
from resources.Classes.Wave_Packet_Class import Packet



#-----------------------------------------------------------------------------------------------------------------------


class Wave_function(Simulation_parameters):  # Streamlined and unified evolution logic
    def __init__(self, packet_type="gaussian", momenta=[0], means=[0], st_deviations=[0.1],
                 potential=None, gravity_potential=None, mass=1, omega=1, **kwargs):
        """
        Initialize a wave function with optional gravitational potential.

        Parameters:
            potential (callable, optional): A user-defined potential function.
            gravity_potential (bool or None, optional): Whether to include dynamic gravitational potential. Defaults to None (no gravity).
            * Other parameters as defined previously.
        """
        super().__init__(**kwargs)  # Initialize Simulation_parameters first
        self.packet_creator = Packet(
            packet_type=packet_type,
            means=means,
            st_deviations=st_deviations,
            grids=self.grids,
            dx=self.dx,
            mass=mass,
            omega=omega,
            **kwargs)

        self.packet_type = packet_type
        self.potential = potential
        self.gravity_potential = gravity_potential
        self.means = means
        self.st_deviations = st_deviations
        self.mass = mass
        self.omega = omega
        self.momenta = momenta
        self.k_space = self.create_k_space()  # Unified propagator logic follows

        # Gravitational potential will be dynamically updated
        # Initialize wave function
        self.psi_0 = self.packet_creator.create_psi_0()
        self.psi_evolved = self.psi_0

        # Precompute propagators
        self.kinetic_propagator = self.compute_kinetic_propagator()
        self.potential_propagator = self.compute_static_potential_propagator()
        self.dynamic_gravity_potential = None  # Updated dynamically if needed

        # Initialize storage for wave evolution
        self.wave_values = []

        # Start evolution
        # Streamlined evolution process
        self.evolve()

    def create_k_space(self):
        """Creates the k-space (wave vector space) for any arbitrary number of dimensions."""
        # Create k-space components with single-precision floats
        k_components = [
            2 * cp.pi * cp.fft.fftfreq(self.N, d=self.dx[i]).astype(cp.float32)
            for i in range(self.dim)
        ]
        # Create multidimensional k-space
        k_space = cp.meshgrid(*k_components, indexing='ij')
        return k_space

    def compute_kinetic_propagator(self):
        """Compute the kinetic propagator based on Fourier space components."""
        # Use single-precision floats to save memory
        k_shifted = [
            cp.array(k + (momentum / (2 * cp.pi)), dtype=cp.float32)
            for k, momentum in zip(self.k_space, self.momenta)
        ]
        # Compute k_squared_sum in-place to save memory
        k_squared_sum = cp.zeros_like(k_shifted[0])
        for k in k_shifted:
            k_squared_sum += k ** 2
        return cp.exp(-1j * (self.h / 2) * k_squared_sum / self.mass, dtype=cp.complex64)

    def update_total_potential(self, psi):
        """Compute total propagator by combining gravitational & static potential."""
        if self.gravity_potential:
            density = self.compute_density()
            gravity_potential = self.solve_poisson(density)


            return cp.exp(-1j * self.h * gravity_potential)
        return cp.ones_like(psi)

    def compute_static_potential_propagator(self):
        """Compute the static potential propagator."""
        if self.potential:

            potential_values = self.potential(self)
            return cp.exp(-1j * self.h * potential_values, dtype=cp.complex64)
        return cp.ones_like(self.psi_0, dtype=cp.complex64)

    def evolve_wavefunction_split_step(self, psi, step_index, total_steps):
        """
        Evolve the wavefunction using the split-step Fourier method, alternating
        between potential and kinetic propagators. Adjusted to handle dynamic
        potential updates if enabled.

        Parameters:
            psi (cp.ndarray): Initial wave function.
            step_index (int): Index of the current step (starting at 0).
            total_steps (int): Total number of steps in the simulation.

        Returns:
            cp.ndarray: Evolved wave function after one time step.
        """
        # If it's the first step, apply half the potential propagator
        gravitational_propagator = self.update_total_potential(psi)  # Dynamic propagator if gravity is enabled
        print("----------------------------------------------------------------")
        print("tohle je grav propag")
        print(gravitational_propagator)
        print("---------------------------------------------------")
        if step_index == 0:
            psi *= cp.sqrt(self.potential_propagator*gravitational_propagator)

        psi *= gravitational_propagator*self.potential_propagator  # Apply dynamic potential propagator

        # Apply kinetic evolution in Fourier space
        psi_k = cp.fft.fftn(psi)
        psi_k *= self.kinetic_propagator
        psi = cp.fft.ifftn(psi_k)



        # If it's the last step, apply only a half potential step
        if step_index == total_steps - 1:
            psi *= cp.sqrt(self.potential_propagator*gravitational_propagator)  # Half potential step at the end



        return psi

    def evolve(self, save_every=20):
        """
        Perform the full time evolution for the wave function using the split-step Fourier method.

        This method updates the wave function (`self.psi_evolved`) as it evolves in time and stores
        the wave function at every `save_every` step in `self.wave_values`.

        Parameters:
            save_every (int): Frequency of saving the wave function values. Defaults to 2 (saves every 2nd step).

        Returns:
            None
        """
        psi = self.psi_0  # Initial wave function
        self.psi_evolved = psi

        self.wave_values.append(self.psi_evolved.copy())  # Save the initial state

        for step in range(self.num_steps):
            # Perform the evolution step
            psi = self.evolve_wavefunction_split_step(
                psi, step_index=step, total_steps=self.num_steps
            )
            self.psi_evolved = psi  # Update evolved wave function

            if step % save_every == 0:
                # Save the wavefunction copy at the current step
                self.wave_values.append(self.psi_evolved.copy())

            # Explicitly delete intermediate variables when step is not saved
            else:
                del psi
                cp.get_default_memory_pool().free_all_blocks()  # Free unused data from GPU memory
                psi = self.psi_evolved

        # Free GPU memory once the evolution is complete
        cp.get_default_memory_pool().free_all_blocks()

    def wave_function_at_time(self, time):
        """
        Get the wave function at a specific time by evolving it from the current state.

        Parameters:
            time (float): The time at which the wave function should be queried.

        Returns:
            cp.ndarray: The wave function at the specified time.
        """
        if time < 0 or time > self.total_time:
            raise ValueError("Specified time is out of bounds: must be in [0, total_time].")

        steps = int(time / self.h)  # Calculate the number of steps to evolve

        # If the wave function for the requested time has already been computed
        if len(self.wave_values) > steps:
            return self.wave_values[steps]  # Retrieve from the precomputed list

        # Otherwise, recompute and return the wave function at the specific time
        psi = self.psi_0  # Start with the initial wave function
        for step in range(steps):
            psi = self.evolve_wavefunction_split_step(psi, step_index=step, total_steps=self.num_steps)

            # Normalize the wave function at each step
            dx_total = cp.prod(cp.array(self.dx))
            psi = normalize_wavefunction(psi, dx_total)

        return psi

    def compute_density(self):
        """Calculate the density rho = m|psi|^2."""
        rho = self.mass * cp.abs(self.psi_evolved).astype(cp.float32) ** 2  # Ensure float32 precision
        #density_sum = cp.sum(rho) * (cp.prod(cp.array(self.dx, dtype=cp.float32)) ** 3)
        # print("density sum is equal to:")
        # print(density_sum)
        return rho

    def solve_poisson(self, density):
        """
        Solve the Poisson equation ∇²V = 4πGρ, ignoring the zero mode.

        Parameters:
            density (cp.ndarray): Mass density ρ = m|ψ|².

        Returns:
            cp.ndarray: Gravitational potential V.
        """
        k_space = self.k_space  # Use existing k-space grids
        k_squared_sum = sum(k ** 2 for k in k_space)


        G = 1  # Gravitational constant
        density_k = cp.fft.fftn(density.astype(cp.complex64))

        # Set zero mode to 0 dynamically based on dimensions
        zero_mode_index = tuple([0] * self.dim)
        density_k -= cp.mean(density_k)

        # Protect against division by zero in k^2 sum
        k_squared_sum = cp.where(k_squared_sum == 0, 1e-12, k_squared_sum)

        # Compute potential in Fourier space with single precision
        potential_k = (4 * cp.pi * G * density_k) / k_squared_sum.astype(cp.complex64)

        # Transform back to real space, cast to real32
        potential = cp.fft.ifftn(potential_k).real.astype(cp.float32)
        return potential










