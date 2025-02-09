from venv import create

import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from resources.Schrodinger_eq_functions import *
from resources.Errors import *

class Simulation_parameters():
    """
        A class to manage simulation parameters.

        Parameters:
        dim (int): Number of dimensions in the simulation (default = 1).
        boundaries (list[tuple]): List of tuples specifying the min and max for each dimension (e.g., [(-1, 1)]).
        N (int): Number of spatial points for each dimension (default = 1024).
        total_time (float): Total simulation time (default = 10).
        h (float): Time step size for propagation (default = 0.1).
        """

    def __init__(
            self,
            dim=1,
            boundaries=[(-1, 1)],
            N=1024,
            total_time=10,
            h=0.1,
            dx=(1 + 1) / (1024 - 1)
            ):
        # Setup parameters
        self.dim = dim
        self.boundaries = boundaries
        self.N = N
        self.total_time = total_time
        self.h = h  # Propagation parameter (time step size)
        self.num_steps = int(self.total_time / self.h)
        self.dx = []
        self.grids = []

        # Compute dx and spatial grids for each dimension
        self.dx, self.grids = self.unpack_boundaries()



    def unpack_boundaries(self):
        """Validates the format of the boundaries and unpacks them into dx and grids.
            raises BoundaryFormatError if the boundaries are invalid."""
        if len(self.boundaries) != self.dim:
            raise BoundaryFormatError(
                message=f"Expected boundaries for {self.dim} dimensions but got {len(self.boundaries)}",
                tag="boundaries"
            )
        for i, (a, b) in enumerate(self.boundaries):
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise BoundaryFormatError(
                    message=f"Boundary {i} must be a tuple of two numbers, but got {(a, b)}",
                    tag=f"boundary_{i}"
                )
            if a >= b:
                raise BoundaryFormatError(
                    message=f"Boundary {i} values are invalid: {a} must be less than {b}",
                    tag=f"boundary_{i}"
                )
            # If the boundaries are valid, unpack them
            dx_dim = (b - a) / (self.N - 1)
            self.dx.append(dx_dim)
            self.grids.append(cp.linspace(a, b, self.N))

        return self.dx, self.grids



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
        super().__init__(**kwargs)
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
        self.psi_0 = self.create_psi_0()
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

    def update_total_potential(self, psi):
        """Compute total propagator by combining gravitational & static potential."""
        if self.gravity_potential:
            print("zkusím zapojit gravitaci")
            density = self.compute_density()
            print("hustota")
            print(density)
            print("konec hustoty jde se do poissona")
            gravity_potential = self.solve_poisson(density)
            print("#--------------------------------------------------------------#")
            print(gravity_potential)
            grav_potential = gravity_potential
            print(grav_potential)
            print("#--------------------------------------------------------------#")
            return cp.exp(-1j * self.h * grav_potential)
        return cp.ones_like(psi)

    def compute_kinetic_propagator(self):
        """Compute the kinetic propagator based on Fourier space components."""
        k_shifted = [k + (momentum / (2 * cp.pi)) for k, momentum in zip(self.k_space, self.momenta)]
        k_squared_sum = sum(k ** 2 for k in k_shifted)
        return cp.exp(-1j * (self.h / 2) * k_squared_sum / self.mass)

    def compute_static_potential_propagator(self):
        """Compute the static potential propagator."""
        if self.potential:
            grid = cp.meshgrid(*self.grids, indexing='ij')
            potential_values = self.potential(self)
            return cp.exp(-1j * self.h * potential_values)
        return cp.ones_like(self.psi_0)

    def create_psi_0(self):
        """Validates the format of the inputted means and standard deviations
           and creates the normalized initial wave function (`psi_0`).

           Raises InitWaveParamsError if the initialization parameters do not match the number of dimensions.
        """
        # Validation of means and standard deviations
        if len(self.means) != self.dim:
            raise InitWaveParamsError(
                message=f"Expected {self.dim} means in the format [0, 1, 0.5, ...] but got {len(self.means)}",
                tag="means"
            )
        if len(self.st_deviations) != self.dim:
            raise InitWaveParamsError(
                message=f"Expected {self.dim} standard deviations in the format [0.1, 1, 0.3, ...] but got {len(self.st_deviations)}",
                tag="st_deviations"
            )

        # Start creating the wavefunction
        if self.packet_type == "gaussian":
            # Compute ψ₀ for the first dimension

            psi_0 = gaussian_packet(self.grids[0], self.means[0], self.st_deviations[0])

            # If more than one dimension, iteratively compute the outer product
            for i in range(1, self.dim):
                psi_dim = gaussian_packet(self.grids[i], self.means[i], self.st_deviations[i])
                psi_0 = cp.outer(psi_0, psi_dim)

            # Reshape into a proper multidimensional grid if needed
            psi_0 = psi_0.reshape([self.N] * self.dim)

            # Normalize the wave function over all dimensions
            dx_total = cp.prod(cp.array(self.dx))  # Total grid spacing in all dimensions
            psi_0 = normalize_wavefunction(psi_0, dx_total)

            return psi_0 + 0j  # Ensures complex128 type for `psi_0`
        if self.packet_type == "LHO":
            # Compute ψ₀ for the first dimension
            psi_0 = lin_harmonic_oscillator(self)
            # If more than one dimension, iteratively compute the outer product
            for i in range(1, self.dim):
                psi_dim = lin_harmonic_oscillator(self)
                psi_0 = cp.outer(psi_0, psi_dim)
            # Reshape into a proper multidimensional grid if needed
            psi_0 = psi_0.reshape([self.N] * self.dim)
            # Normalize the wave function over all dimensions
            dx_total = cp.prod(cp.array(self.dx))  # Total grid spacing in all dimensions
            psi_0 = normalize_wavefunction(psi_0, dx_total)

            return psi_0 + 0j  # Ensures complex128 type for `psi_0`

    def create_k_space(self):
        """Creates the k-space (wave vector space) for any arbitrary number of dimensions."""
        k_components = [2*cp.pi*cp.fft.fftfreq(self.N, d=self.dx[i]) for i in range(self.dim)]
        k_space = cp.meshgrid(*k_components, indexing='ij')  # Create multidimensional k-space
        return k_space

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
        else:

            psi *=self.potential_propagator*gravitational_propagator


        return psi

    def evolve(self, save_every=5):
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

        self.wave_values.append(self.psi_evolved.copy())

        for step in range(self.num_steps):
            psi = self.evolve_wavefunction_split_step(
                psi, step_index=step, total_steps=self.num_steps
            )  # Corrected the function call here
            self.psi_evolved = psi  # Update evolved wave function

            if step % save_every == 0:
                self.wave_values.append(self.psi_evolved.copy())

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
        return self.mass * cp.abs(self.psi_evolved) ** 2

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

        # Solve Poisson in Fourier space
        G = 1  # Gravitational constant (in arbitrary units)
        density_k = cp.fft.fftn(density)

        # Set zero mode to 0 dynamically based on dimensions
        zero_mode_index = tuple([0] * self.dim)  # e.g., (0,) for 1D, (0, 0) for 2D, etc.
        density_k[zero_mode_index] = 0  # Dynamically target zero mode for N dimensions

        # Protect against division by zero in k^2 sum
        k_squared_sum = cp.where(k_squared_sum == 0, 1e-10, k_squared_sum)

        # Compute potential in Fourier space
        potential_k = (4 * cp.pi * G * density_k) / k_squared_sum

        # Transform back to real space
        potential = cp.fft.ifftn(potential_k).real
        return potential










