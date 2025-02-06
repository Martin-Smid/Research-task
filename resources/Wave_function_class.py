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


class Wave_function(Simulation_parameters):
    def __init__(self, packet_type="gaussian", momenta=[0], means=[0], st_deviations=[0.1],potential = None, mass = 1, omega = 1, **kwargs):
        """
        Initialize a wave function with custom momenta for arbitrary dimensions.

        Parameters:
            packet_type (str): Type of wave packet (e.g., "gaussian").
            momenta (list): Initial momenta along each dimension.
            means (list): Means for the wave packet (one value per dimension).
            st_deviations (list): Standard deviations for the wave packet (one value per dimension).
        """
        # Call the parent (Simulation_parameters) initializer
        super().__init__(**kwargs)
        #použít analytical
        # Additional parameters specific to Wave_function
        self.packet_type = packet_type
        self.potential = potential
        self.means = means
        self.mass = mass
        self.omega = omega
        self.st_deviations = st_deviations
        self.momenta = momenta  # Added momentum parameter
        self.psi_0 = self.create_psi_0()
        self.psi_evolved = self.psi_0  # Initialize the evolved wave function


        self.k_space = self.create_k_space()  # Generate the k-space grid

        self.kinetic_propagator, self.potential_propagator = self.compute_propagators()  # Compute propagator (can be updated later)

        self.wave_values = []

        self.evolve()

    def compute_propagators(self):
        """
        Create the time evolution propagators for both kinetic and (if provided) potential energy terms.

        Parameters:
            potential (callable, optional): A function to calculate potential energy at given spatial points.
            mass (float, optional): Particle mass. Defaults to 1.0.

        Returns:
            tuple: Kinetic and potential propagators as separate components.
        """

        potential = self.potential
        # Kinetic propagator in Fourier space
        k_shifted = [k + (momentum / (2 * cp.pi)) for k, momentum in zip(self.k_space, self.momenta)]
        k_squared_sum = sum(k ** 2 for k in k_shifted)
        kinetic_propagator = cp.exp(-1j * (self.h /2) * k_squared_sum / self.mass)

        # Potential propagator in real space
        if potential is not None:
            # Build position-space grid to compute the potential
            grid = cp.meshgrid(*self.grids, indexing='ij')  # Handle dimensions automatically
            potential_values = potential(self)  # Apply the potential function to the grid
            potential_propagator = cp.exp(-1j * self.h * potential_values / 2)

        else:
            # If no potential, assume no effect (identity propagator)
            potential_propagator = cp.ones_like(self.psi_0)

        self.kinetic_propagator = kinetic_propagator
        self.potential_propagator = potential_propagator

        return kinetic_propagator, potential_propagator

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

            # Normalize the wavefunction over all dimensions
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
            # Normalize the wavefunction over all dimensions
            dx_total = cp.prod(cp.array(self.dx))  # Total grid spacing in all dimensions
            psi_0 = normalize_wavefunction(psi_0, dx_total)
            return psi_0 + 0j  # Ensures complex128 type for `psi_0`

    def create_k_space(self):
        """Creates the k-space (wave vector space) for any arbitrary number of dimensions."""
        k_components = [2*cp.pi*cp.fft.fftfreq(self.N, d=self.dx[i]) for i in range(self.dim)]
        k_space = cp.meshgrid(*k_components, indexing='ij')  # Create multidimensional k-space
        return k_space


    def evolve_wavefunction_split_step(self, psi):
        """
        Evolve the wavefunction using the split-step Fourier method, alternating
        between potential and kinetic propagators.

        Parameters:
            psi (cp.ndarray): Initial wave function.
            mass (float, optional): Particle mass. Defaults to 1.0.

        Returns:
            cp.ndarray: Evolved wave function after one time step.
        """
        mass = self.mass
        psi *= self.potential_propagator  # Potential propagator

        # Apply kinetic evolution in Fourier space
        psi_k = cp.fft.fftn(psi)
        psi_k *= self.kinetic_propagator
        psi = cp.fft.ifftn(psi_k)


        psi *= self.potential_propagator


        norm = cp.sum(cp.abs(psi) ** 2) * cp.prod(cp.array(self.dx))  # Total norm
        return psi

    def evolve(self):
        """
        Perform the full time evolution for the wave function using the split-step Fourier method.

        This method updates the wave function (`self.psi_evolved`) as it evolves in time and stores
        the wave function at each time step in `self.wave_values`.

        Returns:
            None
        """
        psi = self.psi_0  # Start with the initial wave function
        self.psi_evolved = psi  # Start evolution with the initial state

        # Store the initial state in the wave_values list
        self.wave_values.append(self.psi_evolved.copy())

        for _ in range(self.num_steps):
            # Evolve the wave function for one step
            psi = self.evolve_wavefunction_split_step(self.psi_evolved)

            # Update the evolved state
            self.psi_evolved = psi

            # Append the wave function at this step to the wave_values list
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
        for _ in range(steps):
            psi = self.evolve_wavefunction_split_step(psi)

            # Normalize the wave function at each step
            dx_total = cp.prod(cp.array(self.dx))
            psi = normalize_wavefunction(psi, dx_total)

        return psi


