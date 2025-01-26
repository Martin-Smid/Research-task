from venv import create

import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Schrodinger_eq_functions import *
from Errors import *

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
        """Validades the format of the boundaries and unpacks them into dx and grids.
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
            print(self.dx)
        return self.dx, self.grids



class Wave_function(Simulation_parameters):
    def __init__(self, packet_type="gaussian", means=[0], st_deviations=[0.1], **kwargs):
        # Call the parent (Simulation_parameters) initializer
        super().__init__(**kwargs)

        # Additional parameters specific to Wave_function
        self.packet_type = packet_type
        self.means = means
        self.st_deviations = st_deviations
        self.psi_0 = self.create_psi_0()
        self.k_space = self.create_k_space()  # Generate the k-space
        self.psi_k = self.transform_psi_0_to_k_space()
        self.propagator = self.compute_propagator()  # Default propagator (can be set later)


    def compute_propagator(self, potential=None, mass=1.0):
        """Create the time evolution propagator based on the wave's k-space."""
        if potential is None:  # Example: Free evolution propagator
            if self.dim == 1:
                self.propagator = cp.exp(-1j * (self.h / 2) * (self.k_space ** 2) / mass)
            elif self.dim == 2:
                kx, ky = self.k_space
                self.propagator = cp.exp(-1j * (self.h / 2) * ((kx ** 2 + ky ** 2) / mass))
            else:
                raise ValueError(f"Unsupported dimensionality: {self.dim}")
        else:
            # Example: With potential (needs spatial grid)
            self.propagator = cp.exp(-1j * self.h * (potential + (self.k_space ** 2) / (2 * mass)))
        return self.propagator

    def create_psi_0(self):
        """Validates the format of the inputted means and standard deviations
           and creates the normalized initial wavefunction (`psi_0`).

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

            return psi_0  # Returns a normalized `psi_0`

    def create_k_space(self):
        """Creates the k-space (wave vector space) for 1D or 2D wave functions."""
        if self.dim == 1:
            # For 1D, use cp.fft.fftfreq with the grid spacing
            k_space = cp.fft.fftfreq(self.N, d=self.dx[0])
        elif self.dim == 2:
            # For 2D, use cp.fft.fftfreq for each dimension and create a meshgrid
            kx = cp.fft.fftfreq(self.N, d=self.dx[0])
            ky = cp.fft.fftfreq(self.N, d=self.dx[1])
            k_space = cp.meshgrid(kx, ky, indexing='ij')  # Create 2D k-space
        else: #need to add proper errorhandeling
            raise ValueError(f"Unsupported dimensionality: {self.dim}")
        return k_space

    def transform_psi_0_to_k_space(self):
        """Performs the Fourier transform of the initial wavefunction to obtain psi_k."""
        if self.dim == 1:
            self.psi_k = cp.fft.fft(self.psi_0)  # 1D Fourier Transform
        elif self.dim == 2:
            self.psi_k = cp.fft.fftn(self.psi_0)  # 2D Fourier Transform
        else:#need to add proper errorhandeling
            raise ValueError(f"Unsupported dimensionality: {self.dim}")
        return self.psi_k
