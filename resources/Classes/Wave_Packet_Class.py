from resources.Functions.Schrodinger_eq_functions import *
from resources.Errors.Errors import *
import cupy as cp
import os
import pandas as pd

class Packet():
    """
    A class for wave packet initialization. Inherits from Wave_function and handles
    the creation of the initial wavefunction, based on the selected packet type.
    """

    def __init__(self, packet_type="gaussian",dim =1, boundaries=None, potential=None, means=None, st_deviations=None, grids=None, dx=None, mass=1, omega=1,momenta = [0],
                 *args, **kwargs):
        """
        Initialize a Packet instance.

        Parameters:
            packet_type (str): Type of the wave packet ('gaussian', 'LHO', etc.).
            means (list): Mean positions for the wave packet in each dimension.
            st_deviations (list): Standard deviations for the wave packet in each dimension.
            grids (list[cp.ndarray]): Precomputed grids for the domain (one grid per dimension).
            dx (list[float]): List of grid spacings for each dimension.
            mass (float): Mass of the particle in the wave packet.
            omega (float): Frequency of the harmonic oscillator.
        """

        self.dim = dim  # Number of dimensions
        self.momenta = momenta
        self.packet_type = packet_type
        self.means = means if means else [0] * self.dim
        self.st_deviations = st_deviations if st_deviations else [0.1] * self.dim
        self.grids = grids  # Receive grids from parent class
        self.potential = potential  # Potential function
        self.dx = dx  # Receive dx from parent class
        self.mass = mass  # New attribute for particle mass
        self.omega = omega  # New attribute for harmonic oscillator frequency
        self.momentum_propagator = self.compute_momentum_propagator()


        if self.grids is None or len(self.grids) != self.dim:
            raise ValueError("Grids must be provided for each dimension.")

    def compute_momentum_propagator(self):
        """Compute the kinetic propagator based on Fourier space components."""
        # Use single-precision floats to save memory
        momenta = [
            -1j * cp.array((momentum / 1) * grid, dtype=cp.float32)
            for momentum, grid in zip(self.momenta, self.grids)]
        summed_momenta = cp.zeros_like(momenta[0])
        for momentum in momenta:
            summed_momenta += momentum
        print("tu dobrý")

        return cp.exp(summed_momenta)

    def create_psi_0(self):
        """
        Creates the normalized initial wavefunction (ψ₀), based on the specified packet type.

        Raises:
            InitWaveParamsError: If the initialization parameters do not match the number of dimensions.
        """
        # Validation of means and standard deviations
        if len(self.means) != self.dim:
            raise InitWaveParamsError(
                message=f"Expected {self.dim} means, but got {len(self.means)}",
                tag="means"
            )
        if len(self.st_deviations) != self.dim:
            raise InitWaveParamsError(
                message=f"Expected {self.dim} standard deviations, but got {len(self.st_deviations)}",
                tag="st_deviations"
            )

        # Check if packet_type is a file path
        is_file = os.path.isfile(self.packet_type) or self.packet_type.endswith(('.txt', '.dat', '.csv', '.npy'))

        if is_file:
            # If `packet_type` is a path, treat it as a file
            file_path = self.packet_type
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Could not find the specified wave function file '{file_path}'. Raised from Wave_Packet_Class.py")
            wave_packet = self.create_ground_state(file_path)
            wave_packet *= self.momentum_propagator
            return wave_packet

        # Start creating the wavefunction
        if self.packet_type == "gaussian":
            wave_packet = self._create_gaussian_packet()
            wave_packet *= self.momentum_propagator
            return wave_packet
        elif self.packet_type == "LHO":
            return self._create_LHO_packet()
        else:
            raise IncorrectPacketTypeError(
                message=f"The packet type '{self.packet_type}' is not recognized.",
                packet_type=self.packet_type
            )

    def _create_gaussian_packet(self):
        """
            Creates a Gaussian wave packet based on the provided means and standard deviations.

            Returns:
                cp.ndarray: The normalized Gaussian wavefunction.
            """
        # Convert means and st_deviations to cupy arrays
        means_arr = cp.array(self.means)
        st_deviations_arr = cp.array(self.st_deviations)

        # Stack grids for proper broadcasting
        # grids_stacked = cp.stack(self.grids, axis=0)
        psi_0 = cp.ones_like(self.grids[0])
        for i in range(len(self.grids)):
            psi_0 *= gaussian_packet(self.grids[i], means_arr[i], st_deviations_arr[i])

        '''psi_0 = cp.exp(-cp.sum(((grids_stacked - means_arr[:, np.newaxis, np.newaxis, np.newaxis]) ** 2) /
                               (2 * st_deviations_arr[:, np.newaxis, np.newaxis, np.newaxis] ** 2), axis=0),
                       dtype=cp.complex128)        '''

        # Normalize the wavefunction over all dimensions
        dx_total = cp.prod(cp.array(self.dx))  # Total grid spacing in all dimensions
        # psi_0 = normalize_wavefunction(psi_0, dx_total)
        return psi_0 + 0j

    def create_ground_state(self, file_path):
        """
        Creates a ground state wave packet from data file.

        Parameters:
            file_path (str): Path to the file containing ground state data.

        Returns:
            cp.ndarray: The normalized ground state wavefunction.
        """

        data = self.read_ground_state_data(file_path)

        # Extract radial distance and wave function values
        r_values = data[:, 0]  # First column is r (sorted in file)
        phi_values = data[:, 1]  # Second column is phi

        # Compute r_distance efficiently for all grid points
        r_distance = cp.zeros_like(self.grids[0])
        for dim in range(self.dim):
            r_distance += (self.grids[dim] - self.means[dim]) ** 2
        r_distance = cp.sqrt(r_distance)


        closest_r_indices = cp.searchsorted(r_values, r_distance)

        # Ensure indices stay within bounds
        closest_r_indices = cp.clip(closest_r_indices, 0, len(r_values) - 1)

        # Assign phi_values based on the found indices
        psi_0 = phi_values[closest_r_indices]

        return psi_0.astype(cp.complex64)

    def read_ground_state_data(self, file_path):
        """
        Reads the ground state wavefunction data from a file.

        Args:
            file_path (str): Path to the file containing wave function data.
            The expected format is:
            #r phi U
            0.0 1.0 -4.756424826198888
            ...

        Returns:
            numpy.ndarray: The loaded wave function data as a numpy array.
        """
        try:
            # Read all lines from the file
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Skip comment lines (starting with #)
            data_lines = [line for line in lines if not line.strip().startswith('#')]

            # Process each line to extract the values
            data = []
            for line in data_lines:
                values = line.strip().split()
                if len(values) >= 3:  # Ensure at least r, phi, and U values
                    r = float(values[0])
                    phi = float(values[1])
                    # U is at index 2, but we don't need it
                    data.append([r, phi])

            # Converting to cupy array
            return cp.array(data)

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")
        except ValueError:
            raise ValueError(f"Invalid data format in file '{file_path}'.")
        except Exception as e:
            raise Exception(f"Error reading file '{file_path}': {str(e)}")

    def _create_LHO_packet(self):
        """
        Creates a wave packet for the linear harmonic oscillator (LHO).

        Returns:
            cp.ndarray: The normalized LHO wavefunction.
        """
        # Compute linear harmonic oscillator wave packet over the full meshgrid
        psi_0 = lin_harmonic_oscillator(self)

        # Normalize the wavefunction over all dimensions
        dx_total = cp.prod(cp.array(self.dx))  # Total grid spacing in all dimensions
        #psi_0 = normalize_wavefunction(psi_0, dx_total)
        return psi_0 + 0j
