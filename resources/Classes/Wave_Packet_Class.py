from resources.Functions.Schrodinger_eq_functions import *
from resources.Errors.Errors import *
import numpy as np
import os
from scipy.interpolate import interp1d
import pandas as pd

class Packet():
    """
    A class for wave packet initialization. Inherits from Wave_function and handles
    the creation of the initial wavefunction, based on the selected packet type.
    """

    def __init__(self, packet_type="gaussian",h_bar_tilde=1,dim =1, boundaries=None, potential=None, means=None, st_deviations=None, grids=None, dx=None, mass=1, omega=1,momenta = [0],
                 *args, **kwargs):
        """
        Initialize a Packet instance.

        Parameters:
            packet_type (str): Type of the wave packet ('gaussian', 'LHO', etc.).
            means (list): Mean positions for the wave packet in each dimension.
            st_deviations (list): Standard deviations for the wave packet in each dimension.
            grids (list[np.ndarray]): Precomputed grids for the domain (one grid per dimension).
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
        self.mass_s = mass
        self.h_bar_tilde = h_bar_tilde

        self.omega = omega
        self.momentum_propagator = self.compute_momentum_propagator()



        if self.grids is None or len(self.grids) != self.dim:
            raise ValueError("Grids must be provided for each dimension.")

    def compute_momentum_propagator(self):
        """Compute the kinetic propagator based on Fourier space components."""
        # Use single-precision floats to save memory
        momenta = [
            1j * np.array((momentum / self.h_bar_tilde) * grid, dtype=np.float32)
            for momentum, grid in zip(self.momenta, self.grids)]
        summed_momenta = np.zeros_like(momenta[0])
        for momentum in momenta:
            summed_momenta += momentum


        return np.exp(summed_momenta)

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

            wave_packet = self._create_ground_state(file_path)

            wave_packet *= self.momentum_propagator

            return wave_packet

        # Start creating the wavefunction
        if self.packet_type == "gaussian":

            wave_packet = self._create_gaussian_packet()

            wave_packet *= self.momentum_propagator

            return wave_packet
        elif self.packet_type == "LHO":
            wave_packet = self._create_LHO_packet()

            wave_packet *= self.momentum_propagator

            return wave_packet
        else:
            raise IncorrectPacketTypeError(
                message=f"The packet type '{self.packet_type}' is not recognized.",
                packet_type=self.packet_type
            )

    def _create_gaussian_packet(self):
        """
            Creates a Gaussian wave packet based on the provided means and standard deviations.

            Returns:
                    np.ndarray: The normalized Gaussian wavefunction.
            """
        means_arr = np.array(self.means)
        st_deviations_arr = np.array(self.st_deviations)
        # Stack grids for proper broadcasting
        # grids_stacked = np.stack(self.grids, axis=0)
        psi_0 = np.ones_like(self.grids[0])
        for i in range(len(self.grids)):
            psi_0 *= gaussian_packet(self.grids[i], means_arr[i], st_deviations_arr[i])

        dx_total = np.prod(np.array(self.dx))  # Total grid spacing in all dimensions
        # psi_0 = normalize_wavefunction(psi_0, dx_total)
        return psi_0 + 0j

    def _create_ground_state(self, file_path):
        """
        Creates a ground state wave packet from data file using interpolation.

        Parameters:
            file_path (str): Path to the file containing ground state data.

        Returns:
            np.ndarray: The interpolated ground state wavefunction.
        """

        data = self.read_ground_state_data(file_path)

        # Extract r and phi
        r_values = data[:, 0]
        phi_values = data[:, 1]

        # Interpolator for phi(r)
        interp_phi = interp1d(r_values, phi_values, kind='linear',
                              bounds_error=False, fill_value=0.0)

        # Compute r at each grid point
        r_distance = np.zeros_like(self.grids[0])
        for dim in range(self.dim):
            r_distance += (self.grids[dim] - self.means[dim]) ** 2
        r_distance = np.sqrt(r_distance)

        # Interpolate φ at each grid point
        psi_0 = interp_phi(r_distance)

        return psi_0.astype(np.complex64)

    def read_ground_state_data(self, file_path):
        """
        Reads the ground state wavefunction data from a file.

        Args:
            file_path (str): Path to the file containing wave function data.
            The expected format is:
            #r phi U
            0.0 1.0 -4.75
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

            return np.array(data)

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")
        except ValueError:
            raise ValueError(f"Invalid data format in file '{file_path}'.")
        except Exception as e:
            raise Exception(f"Error reading file '{file_path}': {str(e)}")

    # In _create_LHO_packet method
    def _create_LHO_packet(self):
        """
        Creates a wave packet for the linear harmonic oscillator (LHO).

        Returns:
            np.ndarray: The normalized LHO wavefunction.
        """
        # Pass self to lin_harmonic_oscillator
        psi_0 = lin_harmonic_oscillator(self)

        # Normalize the wavefunction over all dimensions
        dx_total = np.prod(np.array(self.dx))
        return psi_0 + 0j

    def compute_original_soliton_mass(self, file_path):
        """
        Compute the mass of the original soliton from file data,
        assuming spherical symmetry: M = 4π ∫ |phi(r)|² * r² dr

        Parameters:
            file_path (str): Path to the file with r and phi values.

        Returns:
            float: The computed original soliton mass.
        """
        # Load data from file using CPU (NumPy)
        data = self.read_ground_state_data(file_path)

        r = data[:, 0]
        phi = data[:, 1]
        # Compute density = |phi(r)|^2, assuming real phi
        density = phi ** 2
        # Compute integrand: ρ(r) * r²
        integrand = density * r ** 2
        # Integrate using the trapezoidal rule
        mass_integral = np.trapz(integrand, x=r)
        # Multiply by 4π for spherical symmetry
        M_original = 4 * np.pi * mass_integral
        return M_original

