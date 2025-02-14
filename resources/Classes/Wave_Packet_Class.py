from resources.Classes.Base_Wave_Function import Base_Wave_function
from resources.Functions.Schrodinger_eq_functions import *
from resources.Errors.Errors import *
import cupy as cp


class Packet(Base_Wave_function):
    """
    A class for wave packet initialization. Inherits from Wave_function and handles
    the creation of the initial wavefunction, based on the selected packet type.
    """

    def __init__(self, packet_type="gaussian", means=None, st_deviations=None, grids=None, dx=None, mass=1, omega=1,
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
        super().__init__(*args, **kwargs)  # Call the parent initializer
        self.packet_type = packet_type
        self.means = means if means else [0] * self.dim
        self.st_deviations = st_deviations if st_deviations else [0.1] * self.dim
        self.grids = grids  # Receive grids from parent class

        self.dx = dx  # Receive dx from parent class
        self.mass = mass  # New attribute for particle mass
        self.omega = omega  # New attribute for harmonic oscillator frequency

        if self.grids is None or len(self.grids) != self.dim:
            raise ValueError("Grids must be provided for each dimension.")


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

        # Start creating the wavefunction
        if self.packet_type == "gaussian":
            return self._create_gaussian_packet()
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
        # Compute ψ₀ for the first dimension
        # Compute Gaussian wave packet directly over the full meshgrid
        psi_0 = cp.exp(-cp.sum(((self.grids - cp.array(self.means)) ** 2) /
                               (2 * cp.array(self.st_deviations) ** 2), axis=0), dtype=cp.complex64)

        # Normalize the wavefunction over all dimensions
        dx_total = cp.prod(cp.array(self.dx))  # Total grid spacing in all dimensions
        psi_0 = normalize_wavefunction(psi_0, dx_total)
        return psi_0 + 0j  # Ensure complex128 type for `psi_0`

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
        psi_0 = normalize_wavefunction(psi_0, dx_total)
        return psi_0 + 0j

