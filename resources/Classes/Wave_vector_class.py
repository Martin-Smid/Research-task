import numpy as np
import cupy as cp

from resources.Classes.Wave_function_class import *
from numpy import random


class Wave_vector_class:
    """
    A class that represents wave functions with different spin states (0, 1, 2).

    This class handles the creation of wave vectors based on the provided wave function blueprint.
    It supports scalar (spin 0), vector (spin 1), and tensor (spin 2) representations as described
    in López-Sánchez et al. (2025) "Scaling relations and tidal disruption in spin s ultralight
    dark matter models".

    Uses CuPy for GPU-accelerated calculations wherever possible.

    Attributes:
        wave_blueprint (Wave_function_class): The wave function template to use
        spin (int): The spin of the wave vector (0, 1, or 2)
        polarization_coefficients (cupy.ndarray): Random coefficients for the polarization states
        polarization_phases (cupy.ndarray): Random phases for the polarization states
        polarization_bases (list): Basis vectors/tensors for the polarization states
        wave_vector (list): The resulting combined wave function
    """

    def __init__(self, wave_function, spin=0):
        """
        Initialize a Wave_vector_class instance.

        Args:
            wave_function (Wave_function_class): The wave function blueprint to use
            spin (int, optional): The spin state (0, 1, or 2). Defaults to 0.
        """
        self.wave_blueprint = wave_function
        self.spin = spin

        # Validate spin value
        if spin not in [0, 1, 2]:
            raise ValueError("Spin must be 0, 1, or 2")

        # Number of polarization states is 2*spin + 1
        num_polarization_states = 2 * spin + 1

        # Generate random coefficients and normalize them
        # Use NumPy for initial random generation then move to GPU with CuPy
        self.polarization_coefficients = cp.asarray(np.random.uniform(0, 1, num_polarization_states))
        self.polarization_coefficients = self.polarization_coefficients / cp.linalg.norm(self.polarization_coefficients)

        # Generate random phases
        self.polarization_phases = cp.asarray(cp.random.uniform(0, 2 * cp.pi, num_polarization_states))

        # Initialize polarization bases based on spin
        self._initialize_polarization_bases()

        # Calculate the combined wave function
        self.wave_vector = self._create_combined_wave_function()



    def _initialize_polarization_bases(self):
        """
        Initialize the basis vectors/tensors for the polarization states based on spin.

        For spin 0: Single scalar state
        For spin 1: Three vector states with polarizations -1, 0, +1
        For spin 2: Five tensor states with polarizations -2, -1, 0, +1, +2

        All tensors are stored as CuPy arrays for GPU acceleration.
        """
        if self.spin == 0:
            # Spin 0 has a single scalar state
            self.polarization_bases = [cp.array(1)]

        elif self.spin == 1:
            # Spin 1 has three vector states with polarizations -1, 0, +1
            self.polarization_bases = [
                1 / cp.sqrt(2) * cp.array([1, +1j, 0]),  # Polarization +1
                1 / cp.sqrt(2) * cp.array([1, -1j, 0]),  # Polarization -1
                cp.array([0, 0, 1])  # Polarization 0
            ]

        elif self.spin == 2:
            # Spin 2 has five tensor states with polarizations -2, -1, 0, +1, +2
            self.polarization_bases = [
                # Polarization +2
                1 / 2 * cp.array([
                    [1, 1j, 0],
                    [1j, -1, 0],
                    [0, 0, 0]
                ]),

                # Polarization +1
                1 / 2 * cp.array([
                    [0, 0, 1],
                    [0, 0, 1j],
                    [1, 1j, 0]
                ]),

                # Polarization 0
                1 / cp.sqrt(6) * cp.array([
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 2]
                ]),

                # Polarization -1
                1 / 2 * cp.array([
                    [0, 0, 1],
                    [0, 0, -1j],
                    [1, -1j, 0]
                ]),

                # Polarization -2
                1 / 2 * cp.array([
                    [1, -1j, 0],
                    [-1j, -1, 0],
                    [0, 0, 0]
                ])
            ]

    def _create_combined_wave_function(self):
        """
        Create the combined wave function by summing over all polarization states.

        This implements equation 3 from the paper:
        Ψ(t, x) = ∑_p ψ_p(t, x) ε^(p)

        Where:
        - ψ_p is the field with polarization p (ψ_sol * c_p * e^(-i*θ_p))
        - ε^(p) is the basis vector/tensor for polarization p

        Returns:
            list: Wave functions that are copies of wave_blueprint with updated .psi attributes
        """
        # Import copy to create deep copies of Wave_function_class instances
        import copy

        # Create a list to store our wave functions
        wave_vector = []

        if self.spin == 0:
            # For spin 0, create a single new wave function with updated psi
            new_wave_function = copy.deepcopy(self.wave_blueprint)
            new_wave_function.psi = self.wave_blueprint.psi * self.polarization_coefficients[0] * cp.exp(
                -1j * self.polarization_phases[0])
            wave_vector = [new_wave_function]

        elif self.spin == 1:
            # For spin 1, we compute the weighted sum of the three polarization states
            # First multiply each basis vector by its coefficient
            weighted_bases = [self.polarization_bases[p] * self.polarization_coefficients[p] *
                              cp.exp(-1j * self.polarization_phases[p]) for p in range(3)]

            # Sum these weighted bases to get the combined coefficient vector
            combined_coefficients = cp.sum(cp.array(weighted_bases), axis=0)

            # For each component of the vector, create a new wave function
            for i in range(len(combined_coefficients)):
                new_wave_function = copy.deepcopy(self.wave_blueprint)
                new_wave_function.psi = self.wave_blueprint.psi * combined_coefficients[i]
                wave_vector.append(new_wave_function)

        elif self.spin == 2:
            # For spin 2, we compute the weighted sum of the five polarization states
            # First multiply each basis tensor by its coefficient
            weighted_bases = [self.polarization_bases[p] * self.polarization_coefficients[p] *
                              cp.exp(-1j * self.polarization_phases[p]) for p in range(5)]

            # Sum these weighted bases to get the combined coefficient tensor
            combined_coefficients = cp.sum(cp.array(weighted_bases), axis=0)

            # For each component of the tensor, create a new wave function
            for i in range(combined_coefficients.shape[0]):
                for j in range(combined_coefficients.shape[1]):
                    new_wave_function = copy.deepcopy(self.wave_blueprint)
                    new_wave_function.psi = self.wave_blueprint.psi * combined_coefficients[i, j]
                    wave_vector.append(new_wave_function)

        return wave_vector