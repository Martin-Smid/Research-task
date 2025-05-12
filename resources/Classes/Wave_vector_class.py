import numpy as np
import cupy as cp

from resources.Classes.Wave_function_class import *
from numpy import random
from resources.Errors.Errors import IncorrectWaveBlueprintError


class Wave_vector_class:
    """
    A class that represents wave functions with different spin states (0, 1, 2, 3).

    This class handles the creation of wave vectors based on the provided wave function blueprint.
    It supports scalar (spin 0), vector (spin 1), tensor (spin 2), and rank-3 tensor (spin 3)
    representations as described in López-Sánchez et al. (2025) "Scaling relations and tidal disruption
    in spin s ultralight dark matter models".

    Uses CuPy for GPU-accelerated calculations wherever possible.

    Attributes:
        wave_blueprint (Wave_function_class or list): The wave function template(s) to use
        spin (int): The spin of the wave vector (0, 1, 2, or 3)
        polarization_coefficients (cupy.ndarray): Random coefficients for the polarization states
        polarization_phases (cupy.ndarray): Random phases for the polarization states
        polarization_bases (list): Basis vectors/tensors for the polarization states
        wave_vector (list): The resulting combined wave function
    """

    def __init__(self, wave_function, spin=0):
        """
        Initialize a Wave_vector_class instance.

        Args:
            wave_function (Wave_function_class or list): The wave function blueprint(s) to use
            spin (int, optional): The spin state (0, 1, 2, or 3). Defaults to 0.
        """
        self.spin = spin

        # Validate spin value
        if spin not in [0, 1, 2, 3]:
            raise ValueError("Spin must be 0, 1, 2, or 3")

        # Validate and set wave_blueprint
        self.wave_blueprint = self._validate_wave_blueprint(wave_function)

        # Number of polarization states is 2*spin + 1
        num_polarization_states = 2 * spin + 1

        self.polarization_coefficients = cp.asarray(np.random.uniform(0, 1, num_polarization_states))
        self.polarization_coefficients = self.polarization_coefficients / cp.linalg.norm(self.polarization_coefficients)

        self.polarization_phases = cp.asarray(cp.random.uniform(0, 2 * cp.pi, num_polarization_states))

        # Initialize polarization bases based on spin
        self._initialize_polarization_bases()

        # Calculate the combined wave function
        self.wave_vector = self._create_combined_wave_function()

    def _validate_wave_blueprint(self, wave_blueprint):
        """
        Validate the provided wave blueprint based on spin value.

        For any spin:
        - Single wave function or list of 1 wave function is valid

        Additionally:
        - For spin 1: List of 3 wave functions is also valid
        - For spin 2: List of 9 wave functions is also valid
        - For spin 3: List of 27 wave functions is also valid

        Args:
            wave_blueprint: Single Wave_function_class or list of Wave_function_class instances

        Returns:
            List of validated Wave_function_class instances

        Raises:
            IncorrectWaveBlueprintError: If the provided blueprint doesn't match requirements
        """
        # How many wave functions are expected based on spin
        expected_counts = {
            0: [1],
            1: [1, 3],
            2: [1, 3, 9],
            3: [1, 3, 9, 27]
        }

        if self.spin not in expected_counts:
            raise ValueError(f"Invalid spin {self.spin}. Must be 0, 1, 2, or 3.")

        allowed_counts = expected_counts[self.spin]

        # Handle single Wave_function instance
        if isinstance(wave_blueprint, Wave_function):
            return [wave_blueprint]  # Always wrap in list for consistent handling

        # Handle list of Wave_function instances
        elif isinstance(wave_blueprint, list):
            if all(isinstance(wf, Wave_function) for wf in wave_blueprint):
                count = len(wave_blueprint)
                if count in allowed_counts:
                    return wave_blueprint
                else:
                    raise IncorrectWaveBlueprintError(
                        f"For spin {self.spin}, expected {allowed_counts} wave functions, but got {count}",
                        provided=count)
            else:
                non_wave_types = [type(item) for item in wave_blueprint if not isinstance(item, Wave_function)]
                raise IncorrectWaveBlueprintError(
                    "All elements must be instances of Wave_function_class",
                    provided=f"List containing types: {non_wave_types}")
        else:
            raise IncorrectWaveBlueprintError(
                "Wave blueprint must be a Wave_function_class instance or a list of them",
                provided=type(wave_blueprint))

    def _initialize_polarization_bases(self):
        """
        Initialize the basis vectors/tensors for the polarization states based on spin.

        For spin 0: Single scalar state
        For spin 1: Three vector states with polarizations -1, 0, +1
        For spin 2: Five tensor states with polarizations -2, -1, 0, +1, +2
        For spin 3: Seven rank-3 tensor states with polarizations -3, -2, -1, 0, +1, +2, +3

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

        elif self.spin == 3:
            # Spin 3 has seven rank-3 tensor states with polarizations -3, -2, -1, 0, +1, +2, +3
            # Initialize empty 3D arrays for each polarization state
            pol_shape = (3, 3, 3)  # 3×3×3 tensor for rank-3

            # Create empty tensors for each polarization state
            p_plus3 = cp.zeros(pol_shape, dtype=cp.complex128)
            p_plus2 = cp.zeros(pol_shape, dtype=cp.complex128)
            p_plus1 = cp.zeros(pol_shape, dtype=cp.complex128)
            p_zero = cp.zeros(pol_shape, dtype=cp.complex128)
            p_minus1 = cp.zeros(pol_shape, dtype=cp.complex128)
            p_minus2 = cp.zeros(pol_shape, dtype=cp.complex128)
            p_minus3 = cp.zeros(pol_shape, dtype=cp.complex128)

            # Polarization +3
            # Most elements are zero except for specific components
            p_plus3[0, 0, 0] = 1
            p_plus3[0, 0, 1] = p_plus3[0, 1, 0] = p_plus3[1, 0, 0] = 3j
            p_plus3[0, 1, 1] = p_plus3[1, 0, 1] = p_plus3[1, 1, 0] = -3
            p_plus3[1, 1, 1] = -1j
            p_plus3 = p_plus3 / (2 * cp.sqrt(10))  # Normalization factor

            # Polarization +2
            # Fill select components for the +2 polarization
            p_plus2[0, 0, 2] = p_plus2[0, 2, 0] = p_plus2[2, 0, 0] = 1
            p_plus2[0, 1, 2] = p_plus2[0, 2, 1] = p_plus2[1, 0, 2] = p_plus2[1, 2, 0] = p_plus2[2, 0, 1] = p_plus2[
                2, 1, 0] = 1j
            p_plus2[1, 1, 2] = p_plus2[1, 2, 1] = p_plus2[2, 1, 1] = -1
            p_plus2 = p_plus2 / (2 * cp.sqrt(15))  # Normalization factor

            # Polarization +1
            # Fill select components for the +1 polarization
            p_plus1[0, 2, 2] = p_plus1[2, 0, 2] = p_plus1[2, 2, 0] = 1
            p_plus1[1, 2, 2] = p_plus1[2, 1, 2] = p_plus1[2, 2, 1] = 1j
            p_plus1 = p_plus1 / (2 * cp.sqrt(5))  # Normalization factor

            # Polarization 0
            # The zero polarization state is typically diagonal
            p_zero[0, 0, 0] = p_zero[1, 1, 1] = -3
            p_zero[2, 2, 2] = 6
            p_zero = p_zero / (6 * cp.sqrt(10))  # Normalization factor

            # Polarization -1
            # Conjugate of +1 polarization
            p_minus1[0, 2, 2] = p_minus1[2, 0, 2] = p_minus1[2, 2, 0] = 1
            p_minus1[1, 2, 2] = p_minus1[2, 1, 2] = p_minus1[2, 2, 1] = -1j
            p_minus1 = p_minus1 / (2 * cp.sqrt(5))  # Normalization factor

            # Polarization -2
            # Conjugate of +2 polarization
            p_minus2[0, 0, 2] = p_minus2[0, 2, 0] = p_minus2[2, 0, 0] = 1
            p_minus2[0, 1, 2] = p_minus2[0, 2, 1] = p_minus2[1, 0, 2] = p_minus2[1, 2, 0] = p_minus2[2, 0, 1] = \
            p_minus2[2, 1, 0] = -1j
            p_minus2[1, 1, 2] = p_minus2[1, 2, 1] = p_minus2[2, 1, 1] = -1
            p_minus2 = p_minus2 / (2 * cp.sqrt(15))  # Normalization factor

            # Polarization -3
            # Conjugate of +3 polarization
            p_minus3[0, 0, 0] = 1
            p_minus3[0, 0, 1] = p_minus3[0, 1, 0] = p_minus3[1, 0, 0] = -3j
            p_minus3[0, 1, 1] = p_minus3[1, 0, 1] = p_minus3[1, 1, 0] = -3
            p_minus3[1, 1, 1] = 1j
            p_minus3 = p_minus3 / (2 * cp.sqrt(10))  # Normalization factor

            # Store all polarization bases in a list
            self.polarization_bases = [
                p_plus3,  # Polarization +3
                p_plus2,  # Polarization +2
                p_plus1,  # Polarization +1
                p_zero,  # Polarization 0
                p_minus1,  # Polarization -1
                p_minus2,  # Polarization -2
                p_minus3  # Polarization -3
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
            new_wave_function = copy.deepcopy(self.wave_blueprint[0])
            new_wave_function.psi = self.wave_blueprint[0].psi * self.polarization_coefficients[0] * cp.exp(
                -1j * self.polarization_phases[0])
            wave_vector = [new_wave_function]

        elif self.spin == 1:
            # For spin 1, we handle both cases: single wave function or list of 3
            if len(self.wave_blueprint) == 1:
                # Use a single wave function as blueprint for all polarizations
                # Compute the weighted sum of the three polarization states
                weighted_bases = [self.polarization_bases[p] * self.polarization_coefficients[p] *
                                  cp.exp(-1j * self.polarization_phases[p]) for p in range(3)]

                # Sum these weighted bases to get the combined coefficient vector
                combined_coefficients = cp.sum(cp.array(weighted_bases), axis=0)

                # For each component of the vector, create a new wave function
                for i in range(len(combined_coefficients)):
                    new_wave_function = copy.deepcopy(self.wave_blueprint[0])
                    new_wave_function.psi = self.wave_blueprint[0].psi * combined_coefficients[i]
                    wave_vector.append(new_wave_function)
            else:
                # Use three separate wave functions, one for each polarization
                for p in range(3):
                    new_wave_function = copy.deepcopy(self.wave_blueprint[p])
                    new_wave_function.psi = self.wave_blueprint[p].psi * self.polarization_coefficients[p] * cp.exp(
                        -1j * self.polarization_phases[p])
                    wave_vector.append(new_wave_function)

        elif self.spin == 2:
            if len(self.wave_blueprint) == 1:
                # Use a single wave function as blueprint for all polarizations
                # Compute the weighted sum of the five polarization states
                weighted_bases = [self.polarization_bases[p] * self.polarization_coefficients[p] *
                                  cp.exp(-1j * self.polarization_phases[p]) for p in range(5)]

                # Sum these weighted bases to get the combined coefficient tensor
                combined_coefficients = cp.sum(cp.array(weighted_bases), axis=0)

                # For each component of the tensor, create a new wave function
                for i in range(combined_coefficients.shape[0]):
                    for j in range(combined_coefficients.shape[1]):
                        new_wave_function = copy.deepcopy(self.wave_blueprint[0])
                        new_wave_function.psi = self.wave_blueprint[0].psi * combined_coefficients[i, j]
                        wave_vector.append(new_wave_function)
            elif len(self.wave_blueprint) == 3:
                # Use three separate wave functions, one for each spatial direction
                idx = 0
                for i in range(3):
                    for j in range(3):
                        if idx >= len(self.wave_blueprint):
                            # This shouldn't happen with proper validation
                            break

                        # Find which components to apply from each polarization state
                        polarization_sum = cp.zeros((), dtype=cp.complex128)
                        for p in range(5):  # 5 polarization states for spin 2
                            if i < self.polarization_bases[p].shape[0] and j < self.polarization_bases[p].shape[1]:
                                polarization_sum += self.polarization_bases[p][i, j] * self.polarization_coefficients[
                                    p] * cp.exp(
                                    -1j * self.polarization_phases[p])

                        new_wave_function = copy.deepcopy(self.wave_blueprint[idx % 3])
                        new_wave_function.psi = self.wave_blueprint[idx % 3].psi * polarization_sum
                        wave_vector.append(new_wave_function)
                        idx += 1
            else:
                # Use nine separate wave functions, one for each tensor component
                idx = 0
                for i in range(3):
                    for j in range(3):
                        new_wave_function = copy.deepcopy(self.wave_blueprint[idx])

                        # Find which components to apply from each polarization state
                        polarization_sum = cp.zeros((), dtype=cp.complex128)
                        for p in range(5):  # 5 polarization states for spin 2
                            if i < self.polarization_bases[p].shape[0] and j < self.polarization_bases[p].shape[1]:
                                polarization_sum += self.polarization_bases[p][i, j] * self.polarization_coefficients[
                                    p] * cp.exp(
                                    -1j * self.polarization_phases[p])

                        new_wave_function.psi = self.wave_blueprint[idx].psi * polarization_sum
                        wave_vector.append(new_wave_function)
                        idx += 1

        elif self.spin == 3:
            # For spin 3, we need to handle rank-3 tensors (3×3×3)
            if len(self.wave_blueprint) == 1:
                # Use a single wave function as blueprint for all polarizations
                # Compute the weighted sum of the seven polarization states
                weighted_bases = [self.polarization_bases[p] * self.polarization_coefficients[p] *
                                  cp.exp(-1j * self.polarization_phases[p]) for p in range(7)]

                # Sum these weighted bases to get the combined coefficient tensor
                combined_coefficients = cp.sum(cp.array(weighted_bases), axis=0)

                # For each component of the rank-3 tensor, create a new wave function
                for i in range(combined_coefficients.shape[0]):
                    for j in range(combined_coefficients.shape[1]):
                        for k in range(combined_coefficients.shape[2]):
                            new_wave_function = copy.deepcopy(self.wave_blueprint[0])
                            new_wave_function.psi = self.wave_blueprint[0].psi * combined_coefficients[i, j, k]
                            wave_vector.append(new_wave_function)

            elif len(self.wave_blueprint) == 3:
                # Use three wave functions as blueprints
                idx = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if idx >= len(self.wave_blueprint) * 9:  # 3 blueprints * 9 components each
                                break

                            # Find which components to apply from each polarization state
                            polarization_sum = cp.zeros((), dtype=cp.complex128)
                            for p in range(7):  # 7 polarization states for spin 3
                                polarization_sum += self.polarization_bases[p][i, j, k] * \
                                                    self.polarization_coefficients[p] * cp.exp(
                                    -1j * self.polarization_phases[p])

                            new_wave_function = copy.deepcopy(self.wave_blueprint[idx % 3])
                            new_wave_function.psi = self.wave_blueprint[idx % 3].psi * polarization_sum
                            wave_vector.append(new_wave_function)
                            idx += 1

            elif len(self.wave_blueprint) == 9:
                # Use nine wave functions as blueprints
                idx = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if idx >= len(self.wave_blueprint) * 3:  # 9 blueprints * 3 components each
                                break

                            # Find which components to apply from each polarization state
                            polarization_sum = cp.zeros((), dtype=cp.complex128)
                            for p in range(7):  # 7 polarization states for spin 3
                                polarization_sum += self.polarization_bases[p][i, j, k] * \
                                                    self.polarization_coefficients[p] * cp.exp(
                                    -1j * self.polarization_phases[p])

                            new_wave_function = copy.deepcopy(self.wave_blueprint[idx % 9])
                            new_wave_function.psi = self.wave_blueprint[idx % 9].psi * polarization_sum
                            wave_vector.append(new_wave_function)
                            idx += 1

            else:
                # Use 27 separate wave functions, one for each tensor component
                idx = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            new_wave_function = copy.deepcopy(self.wave_blueprint[idx])

                            # Find which components to apply from each polarization state
                            polarization_sum = cp.zeros((), dtype=cp.complex128)
                            for p in range(7):  # 7 polarization states for spin 3
                                polarization_sum += self.polarization_bases[p][i, j, k] * \
                                                    self.polarization_coefficients[p] * cp.exp(
                                    -1j * self.polarization_phases[p])

                            new_wave_function.psi = self.wave_blueprint[idx].psi * polarization_sum
                            wave_vector.append(new_wave_function)
                            idx += 1

        return wave_vector