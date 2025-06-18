import numpy as np

from sympy.physics.quantum.cg import CG
from sympy import S


from resources.Classes.Wave_function_class import *
from numpy import random
from resources.Errors.Errors import IncorrectWaveBlueprintError
np.random.seed(12345)

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

    def __init__(self, spin=0, **wave_function_kwargs):
        """
        Initialize a Wave_vector_class instance, automatically constructing the required Wave_function(s)
        from the given keyword arguments.

        Args:
            spin (int): The spin state (0, 1, 2, or 3). Defaults to 0.
            **wave_function_kwargs: All the arguments required to construct a Wave_function instance.
        """
        from resources.Classes.Wave_function_class import Wave_function

        self.spin = spin

        # How many wave functions to generate
        count_map = {0: 1, 1: 1, 2: 1, 3: 1,4:1}  # could default to more if needed
        num_wavefunctions = count_map[spin]

        # Build the wave_blueprint: a list of Wave_function instances
        self.wave_blueprint = [Wave_function(**wave_function_kwargs) for _ in range(num_wavefunctions)]


        # Continue as before
        self._setup_vector()

    def _setup_vector(self):

        num_polarization_states = 2 * self.spin + 1
        self.polarization_coefficients = np.asarray(np.random.uniform(-1, 1, num_polarization_states))
        print(self.polarization_coefficients)
        self.polarization_coefficients = self.polarization_coefficients / np.linalg.norm(self.polarization_coefficients)
        self.polarization_phases = np.asarray(np.random.uniform(0, 2 * np.pi, num_polarization_states))
        #self.polarization_phases /= np.linalg.norm(self.polarization_phases)

        self._initialize_polarization_bases()
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
        self.polarization_bases = self.generate_spin_basis()

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
            new_wave_function.psi = self.wave_blueprint[0].psi * self.polarization_coefficients[0] * np.exp(
                -1j * self.polarization_phases[0])
            wave_vector = [new_wave_function]

        elif self.spin == 1:
            # For spin 1, we handle both cases: single wave function or list of 3
            if len(self.wave_blueprint) == 1:
                # Use a single wave function as blueprint for all polarizations
                # Compute the weighted sum of the three polarization states
                weighted_bases = [self.polarization_bases[p] * self.polarization_coefficients[p] *
                                  np.exp(-1j * self.polarization_phases[p]) for p in range(3)]

                # Sum these weighted bases to get the combined coefficient vector
                combined_coefficients = np.sum(np.array(weighted_bases), axis=0)

                # For each component of the vector, create a new wave function
                for i in range(len(combined_coefficients)):
                    new_wave_function = copy.deepcopy(self.wave_blueprint[0])
                    new_wave_function.psi = self.wave_blueprint[0].psi * combined_coefficients[i]
                    wave_vector.append(new_wave_function)
            else:
                # Use three separate wave functions, one for each polarization
                for p in range(3):
                    new_wave_function = copy.deepcopy(self.wave_blueprint[p])
                    new_wave_function.psi = self.wave_blueprint[p].psi * self.polarization_coefficients[p] * np.exp(
                        -1j * self.polarization_phases[p])
                    wave_vector.append(new_wave_function)

        elif self.spin == 2:
            if len(self.wave_blueprint) == 1:
                # Use a single wave function as blueprint for all polarizations
                # Compute the weighted sum of the five polarization states
                weighted_bases = [self.polarization_bases[p] * self.polarization_coefficients[p] *
                                  np.exp(-1j * self.polarization_phases[p]) for p in range(5)]

                # Sum these weighted bases to get the combined coefficient tensor
                combined_coefficients = np.sum(np.array(weighted_bases), axis=0)

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
                        polarization_sum = np.zeros((), dtype=np.complex128)
                        for p in range(5):  # 5 polarization states for spin 2
                            if i < self.polarization_bases[p].shape[0] and j < self.polarization_bases[p].shape[1]:
                                polarization_sum += self.polarization_bases[p][i, j] * self.polarization_coefficients[
                                    p] * np.exp(
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
                        polarization_sum = np.zeros((), dtype=np.complex128)
                        for p in range(5):  # 5 polarization states for spin 2
                            if i < self.polarization_bases[p].shape[0] and j < self.polarization_bases[p].shape[1]:
                                polarization_sum += self.polarization_bases[p][i, j] * self.polarization_coefficients[
                                    p] * np.exp(
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
                                  np.exp(-1j * self.polarization_phases[p]) for p in range(7)]

                # Sum these weighted bases to get the combined coefficient tensor
                combined_coefficients = np.sum(np.array(weighted_bases), axis=0)

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
                            polarization_sum = np.zeros((), dtype=np.complex128)
                            for p in range(7):  # 7 polarization states for spin 3
                                polarization_sum += self.polarization_bases[p][i, j, k] * \
                                                    self.polarization_coefficients[p] * np.exp(
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
                            polarization_sum = np.zeros((), dtype=np.complex128)
                            for p in range(7):  # 7 polarization states for spin 3
                                polarization_sum += self.polarization_bases[p][i, j, k] * \
                                                    self.polarization_coefficients[p] * np.exp(
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
                            polarization_sum = np.zeros((), dtype=np.complex128)
                            for p in range(7):  # 7 polarization states for spin 3
                                polarization_sum += self.polarization_bases[p][i, j, k] * \
                                                    self.polarization_coefficients[p] * np.exp(
                                    -1j * self.polarization_phases[p])

                            new_wave_function.psi = self.wave_blueprint[idx].psi * polarization_sum
                            wave_vector.append(new_wave_function)
                            idx += 1

        return wave_vector

    def generate_spin_basis(self):
        """
        Generate polarization bases for arbitrary integer spin using Clebsch-Gordan coefficients.

        Args:
            spin (int): Total spin (e.g., 0, 1, 2, 3...)

        Returns:
            list of numpy.ndarray: Tensors representing each m state (m = -s ... +s)
        """
        spin = self.spin

        if spin == 0:
            return [np.array(1.0)]

        dim = 3  # Spatial dimension
        m_values = range(-spin, spin + 1)
        basis = []

        # Define unit vectors for spin-1 states
        e_m = {
            -1: np.array([1, -1j, 0]) / np.sqrt(2),
            0: np.array([0, 0, 1]),
            1: np.array([-1, -1j, 0]) / np.sqrt(2),
        }

        # Iterate over all m for given spin
        for m in m_values:
            tensor = np.zeros([dim] * spin, dtype=complex)

            # Sum over all m1, m2, ..., ms such that sum(mi) = m
            def recursive_combination(ms=(), total=0):
                if len(ms) == spin:
                    if total == m:
                        # Get full CG coefficient product
                        cg_total = 1
                        valid = True
                        for i in range(spin - 1):
                            j1 = i + 1 if i > 0 else 1
                            m1 = sum(ms[:i + 1]) if i > 0 else ms[i]
                            j2 = 1
                            m2 = ms[i + 1]
                            j_total = j1 + j2
                            cg_obj = CG(S(j1), S(m1), S(j2), S(m2), S(j_total), S(m1 + m2))
                            cg = cg_obj.doit()

                            if cg == 0:
                                valid = False
                                break
                            cg_total *= cg

                        if not valid:
                            return

                        # Build tensor product of vectors
                        vecs = [e_m[mi] for mi in ms]
                        component = vecs[0]
                        for v in vecs[1:]:
                            component = np.tensordot(component, v, axes=0)
                        tensor[:] += float(cg_total) * component


                else:
                    for mi in [-1, 0, 1]:
                        recursive_combination(ms + (mi,), total + mi)

            recursive_combination()

            # Normalize the tensor
            norm = np.linalg.norm(tensor)
            if norm != 0:
                tensor = tensor / norm

            basis.append(tensor)

        return basis