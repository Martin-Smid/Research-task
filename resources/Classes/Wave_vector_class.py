import gc
from itertools import combinations_with_replacement, permutations

import numpy as np

from resources.Classes.Wave_function_class import Wave_function


class Wave_vector_class:
    """Construct an irreducible integer-spin ULDM wave vector."""

    def __init__(
        self,
        spin=0,
        polarization_coefficients=None,
        polarization_phases=None,
        random_seed=None,
        **wave_function_kwargs,
    ):
        self._validate_spin(spin)
        self.spin = spin
        self.m_values = tuple(range(-spin, spin + 1))
        # Tensor metadata is retained for inspection/validation, but the GPU
        # evolution stores the smaller set of 2s+1 polarization amplitudes.
        self.index_combinations = list(
            combinations_with_replacement(range(3), spin)
        )
        self.num_tensor_components = len(self.index_combinations)
        self.num_components = 2 * spin + 1
        self.index_multiplicities = get_index_multiplicities(
            self.index_combinations
        )

        self.wave_blueprint = Wave_function(**wave_function_kwargs)
        # Preserve compatibility with existing scripts that call
        # np.random.seed(...), while also allowing a per-vector seed.
        rng = (
            np.random
            if random_seed is None
            else np.random.default_rng(random_seed)
        )
        self.polarization_coefficients = self._prepare_coefficients(
            polarization_coefficients, rng
        )
        self.polarization_phases = self._prepare_phases(
            polarization_phases, rng
        )
        self.polarization_bases = self.generate_spin_basis(spin)
        self.wave_vector = self._create_polarization_wave_functions()

    @staticmethod
    def _validate_spin(spin):
        if isinstance(spin, bool) or not isinstance(spin, (int, np.integer)):
            raise TypeError("Spin must be a non-negative integer")
        if spin < 0:
            raise ValueError("Spin must be a non-negative integer")

    def _prepare_coefficients(self, coefficients, rng):
        size = 2 * self.spin + 1
        values = (
            rng.uniform(-1.0, 1.0, size)
            if coefficients is None
            else np.asarray(coefficients, dtype=float)
        )
        if values.shape != (size,):
            raise ValueError(
                f"polarization_coefficients must have shape ({size},), "
                f"ordered by m={self.m_values}"
            )
        norm = np.linalg.norm(values)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("Polarization coefficients need a finite non-zero norm")
        return values / norm

    def _prepare_phases(self, phases, rng):
        size = 2 * self.spin + 1
        values = (
            rng.uniform(0.0, 2.0 * np.pi, size)
            if phases is None
            else np.asarray(phases, dtype=float)
        )
        if values.shape != (size,):
            raise ValueError(
                f"polarization_phases must have shape ({size},), "
                f"ordered by m={self.m_values}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Polarization phases must be finite")
        return values

    @staticmethod
    def spin_one_basis():
        return {
            -1: np.array([1.0, -1.0j, 0.0], complex) / np.sqrt(2.0),
            0: np.array([0.0, 0.0, 1.0], complex),
            1: np.array([-1.0, -1.0j, 0.0], complex) / np.sqrt(2.0),
        }

    @staticmethod
    def _maximal_cg_coefficient(j, m, q):
        """Return <j,m;1,q|j+1,m+q>."""
        denominator = (j + 1) * (2 * j + 1)
        if q == 1:
            return np.sqrt(
                (j + m + 1) * (j + m + 2) / (2.0 * denominator)
            )
        if q == 0:
            return np.sqrt(
                (j - m + 1) * (j + m + 1) / denominator
            )
        if q == -1:
            return np.sqrt(
                (j - m + 1) * (j - m + 2) / (2.0 * denominator)
            )
        raise ValueError("q must be -1, 0, or 1")

    @classmethod
    def generate_spin_basis(cls, spin):
        """Return tensors ordered by m=-spin,...,+spin."""
        cls._validate_spin(spin)
        if spin == 0:
            return [np.array(1.0, dtype=complex)]

        spin_one = cls.spin_one_basis()
        basis_by_m = dict(spin_one)

        for total_spin in range(2, spin + 1):
            previous_spin = total_spin - 1
            next_basis = {}
            for total_m in range(-total_spin, total_spin + 1):
                tensor = np.zeros((3,) * total_spin, dtype=complex)
                for q, vector in spin_one.items():
                    previous_m = total_m - q
                    if previous_m not in basis_by_m:
                        continue
                    coefficient = cls._maximal_cg_coefficient(
                        previous_spin, previous_m, q
                    )
                    tensor += coefficient * np.tensordot(
                        basis_by_m[previous_m], vector, axes=0
                    )
                norm = np.linalg.norm(tensor)
                if not np.isfinite(norm) or norm == 0.0:
                    raise RuntimeError(
                        f"Failed to construct spin={total_spin}, m={total_m}"
                    )
                next_basis[total_m] = tensor / norm
            basis_by_m = next_basis

        return [basis_by_m[m] for m in range(-spin, spin + 1)]

    def _create_polarization_wave_functions(self):
        """Create the directly evolved fields psi_m, ordered by m."""
        result = []
        momentum_factor = (
            1.0
            if self.wave_blueprint.packet_type == "gaussian"
            else self.wave_blueprint.packet_creator.momentum_propagator
        )

        for coefficient, phase in zip(
            self.polarization_coefficients, self.polarization_phases
        ):
            polarization_amplitude = coefficient * np.exp(-1j * phase)
            psi_m = (
                self.wave_blueprint.psi
                * polarization_amplitude
                * momentum_factor
            )
            result.append(
                self.wave_blueprint.softcopy_psi(
                    psi_m, multiplicity=1
                )
            )
        return result

    def cleanup_wave_vector(self):
        self.wave_vector.clear()
        self.wave_blueprint = None
        self.polarization_bases = None
        self.polarization_coefficients = None
        self.polarization_phases = None
        gc.collect()
        try:
            from resources.Classes.Wave_function_class import cp

            cp.get_default_memory_pool().free_all_blocks()
        except (ImportError, AttributeError):
            pass


def get_index_multiplicities(index_combinations):
    return {
        index: len(set(permutations(index)))
        for index in index_combinations
    }
