import numpy as np

from sympy.physics.quantum.cg import CG
from sympy import S


from resources.Classes.Wave_function_class import *
from numpy import random
from resources.Errors.Errors import IncorrectWaveBlueprintError
from itertools import combinations_with_replacement


np.random.seed(12345)

import numpy as np
from itertools import combinations_with_replacement
import copy

from resources.Classes.Wave_function_class import Wave_function

class Wave_vector_class:
    def __init__(self, spin=0, **wave_function_kwargs):
        self.spin = spin
        if spin < 0:
            raise ValueError("Spin must be a non-negative integer")

        self.index_combinations = list(combinations_with_replacement(range(3), spin))
        self.num_components = len(self.index_combinations)

        self.wave_blueprint = Wave_function(**wave_function_kwargs)

        self.polarization_coefficients = np.random.uniform(-1, 1, 2 * spin + 1)
        self.polarization_coefficients /= np.linalg.norm(self.polarization_coefficients)
        self.polarization_phases = np.random.uniform(0, 2 * np.pi, 2 * spin + 1)

        self.index_combinations = list(combinations_with_replacement(range(3), spin))
        self.index_multiplicities = get_index_multiplicities(self.index_combinations)

        self.polarization_bases = self.generate_spin_basis()
        self.wave_vector = self._create_combined_wave_function()

    def generate_spin_basis(self):
        if self.spin == 0:
            return [np.array(1.0)]

        from sympy.physics.quantum.cg import CG
        from sympy import S

        e_m = {
            -1: np.array([1, -1j, 0]) / np.sqrt(2),
            0: np.array([0, 0, 1]),
            1: np.array([-1, -1j, 0]) / np.sqrt(2)
        }

        basis = []
        for m in range(-self.spin, self.spin + 1):
            tensor = np.zeros([3] * self.spin, dtype=complex)

            def build(ms=(), total=0):
                if len(ms) == self.spin:
                    if total != m:
                        return
                    vecs = [e_m[mi] for mi in ms]
                    t = vecs[0]
                    for v in vecs[1:]:
                        t = np.tensordot(t, v, axes=0)
                    tensor[:] += t
                else:
                    for mi in [-1, 0, 1]:
                        build(ms + (mi,), total + mi)

            build()
            norm = np.linalg.norm(tensor)
            basis.append(tensor / norm if norm > 0 else tensor)
        return basis

    def _create_combined_wave_function(self):
        result = []

        # Compute the full weighted tensor from all polarizations
        full_tensor = sum(
            coeff * np.exp(-1j * phase) * basis
            for coeff, phase, basis in zip(
                self.polarization_coefficients,
                self.polarization_phases,
                self.polarization_bases)
        )

        for idx in self.index_combinations:
            value = full_tensor[idx]
            new_wf = copy.deepcopy(self.wave_blueprint)

            new_wf.psi = self.wave_blueprint.psi * value * self.wave_blueprint.packet_creator.momentum_propagator
            new_wf.multiplicity = self.index_multiplicities[idx]
            new_wf.psi *= 1 / np.sqrt(new_wf.multiplicity)

            result.append(new_wf)

        return result


from itertools import permutations

def get_index_multiplicities(index_combinations):
    multiplicities = {}
    for idx in index_combinations:
        perms = set(permutations(idx))
        multiplicities[idx] = len(perms)
    return multiplicities
