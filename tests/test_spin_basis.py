"""Test: the polarization tensors of Wave_vector_class.generate_spin_basis are the textbook spin-s states.

The spin-s basis is built by repeated Clebsch-Gordan coupling of spin-1 vectors. This test checks the result against
the definition of an irreducible spin-s representation on rank-s Cartesian tensors, with the rotation generators
(S_a)_ij = -i eps_aij acting on every index.

Checks, for s = 0 ... MAX_SPIN
    1) orthonormality: <T_m | T_m'> = delta_mm' (Frobenius inner product),
    2) every tensor is symmetric under index permutations and traceless (contraction of any two indices vanishes),
    3) J_z T_m = m T_m,
    4) J^2 T_m = s (s + 1) T_m,
    5) J_+ T_m = sqrt(s (s + 1) - m (m + 1)) T_(m+1) (Condon-Shortley phases; a wrong sign here does not change any
       gravity-only result, it only matters when the polarizations are compared with other work)."""

from itertools import permutations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from resources.Classes.Wave_vector_class import Wave_vector_class


MAX_SPIN = 4
TOLERANCE = 1.0e-12


def spin_one_generators():
    """(S_a)_ij = -i eps_aij for a = x, y, z."""
    eps = np.zeros((3, 3, 3))
    for (i, j, k), sign in {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.items():
        eps[i, j, k] = sign
    return [-1j * eps[a] for a in range(3)]


def apply_generator(S, tensor):
    """Sum over the indices of the rank-s tensor of S acting on that index."""
    result = np.zeros_like(tensor)
    for axis in range(tensor.ndim):
        result += np.moveaxis(np.tensordot(S, tensor, axes=([1], [axis])), 0, axis)
    return result


def main():
    Sx, Sy, Sz = spin_one_generators()
    Splus = Sx + 1j * Sy
    worst = {"orthonormal": 0.0, "symmetric": 0.0, "traceless": 0.0, "Jz": 0.0, "J2": 0.0, "Jplus": 0.0}

    for s in range(MAX_SPIN + 1):
        basis = Wave_vector_class.generate_spin_basis(s)
        assert len(basis) == 2 * s + 1, f"spin {s}: expected {2 * s + 1} tensors, got {len(basis)}"
        ms = list(range(-s, s + 1))

        gram = np.array([[np.vdot(a, b) for b in basis] for a in basis])
        worst["orthonormal"] = max(worst["orthonormal"], np.abs(gram - np.eye(2 * s + 1)).max())

        for m, T in zip(ms, basis):
            if s >= 2:
                for perm in permutations(range(s)):
                    worst["symmetric"] = max(worst["symmetric"], np.abs(np.transpose(T, perm) - T).max())
                worst["traceless"] = max(worst["traceless"], np.abs(np.trace(T, axis1=0, axis2=1)).max())

            if s == 0:
                continue
            JzT = apply_generator(Sz, T)
            worst["Jz"] = max(worst["Jz"], np.abs(JzT - m * T).max())
            J2T = sum(apply_generator(S, apply_generator(S, T)) for S in (Sx, Sy, Sz))
            worst["J2"] = max(worst["J2"], np.abs(J2T - s * (s + 1) * T).max())
            JpT = apply_generator(Splus, T)
            expected = np.sqrt(s * (s + 1) - m * (m + 1)) * basis[ms.index(m + 1)] if m < s else np.zeros_like(T)
            worst["Jplus"] = max(worst["Jplus"], np.abs(JpT - expected).max())

        print(f"spin {s}: {2 * s + 1} tensors of shape {basis[0].shape}, checked")

    for name, value in worst.items():
        print(f"largest deviation, {name:11s}: {value:.2e}")
        assert value < TOLERANCE, f"spin basis check '{name}' failed (deviation {value:.2e} > {TOLERANCE:.0e})"
    print(f"PASS: spin basis is orthonormal, symmetric, traceless and has J_z, J^2, J_+ of a spin-s multiplet up to s = {MAX_SPIN}")


if __name__ == "__main__":
    main()
