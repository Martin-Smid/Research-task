# Spin tests

Three plain Python scripts that check the spin (vector / tensor) part of the solver. Each prints `PASS` or raises
an `AssertionError`. They need the same CUDA/CuPy environment as the simulation. Gravity only, no baryons.

| script | what it checks |
|---|---|
| `test_spin_basis.py` | The polarization tensors of `Wave_vector_class.generate_spin_basis` are the textbook spin-s states (s = 0..4): orthonormal, symmetric, traceless, eigenstates of J_z and J^2, J_+ with Condon-Shortley phases. Pure NumPy, no GPU. |
| `test_spin_rotation_invariance.py` | Mixing the 2s+1 polarization components with a unitary matrix leaves the evolution unchanged (rho and psi' = U psi), spin 1 and 2, 100 steps. A non-mixed control proves the test can tell coupled from uncoupled. |
| `test_spin_scalar_reduction.py` | A spin-s vector whose components share one spatial profile evolves exactly like the scalar field (spin 1, 2, 3; random amplitudes and a single polarization). |

## How to run

From the repository root, with the project venv, one script at a time:

```
python tests/spin/test_spin_basis.py
python tests/spin/test_spin_rotation_invariance.py
python tests/spin/test_spin_scalar_reduction.py
```

(Windows: `.\venv\Scripts\python.exe tests\spin\test_spin_basis.py`, and so on.)
`test_spin_scalar_reduction.py` imports helpers from `test_spin_rotation_invariance.py`, so keep them in the same folder.
They are not part of `tests/run_all.py`. Runtime: seconds on a laptop GPU. The solver prints many debug lines
(`passed mass is ...`); the lines that matter start with `spin`, `largest deviation` or `PASS`.

## Reference result (RTX 3070 Laptop, 2026-10-30, commit 7853af1 plus the move to this folder)

All three print PASS; every error is at rounding level (about 1e-14 or smaller).

| script | printed numbers |
|---|---|
| `test_spin_basis.py` | largest deviation: orthonormal 6.66e-16, symmetric 1.11e-16, traceless 2.22e-16, Jz 4.16e-17, J2 1.78e-15, Jplus 8.88e-16 |
| `test_spin_rotation_invariance.py` | spin 1: rho error 3.88e-15, psi error 1.29e-14, rho changed by 29.72, uncoupled control differs by 0.98. Spin 2: rho error 2.88e-15, psi error 9.46e-15, rho changed by 19.81, control differs by 0.96 |
| `test_spin_scalar_reduction.py` | random amplitudes: spin 1 rho 1.34e-14 / psi_m 3.26e-14 / mass 0; spin 2 rho 7.79e-15 / psi_m 1.75e-14 / mass 1.86e-16; spin 3 rho 7.40e-15 / psi_m 1.46e-14 / mass 0. Single polarization m = 0: all errors exactly 0 for spin 1, 2, 3 |

On another GPU (e.g. the Koios P100) the numbers can differ in the last digits; PASS with errors around 1e-10 or
smaller is the expected outcome. A failure means a real change in the spin sector (or the test), so report it
instead of changing tolerances.

## Note

The scripts originally lived in `tests/`. The only change made when moving them here is the project-root line
`Path(__file__).resolve().parents[1]` -> `parents[2]` (one folder deeper).
