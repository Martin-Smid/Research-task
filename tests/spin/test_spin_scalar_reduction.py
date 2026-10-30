"""Test: a spin-s wave vector whose components share one spatial profile evolves exactly like the scalar field.

If psi_m = a_m psi_0 with sum |a_m|^2 = 1, the total density is |psi_0|^2 at all times (the gravity only sees rho), so
every component must stay a_m psi_0(t) with psi_0(t) the scalar (spin 0) result. This checks the normalization of the
polarization amplitudes, the multiplicity handling and the density sum in the evolution.

Checks, for spin 1, 2 and 3
    1) random amplitudes a_m (the defaults of Wave_vector_class, seeded): rho equals the scalar rho and psi_m = a_m psi_0
       (MAX_ERROR),
    2) a single polarization (a = delta_(m,0)) equals the scalar run,
    3) the total mass of the vector equals the scalar mass (the amplitudes are normalized to 1)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from test_spin_rotation_invariance import build_sim, random_fields, run


SPINS = (1, 2, 3)
MAX_ERROR = 1.0e-10


def main():
    psi0 = random_fields(1, seed=7)[0]
    scalar_sim, _ = build_sim(0)
    scalar_psi, scalar_rho = run(scalar_sim, [psi0])
    scalar_mass = float(scalar_rho.sum())

    for spin in SPINS:
        size = 2 * spin + 1
        for label, kwargs in (
            ("random amplitudes", dict(random_seed=100 + spin)),
            ("single polarization m = 0", dict(polarization_coefficients=[1.0 if m == 0 else 0.0 for m in range(-spin, spin + 1)],
                                               polarization_phases=[0.0] * size)),
        ):
            sim, vector = build_sim(spin, **kwargs)
            amplitudes = vector.polarization_coefficients * np.exp(-1j * vector.polarization_phases)
            psi, rho = run(sim, [a * psi0 for a in amplitudes])

            rho_error = np.abs(rho - scalar_rho).max() / scalar_rho.max()
            psi_error = max(np.abs(psi[m] - amplitudes[m] * scalar_psi[0]).max() for m in range(size)) / np.sqrt(scalar_rho.max())
            mass_error = abs(float(rho.sum()) - scalar_mass) / scalar_mass
            print(f"spin {spin}, {label}: rho error {rho_error:.2e}, psi_m error {psi_error:.2e}, mass error {mass_error:.2e}")
            assert rho_error < MAX_ERROR, f"spin {spin} ({label}): rho differs from the scalar run ({rho_error:.2e})"
            assert psi_error < MAX_ERROR, f"spin {spin} ({label}): psi_m != a_m psi_0 ({psi_error:.2e})"
            assert mass_error < MAX_ERROR, f"spin {spin} ({label}): total mass differs from the scalar run ({mass_error:.2e})"
    print("PASS: spin-s vectors with a common spatial profile reduce exactly to the scalar field")


if __name__ == "__main__":
    main()
