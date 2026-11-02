"""Test: Wave_vector_class(independent_fields=True) gives every polarization its own Gaussian random spatial field.

The default path builds psi_m = a_m psi_0 with ONE shared profile psi_0 (the density statistics are those of the scalar
field). The opt-in path builds psi_m = a_m g_m with independent Gaussian random fields g_m of the scalar mass M, so
a_m^2 are the mass fractions (equal by default, a_m = 1 / sqrt(n)). For the fractions f_m = a_m^2
    variance(rho / rho_mean) = sum_m f_m^2   (= 1 / n for equal fractions, where rho / rho_mean ~ Gamma(n, 1/n)).

Checks
    1) the code's own fields, s = 1, 2, 3, equal fractions, at t = 0 and after free evolution: variance * n = 1,
       skewness * sqrt(n) / 2 = 1, small KS distance to Gamma(n, 1/n) (same tolerances as test_spin_granule_statistics.py),
    2) unequal fractions: the measured variance equals sum f_m^2 (this also checks the per-component normalization),
    3) the mass of every component is a_m^2 M and the total mass equals the scalar mass M,
    4) the same seed gives bitwise identical fields, a different seed a different field, and two components are
       uncorrelated (cross-correlation below 5 / sqrt(number of cells), the statistical noise level of independent fields),
    5) the momentum factor is applied exactly once: for zero momentum the factor is exactly 1, with a momentum p the fields
       are the zero-momentum ones times the factor, and the mean wave number moves by p / h_bar_tilde; power_spectrum
       is exercised here (a band-limited spectrum keeps the shift free of Nyquist wrap-around, and has no power above the cut),
    6) the default path is unchanged: all components share the profile of the blueprint (rho = |psi_0|^2 sum a_m^2).
(packet_type="gaussian" is not used: it fails already in the default path, with a TypeError in Wave_function.)"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from resources.Classes.Wave_vector_class import Wave_vector_class
from test_spin_granule_statistics import MAX_KS, MAX_SKEWNESS_ERROR, MAX_VARIANCE_ERROR, N_GRID, statistics
from test_spin_rotation_invariance import build_sim, run


SPINS = (1, 2, 3)
UNEQUAL = [1.0, 2.0, 3.0]           # spin 1 coefficients: mass fractions 1/14, 4/14, 9/14
N_MOMENTUM = 32                     # small grid for the momentum check
K_CUT_MODES = 4                     # the spectrum of the momentum check holds modes |k| <= 4 fundamentals
SHIFT_MODES = 2                     # the momentum is 2 fundamentals of the box
MAX_SHIFT_ERROR = 1.0e-5            # relative; the momentum factor is stored in single precision


def fields_of(vector):
    return [np.asarray(w.psi) for w in vector.wave_vector]


def blueprint_mass(sim, vector):
    return np.sum(np.abs(vector.wave_blueprint.psi) ** 2) * np.prod(sim.dx)


def make_vector(sim, spin, **kwargs):
    return Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat",
        means=[0, 0, 0],
        st_deviations=[0.5, 0.5, 0.5],
        simulation=sim,
        mass=1,
        omega=1,
        spin=spin,
        desired_soliton_mass=5.0e7,
        **kwargs,

    )


def mean_kx(fields, sim):
    kx = 2.0 * np.pi * np.fft.fftfreq(fields[0].shape[0], d=sim.dx[0])[:, None, None]
    power = sum(np.abs(np.fft.fftn(f)) ** 2 for f in fields)
    return np.sum(kx * power) / np.sum(power)


def main():
    for spin in SPINS:
        n = 2 * spin + 1
        sim, vector = build_sim(spin, n_grid=N_GRID, use_gravity=False, independent_fields=True, random_seed=100 + spin)
        fields = fields_of(vector)
        rho_start = sum(np.abs(f) ** 2 for f in fields)
        _, rho_end = run(sim, fields)

        print(f"spin {spin} (n = {n}), code's own fields:  variance*n, skewness*sqrt(n)/2, KS")
        for label, rho in (("t = 0  ", rho_start), ("t = end", rho_end)):
            variance, skewness, ks = statistics(rho, n)
            print(f"    {label}  {variance:.4f}  {skewness:.4f}  {ks:.4f}")
            assert abs(variance - 1) < MAX_VARIANCE_ERROR, f"spin {spin}, {label}: variance*n = {variance:.4f}"
            assert abs(skewness - 1) < MAX_SKEWNESS_ERROR, f"spin {spin}, {label}: skewness*sqrt(n)/2 = {skewness:.4f}"
            assert ks < MAX_KS, f"spin {spin}, {label}: KS distance {ks:.4f}"

        # mass: a_m^2 M per component, M in total
        sim, vector = build_sim(spin, n_grid=N_GRID, use_gravity=False, independent_fields=True, random_seed=100 + spin)
        mass = blueprint_mass(sim, vector)
        component_masses = np.array([np.sum(np.abs(f) ** 2) * np.prod(sim.dx) for f in fields_of(vector)])
        fraction_error = np.abs(component_masses / mass - vector.polarization_coefficients ** 2).max()
        total_error = abs(component_masses.sum() / mass - 1)
        print(f"    mass: fraction error {fraction_error:.2e}, total mass error {total_error:.2e}")
        assert fraction_error < 1.0e-12 and total_error < 1.0e-12, f"spin {spin}: the component masses are not a_m^2 M"

    # unequal mass fractions
    sim, vector = build_sim(1, n_grid=N_GRID, use_gravity=False, independent_fields=True, random_seed=7,
                            polarization_coefficients=UNEQUAL)
    fractions = np.array(UNEQUAL) ** 2 / np.sum(np.array(UNEQUAL) ** 2)
    rho = sum(np.abs(f) ** 2 for f in fields_of(vector))
    variance = np.var(rho / rho.mean())
    masses = np.array([np.sum(np.abs(f) ** 2) for f in fields_of(vector)])
    print(f"unequal fractions {np.round(fractions, 4)}: variance {variance:.4f}, sum f^2 {np.sum(fractions ** 2):.4f}, "
          f"measured fractions {np.round(masses / masses.sum(), 4)}")
    assert abs(variance / np.sum(fractions ** 2) - 1) < MAX_VARIANCE_ERROR, "unequal fractions: variance != sum f_m^2"
    assert np.abs(masses / masses.sum() - fractions).max() < 1.0e-12, "unequal fractions: wrong component masses"

    # seeds and independence of the components
    sim, _ = build_sim(1, n_grid=N_GRID, use_gravity=False)
    a = fields_of(make_vector(sim, 1, independent_fields=True, random_seed=11))
    b = fields_of(make_vector(sim, 1, independent_fields=True, random_seed=11))
    c = fields_of(make_vector(sim, 1, independent_fields=True, random_seed=12))
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "the same seed gives different fields"
    assert not np.allclose(a[0], c[0]), "different seeds give the same field"
    cross = max(abs(np.mean(np.conj(a[i]) * a[j])) / np.sqrt(np.mean(np.abs(a[i]) ** 2) * np.mean(np.abs(a[j]) ** 2))
                for i in range(3) for j in range(i + 1, 3))
    noise = 5.0 / np.sqrt(a[0].size)
    print(f"seeds: same seed bitwise identical, different seed differs; largest cross-correlation {cross:.2e} "
          f"(noise level {noise:.2e})")
    assert cross < noise, f"the components are correlated ({cross:.2e})"

    # momentum factor, applied exactly once
    sim, _ = build_sim(1, n_grid=N_MOMENTUM, use_gravity=False)
    length = N_MOMENTUM * sim.dx[0]
    k_fund = 2.0 * np.pi / length
    spectrum = lambda k: (k <= K_CUT_MODES * k_fund + 1.0e-9).astype(float)
    still = make_vector(sim, 1, momenta=[0, 0, 0], independent_fields=True, random_seed=3, power_spectrum=spectrum)
    moving = make_vector(sim, 1, momenta=[SHIFT_MODES * k_fund * sim.h_bar_tilde, 0, 0], independent_fields=True,
                         random_seed=3, power_spectrum=spectrum)
    factor = moving.wave_blueprint.packet_creator.momentum_propagator
    assert np.all(still.wave_blueprint.packet_creator.momentum_propagator == 1), "the zero-momentum factor is not 1"
    once = max(np.abs(m / factor - s).max() / np.abs(s).max() for m, s in zip(fields_of(moving), fields_of(still)))
    shift = mean_kx(fields_of(moving), sim) - mean_kx(fields_of(still), sim)
    above_cut = np.sum(np.abs(np.fft.fftn(fields_of(still)[0])[K_CUT_MODES + 1:N_MOMENTUM - K_CUT_MODES]) ** 2) / \
        np.sum(np.abs(np.fft.fftn(fields_of(still)[0])) ** 2)
    print(f"momentum: moving / factor vs still {once:.2e}, mean k_x shift {shift / k_fund:.6f} fundamentals "
          f"(expected {SHIFT_MODES}), power above the spectrum cut {above_cut:.1e}")
    assert once < 1.0e-12, f"the momentum factor is not applied exactly once ({once:.2e})"
    assert abs(shift / (SHIFT_MODES * k_fund) - 1) < MAX_SHIFT_ERROR, f"the mean k_x moved by {shift / k_fund:.6f}, not {SHIFT_MODES}"
    assert above_cut < 1.0e-20, f"power_spectrum is not respected (power above the cut {above_cut:.1e})"

    # the default path is unchanged: one shared profile
    sim, vector = build_sim(2, n_grid=N_GRID, use_gravity=False, random_seed=5)
    rho = sum(np.abs(f) ** 2 for f in fields_of(vector))
    shared = np.abs(vector.wave_blueprint.psi) ** 2 * np.sum(vector.polarization_coefficients ** 2)
    default_error = np.abs(rho - shared).max() / shared.max()
    print(f"default path: rho equals the shared profile |psi_0|^2 (error {default_error:.2e})")
    assert default_error < 1.0e-12, "the default path no longer uses one shared spatial profile"
    print("PASS: independent_fields=True gives independent Gaussian polarization fields with exact mass fractions; "
          "seeds, momentum and the default path behave as specified")


if __name__ == "__main__":
    main()
