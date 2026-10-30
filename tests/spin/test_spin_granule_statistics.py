"""Test: the density granule statistics of a spin-s field with independent polarization components.

If a spin-s field has n = 2s+1 independent, equal-variance, complex Gaussian polarization components, every |psi_m|^2 is
exponentially distributed and rho = sum_m |psi_m|^2 follows a gamma distribution with shape n and mean rho_mean. With
delta = rho / rho_mean - 1
    variance(delta) = 1 / n,    skewness(delta) = 2 / sqrt(n),    rho / rho_mean ~ Gamma(shape n, scale 1/n).
Free (gravity off) Schroedinger evolution keeps a Gaussian random field Gaussian with the same power spectrum, so these
numbers must not change in time.

Set-up
    N = 64, s = 0 ... 3 (n = 1, 3, 5, 7), n independent white complex Gaussian fields (all k up to Nyquist, fixed seeds),
    scaled to RHO_MEAN, gravity off, NUM_STEPS steps of the second order scheme.

Checks, at t = 0 and after the evolution
    1) variance(delta) * n = 1 and skewness(delta) * sqrt(n) / 2 = 1,
    2) Kolmogorov-Smirnov distance of rho / rho_mean to the Gamma(n, 1/n) CDF.
Tolerances: white noise has uncorrelated cells and free evolution keeps it white, so the N^3 = 262144 cells are independent
samples. The sampling error of the variance is about sqrt((kurtosis - 1) / N^3), at most 0.6 % (n = 1); the skewness is a
third moment and has a larger error (about 2 % for n = 1), and the KS distance of independent samples is about 1 / sqrt(N^3)
= 0.002. The tolerances (3 % variance, 5 % skewness, 0.01 KS) are a few times these errors and were fixed before running.

The code's own polarization amplitudes: Wave_vector_class builds psi_m = a_m psi_0 with ONE spatial profile psi_0 shared by
all components. The last block of the printout shows what that gives: variance * n = n instead of 1 (the same statistics as
the scalar field, whatever the spin). It is printed as a finding and not asserted."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import stats

from test_spin_rotation_invariance import BOX, RHO_MEAN, build_sim, run


N_GRID = 64
SPINS = (0, 1, 2, 3)
MAX_VARIANCE_ERROR = 0.03
MAX_SKEWNESS_ERROR = 0.05
MAX_KS = 0.01


def white_fields(count, seed):
    """`count` independent white complex Gaussian fields, scaled so that the sum of |psi|^2 has the mean RHO_MEAN."""
    rng = np.random.default_rng(seed)
    fields = [rng.normal(size=(N_GRID,) * 3) + 1j * rng.normal(size=(N_GRID,) * 3) for _ in range(count)]
    mean_density = sum(np.mean(np.abs(f) ** 2) for f in fields)
    return [f * np.sqrt(RHO_MEAN / mean_density) for f in fields]


def statistics(rho, n):
    """variance * n, skewness * sqrt(n) / 2 (both 1 for the gamma law) and the KS distance of rho / rho_mean."""
    x = rho / rho.mean()
    delta = x - 1
    variance = np.mean(delta ** 2)
    skewness = np.mean(delta ** 3) / variance ** 1.5
    ks = stats.kstest(x.ravel(), stats.gamma(a=n, scale=1.0 / n).cdf).statistic
    return variance * n, skewness * np.sqrt(n) / 2, ks


def main():
    for spin in SPINS:
        n = 2 * spin + 1
        fields = white_fields(n, seed=30 + spin)
        rho_start = sum(np.abs(f) ** 2 for f in fields)
        _, rho_end = run(build_sim(spin, n_grid=N_GRID, use_gravity=False)[0], fields)

        print(f"spin {spin} (n = {n}):  variance*n, skewness*sqrt(n)/2, KS")
        for label, rho in (("t = 0  ", rho_start), ("t = end", rho_end)):
            variance, skewness, ks = statistics(rho, n)
            print(f"    {label}  {variance:.4f}  {skewness:.4f}  {ks:.4f}")
            assert abs(variance - 1) < MAX_VARIANCE_ERROR, f"spin {spin}, {label}: variance*n = {variance:.4f}"
            assert abs(skewness - 1) < MAX_SKEWNESS_ERROR, f"spin {spin}, {label}: skewness*sqrt(n)/2 = {skewness:.4f}"
            assert ks < MAX_KS, f"spin {spin}, {label}: KS distance {ks:.4f}"
        assert abs(rho_end.mean() / rho_start.mean() - 1) < 1.0e-10, f"spin {spin}: the mean density changed"

    print("code's own amplitudes (one shared spatial profile psi_0, psi_m = a_m psi_0), t = 0:  variance*n")
    for spin in SPINS[1:]:
        n = 2 * spin + 1
        a = build_sim(spin, n_grid=N_GRID, use_gravity=False, random_seed=5)[1].polarization_coefficients
        psi_0 = white_fields(1, seed=40 + spin)[0]
        rho = sum(np.abs(a_m * psi_0) ** 2 for a_m in a)
        print(f"    spin {spin}: {statistics(rho, n)[0]:.4f}  (independent components would give 1)")
    print("PASS: the density of n = 2s+1 independent Gaussian polarizations is Gamma(n, 1/n) and stays so under free evolution")


if __name__ == "__main__":
    main()
