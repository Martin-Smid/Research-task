"""Test: with gravity only, mixing the polarization components with a unitary matrix changes nothing physical.

The 2s+1 polarization fields psi_m evolve independently except through the total density rho = sum_m |psi_m|^2.
A unitary matrix U in polarization space (a change of the polarization basis) leaves rho invariant, so evolving
psi'_m = sum_n U_mn psi_n must give psi'_m(t) = sum_n U_mn psi_n(t) and the same rho(t). A bug in the density sum, in the
per-component normalization or in the evolution of the extra components would break this.

Set-up
    spin s = 1 and 2, a small periodic box, random smooth complex fields with a DIFFERENT spatial profile in every
    component (so that the mixing really matters), self-gravity, 100 steps of the second order scheme.

Checks
    1) rho of the mixed run equals rho of the original run, and psi' = U psi componentwise (MAX_ERROR, relative to max rho
       or max |psi|, both are rounding-level quantities because U is applied before the run and the scheme is linear in psi
       apart from the density),
    2) the fields really evolved (the density changed by more than MIN_EVOLUTION), so the test cannot pass trivially,
    3) a NON-mixed control: evolving the components as independent scalar fields (each with its own gravity) gives a clearly
       different density, so the test can tell the coupling through rho from no coupling."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 9.6                           # the box is [-BOX, BOX]^3 [kpc]
N_GRID = 32
H = 1.0e-3                          # [Gyr], as in the other small tests
NUM_STEPS = 100
RHO_MEAN = 5.0e6                    # mean total density [Msun/kpc^3]; free-fall time ~ 0.2 Gyr, so 0.1 Gyr is nonlinear enough
K_MAX = 3                           # random fields contain the modes |k_i| <= K_MAX (in units of the box fundamental)
SPINS = (1, 2)
MAX_ERROR = 1.0e-10
MIN_EVOLUTION = 1.0e-2              # relative change of rho during the run
MIN_CONTROL_DIFFERENCE = 1.0e-2     # relative density difference of the uncoupled control


def build_sim(spin, **vector_kwargs):
    """Small self-gravitating ULDM simulation with one Wave_vector of the given spin."""
    sim = Simulation_Class(

        dim=3,
        boundaries=[(-BOX, BOX)] * 3,
        N=N_GRID,
        total_time=NUM_STEPS * H,
        h=H,
        order_of_evolution=2,
        use_gravity=True,
        use_units=True,
        use_sponge=False,
        self_int=False,
        m_s=2.5e-22,

    )
    vector = Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat",    # placeholder, psi is replaced by the caller
        means=[0, 0, 0],
        st_deviations=[0.5, 0.5, 0.5],
        simulation=sim,
        mass=1,
        omega=1,
        momenta=[0, 0.0, 0],
        spin=spin,
        desired_soliton_mass=5.0e7,
        **vector_kwargs,

    )
    sim.add_wave_vector(vector)
    with redirect_stdout(io.StringIO()):
        sim.initialize_simulation()
    return sim, vector


def random_fields(count, seed):
    """`count` smooth complex random fields, scaled so that the sum of |psi|^2 has the mean RHO_MEAN."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID)
    mask = (np.abs(k)[:, None, None] <= K_MAX) & (np.abs(k)[None, :, None] <= K_MAX) & (np.abs(k)[None, None, :] <= K_MAX)
    fields = []
    for _ in range(count):
        spectrum = (rng.normal(size=(N_GRID,) * 3) + 1j * rng.normal(size=(N_GRID,) * 3)) * mask
        fields.append(np.fft.ifftn(spectrum))
    mean_density = sum(np.mean(np.abs(f) ** 2) for f in fields)
    return [f * np.sqrt(RHO_MEAN / mean_density) for f in fields]


def run(sim, fields):
    """Evolve the given component fields (list of numpy arrays); return the final fields (numpy) and density."""
    for wave, field in zip(sim.wave_functions, fields):
        wave.psi = cp.asarray(field, dtype=cp.complex128)
    with redirect_stdout(io.StringIO()):
        final = sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)
    shutil.rmtree(sim.evolution.scribe.snapshot_directory, ignore_errors=True)       # the test leaves no output behind
    psi = [cp.asnumpy(wave.psi) for wave in final]
    return psi, sum(np.abs(p) ** 2 for p in psi)


def random_unitary(size, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def main():
    for spin in SPINS:
        size = 2 * spin + 1
        fields = random_fields(size, seed=10 + spin)
        U = random_unitary(size, seed=20 + spin)
        mixed = [sum(U[m, n] * fields[n] for n in range(size)) for m in range(size)]

        psi_a, rho_a = run(build_sim(spin)[0], fields)
        psi_b, rho_b = run(build_sim(spin)[0], mixed)
        psi_a_mixed = [sum(U[m, n] * psi_a[n] for n in range(size)) for m in range(size)]

        rho_error = np.abs(rho_b - rho_a).max() / rho_a.max()
        psi_error = max(np.abs(psi_b[m] - psi_a_mixed[m]).max() for m in range(size)) / np.sqrt(rho_a.max())
        rho_start = sum(np.abs(f) ** 2 for f in fields)
        evolution = np.abs(rho_a - rho_start).max() / rho_start.max()

        # control: every component alone (own gravity), summed afterwards
        rho_control = 0.0
        for m in range(size):
            sim_single, _ = build_sim(spin, polarization_coefficients=[1.0 if i == m else 0.0 for i in range(size)])
            single = [fields[m] if i == m else np.zeros_like(fields[m]) for i in range(size)]
            rho_control = rho_control + run(sim_single, single)[1]
        control_difference = np.abs(rho_control - rho_a).max() / rho_a.max()

        print(f"spin {spin}: rho error {rho_error:.2e}, psi error {psi_error:.2e}, "
              f"rho changed by {evolution:.2f} during the run, uncoupled control differs by {control_difference:.2f}")
        assert evolution > MIN_EVOLUTION, f"spin {spin}: the density did not evolve ({evolution:.2e}), the test is trivial"
        assert control_difference > MIN_CONTROL_DIFFERENCE, f"spin {spin}: the uncoupled control is not distinguishable ({control_difference:.2e})"
        assert rho_error < MAX_ERROR, f"spin {spin}: rho changes under a unitary mixing of the polarizations ({rho_error:.2e})"
        assert psi_error < MAX_ERROR, f"spin {spin}: psi' != U psi after the run ({psi_error:.2e})"
    print("PASS: a unitary change of polarization basis leaves the evolution unchanged (rho and psi' = U psi)")


if __name__ == "__main__":
    main()
