"""Test: merging of sinks (SinkNBody.merge_close_sinks and the sim.sink_merging flag).

Four checks
    1) A pair across the periodic box edge merges at the edge (minimum-image centre of mass, not the middle of the box);
       mass and momentum are conserved.
    2) The bound check uses the SOFTENED potential: a pair deep inside the softening length with a relative speed
       between the softened and the point-mass escape speeds does not merge (the point-mass check would merge it);
       a slower pair does.
    3) Energy bookkeeping: E_total + E_diss as logged by Diagnostics does not jump at a merge (the energy lost in the
       merge, kinetic + softened potential, is added to E_diss_merge_total).
    4) The flag: with sim.sink_merging = True Evolution merges two sinks that fall onto each other, with the flag off
       it does not."""

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np
from scipy.special import erf

from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # the box is [-BOX, BOX]^3 [kpc]
N_GRID = 64                         # grid resolution (cell 0.3125 kpc, softening 1.5 cells)
MAX_ENERGY_JUMP = 0.1               # |jump of E_total + E_diss at a merge| / (energy lost in the merge)
MAX_MOMENTUM_ERROR = 1.0e-12        # relative error of the total momentum
MAX_POSITION_ERROR = 1.0e-9         # error of the merged position [kpc]


output_dirs = []                    # output folders of the simulations built below, removed at the end


def build(masses, positions, velocities, total_time=0.01, h=1.0e-3):
    """Simulation with the given sinks and no wave function; returns (sim, sink)."""

    sim = Simulation_Class(

        dim=3,
        boundaries=[(-BOX, BOX)] * 3,
        N=N_GRID,
        total_time=total_time,
        h=h,
        order_of_evolution=2,
        use_gravity=True,
        use_units=True,
        use_sponge=False,
        self_int=False,

    )
    sink = SinkNBody(

        simulation=sim,
        N_sinks=len(masses),
        initial_masses=masses,
        initial_positions=positions,
        initial_velocities=velocities,

    )
    sim.add_baryons(sink)
    sim.initialize_simulation()
    return sim, sink


def momentum(sink):
    return cp.asnumpy(sink.masses[:, None] * sink.velocities).sum(axis=0)


def check_periodic_merge():
    """1) pair across the box edge: 3e10 at x = 9.9 and 1e10 at x = -9.9 (minimum-image distance 0.2 kpc)."""
    sim, sink = build([3.0e10, 1.0e10], [[9.9, 0.0, 0.0], [-9.9, 0.0, 0.0]],
                      [[1.0, 2.0, 3.0], [-5.0, 0.0, 1.0]])
    p0 = momentum(sink)
    n_merged, dE = sink.merge_close_sinks(r_merge=1.0)
    assert n_merged == 1 and sink.N == 1, f"the pair across the box edge did not merge (merged {n_merged}, N = {sink.N})"
    position = cp.asnumpy(sink.positions)[0]
    expected = np.array([9.9 + 0.25 * 0.2, 0.0, 0.0])           # the heavier sink moves by m2 / M of the 0.2 kpc gap
    position_error = np.abs(position - expected).max()
    momentum_error = np.abs(momentum(sink) - p0).max() / np.abs(p0).max()
    mass_error = abs(float(sink.masses.sum()) / 4.0e10 - 1.0)
    print(f"1) periodic merge: position {position[0]:.6f} (expected {expected[0]:.6f}), error {position_error:.1e} kpc, "
          f"momentum error {momentum_error:.1e}, mass error {mass_error:.1e}")
    assert position_error < MAX_POSITION_ERROR, f"merged position {position} is not at the box edge {expected}"
    assert momentum_error < MAX_MOMENTUM_ERROR, f"the total momentum changed by {momentum_error:.1e}"
    assert mass_error < 1.0e-14, "the mass is not conserved"


def check_softened_bound():
    """2) pair at 0.1 eps. v_rel^2 = 5 G M / eps lies between the softened (1.1) and the point-mass (20) escape values."""
    mass = 5.0e10
    sim, sink = build([mass, mass], [[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    G, eps, r = float(sim.G), float(sink.softening_bh), 0.04
    vesc2_soft = 2.0 * G * 2.0 * mass * erf(r / (np.sqrt(2.0) * eps)) / r
    vesc2_point = 2.0 * G * 2.0 * mass / r
    v_fast = np.sqrt(2.0 * vesc2_soft)
    assert vesc2_soft < v_fast ** 2 < vesc2_point, "test set-up: the speeds are not between the two escape speeds"
    sink.velocities = cp.asarray([[0.0, -v_fast / 2, 0.0], [0.0, v_fast / 2, 0.0]], dtype=cp.float64)
    n_fast, _ = sink.merge_close_sinks(r_merge=1.0)

    v_slow = np.sqrt(0.5 * vesc2_soft)
    sim, sink = build([mass, mass], [[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]], [[0.0, -v_slow / 2, 0.0], [0.0, v_slow / 2, 0.0]])
    n_slow, _ = sink.merge_close_sinks(r_merge=1.0)
    print(f"2) softened bound check: eps = {eps:.3f} kpc, v_rel^2 / vesc_soft^2 = 2 -> merged {n_fast}; "
          f"0.5 -> merged {n_slow} (point-mass vesc^2 is {vesc2_point / vesc2_soft:.0f}x the softened one)")
    assert n_fast == 0, "a pair that is unbound in the softened potential was merged"
    assert n_slow == 1, "a pair that is bound in the softened potential was not merged"


def logged_conserved_energy(sim):
    """E_total + E_diss exactly as Diagnostics / Scribe log it."""
    evolution = sim.evolution
    values = evolution.diagnostics.compute_energy(sim.wave_functions, evolution._compute_total_density(sim.wave_functions), 0.0)
    return values["K_total"] + values["W"] + values["U_iso"] + values["E_diss"], values["E_diss"]


def check_energy_bookkeeping():
    """3) E_total + E_diss does not jump at a merge (off-centre positions, a bound pair 0.9 kpc apart)."""
    mass = 5.0e10
    sim, sink = build([mass, 0.6 * mass], [[-0.37, 0.21, 0.05], [0.52, -0.10, 0.02]],
                      [[0.0, -30.0, 5.0], [10.0, 20.0, -5.0]])
    E_before, diss_before = logged_conserved_energy(sim)
    n_merged, dE = sink.merge_close_sinks(r_merge=1.5)
    assert n_merged == 1, "the pair did not merge"
    E_after, diss_after = logged_conserved_energy(sim)
    jump = abs(E_after - E_before) / dE
    print(f"3) energy bookkeeping: energy lost in the merge {dE:.4e}, E_diss {diss_before:.3e} -> {diss_after:.3e}, "
          f"jump of E_total + E_diss {E_after - E_before:+.3e} ({jump:.1%} of the loss)")
    assert dE > 0.0 and abs(diss_after - diss_before - dE) < 1.0e-6 * dE, "E_diss does not contain the merge energy"
    assert jump < MAX_ENERGY_JUMP, f"E_total + E_diss jumps by {jump:.1%} of the energy lost in the merge"


def check_flag():
    """4) two sinks of 5e10 Msun fall onto each other from 4 kpc (fall time ~0.015 Gyr); only the flag merges them."""
    result = {}
    for flag in (True, False):
        sim, sink = build([5.0e10, 5.0e10], [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                          [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], total_time=0.03, h=2.0e-4)
        sim.sink_merging = flag
        sim.sink_merge_radius = 0.6
        sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)
        output_dirs.append(sim.evolution.scribe.snapshot_directory)      # the folder exists only once the run has started
        result[flag] =(sink.N, float(sink.masses.sum()), float(sink.E_diss_merge_total))
    print(f"4) flag: on -> {result[True][0]} sink(s), mass {result[True][1]:.3e}, E_diss {result[True][2]:.3e}; "
          f"off -> {result[False][0]} sink(s)")
    assert result[True][0] == 1, "sink_merging = True did not merge the falling sinks"
    assert abs(result[True][1] / 1.0e11 - 1.0) < 1.0e-14, "the merged mass is not the sum"
    assert result[True][2] > 0.0, "the merge energy was not recorded"
    assert result[False][0] == 2, "the sinks merged although sink_merging is off"


def main():
    try:
        check_periodic_merge()
        check_softened_bound()
        check_energy_bookkeeping()
        check_flag()
    finally:
        for directory in output_dirs:
            shutil.rmtree(directory, ignore_errors=True)
    print("PASS: sink merging (periodic centre of mass, softened bound check, energy bookkeeping, flag)")


if __name__ == "__main__":
    main()
