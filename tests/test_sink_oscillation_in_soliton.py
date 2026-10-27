"""Test D: a light sink released at rest off-centre in a soliton oscillates through the centre.

Why
    The sink moves under the gravity of the wave function (forces_environment in Evolution._evolve_order_2), while the
    wave function evolves at the same time. This checks the sink motion in a source that is itself dynamical.

Setup
    Ground-state soliton (5.3e7 Msun, in equilibrium up to grid effects). A light sink (negligible mass, it does not
    disturb the soliton) starts at rest at x = X0_FRACTION * r_half on the x axis and falls through the centre.

References
    1) static reference: the same motion integrated with RK4 in the force grid of the initial soliton (frozen source).
       The full simulation differs from it only by the time dependence of the source (grid-scale breathing).
    2) harmonic estimate: omega^2 = (4 pi / 3) G rho_c, valid only for X0 much smaller than the core radius.

Checks
    The period from the simulation agrees with the static reference within PERIOD_TOL and the oscillation amplitude
    stays within AMPLITUDE_TOL of the starting offset (no secular energy gain or loss)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 5.0                       # box is [-BOX, BOX]^3
N_GRID = 64                     # grid resolution (cell 0.156 kpc)
H = 5.0e-4                      # time step [Gyr] (stability limit for this grid is 6.6e-4)
M_SOLITON = 53090072.0          # soliton mass [Msun]
M_SINK = 1.0e4                  # sink mass [Msun], negligible compared to the soliton
X0_FRACTION = 1.0               # starting offset in units of the half-density core radius
N_PERIODS = 2.0                 # how many (reference) periods are simulated
PERIOD_TOL = 0.05               # largest allowed relative difference of the period from the static reference
AMPLITUDE_TOL = 0.10            # largest allowed relative change of the oscillation amplitude


def build_simulation(total_time, x0=None):
    """Soliton + light sink. Returns (sim, sink, wave_function); if x0 is None the sink sits at the centre."""

    sim = Simulation_Class(

        dim=3,                              # 3D simulation
        boundaries=[(-BOX, BOX)] * 3,       # spatial boundaries, centre of the soliton is (0, 0, 0)
        N=N_GRID,                           # grid resolution
        total_time=total_time,              # total simulation time [Gyr]
        h=H,                                # time step [Gyr]
        order_of_evolution=2,               # order of the evolution scheme
        use_gravity=True,                   # gravity on
        use_sponge=False,                   # no absorbing boundary
        self_int=False,                     # no self-interaction
        save_max_vals=False,                # do not store maximum density
        m_s=2.5e-22,                        # ULDM particle mass [eV]

    )

    wave_vector = Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat",    # ground-state soliton profile
        means=[0, 0, 0],                                        # centre
        st_deviations=[0.5, 0.5, 0.5],                          # unused for a profile read from a file
        simulation=sim,                                         # simulation
        mass=1,                                                 # mass
        omega=1,                                                # frequency
        momenta=[0, 0.0, 0],                                    # no momentum
        spin=0,                                                 # scalar field
        desired_soliton_mass=M_SOLITON,                         # soliton mass [Msun]

    )
    sim.add_wave_vector(wave_vector)

    sink = SinkNBody(

        simulation=sim,                                         # simulation the sink belongs to
        N_sinks=1,                                              # one sink
        initial_masses=[M_SINK],                                # its mass [Msun]
        initial_positions=[[0.0 if x0 is None else x0, 0.0, 0.0]],   # starting position
        initial_velocities=[[0.0, 0.0, 0.0]],                   # at rest

    )
    sim.add_baryons(sink)
    sim.initialize_simulation()
    wave_function = sim.wave_functions[0]
    wave_function.psi = cp.asarray(wave_function.psi)
    return sim, sink, wave_function


def static_reference(sim, sink, x0, dt, n_steps):
    """Sink motion in the frozen force grid of the initial soliton (RK4). Returns x(t) of shape (n_steps + 1,)."""

    density = sim.evolution._compute_total_density(sim.wave_functions)
    phi_environment, _ = sim.evolution._compute_gravity_fields(density)
    forces = sim.propagator.compute_force_grids_from_potential(phi_environment)

    def acceleration(x):
        sink.positions = cp.asarray([[x, 0.0, 0.0]], dtype=cp.float64)
        return float(cp.asnumpy(sink._interpolate_force_trilinear(*forces))[0, 0])

    def derivative(state):
        return np.array([state[1], acceleration(state[0])])

    state = np.array([x0, 0.0])
    trajectory = [state[0]]
    for _ in range(n_steps):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory.append(state[0])
    return np.array(trajectory)


def period_and_amplitude(time, x):
    """Period from the zero crossings of x(t) (twice the mean half-period) and the largest |x| in each half-oscillation."""

    sign_change = np.where(np.sign(x[:-1]) * np.sign(x[1:]) < 0)[0]
    crossings = time[sign_change] - x[sign_change] * (time[sign_change + 1] - time[sign_change]) / (
        x[sign_change + 1] - x[sign_change])
    period = 2.0 * float(np.mean(np.diff(crossings))) if len(crossings) > 1 else float("nan")
    edges = np.concatenate(([0], sign_change + 1, [len(x)]))
    amplitudes = [float(np.abs(x[a:b]).max()) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return period, amplitudes


def main():
    # soliton properties and the static reference (a short simulation object is enough for the force grid)
    probe, probe_sink, probe_wave = build_simulation(total_time=H)
    half_radius = float(probe_wave.soliton_radius)
    x0 = X0_FRACTION * half_radius
    rho_centre = float(abs(probe.evolution._compute_total_density(probe.wave_functions)).max())
    g_const = float(probe.G)
    omega = np.sqrt(4.0 * np.pi / 3.0 * g_const * rho_centre)
    period_harmonic = 2.0 * np.pi / omega
    print(f"soliton {M_SOLITON:.2e} Msun: half-density radius {half_radius:.3f} kpc, rho_c = {rho_centre:.3e} Msun/kpc^3, "
          f"sink starts at x0 = {x0:.3f} kpc ({x0 / min(probe.dx):.1f} cells)")

    n_ref = int(round(N_PERIODS * 1.5 * period_harmonic / H))      # generous length for the reference
    ref = static_reference(probe, probe_sink, x0, H, n_ref)
    time_ref = np.arange(len(ref)) * H
    period_ref, amplitudes_ref = period_and_amplitude(time_ref, ref)
    total_time = N_PERIODS * period_ref
    print(f"static reference period {period_ref:.4f} Gyr, harmonic estimate {period_harmonic:.4f} Gyr "
          f"(valid for x0 << core radius), simulating {total_time:.3f} Gyr")

    # the real simulation
    sim, sink, wave_function = build_simulation(total_time=total_time, x0=x0)
    trajectory = [float(cp.asnumpy(sink.positions)[0, 0])]
    original_step = sim.evolution._perform_evolution_step

    def recorded_step(wave_functions, total_density, step, save_step):
        result = original_step(wave_functions, total_density, step, save_step)
        trajectory.append(float(cp.asnumpy(sink.positions)[0, 0]))
        return result

    sim.evolution._perform_evolution_step = recorded_step
    sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)

    x = np.array(trajectory)
    time = np.arange(len(x)) * H
    period_sim, amplitudes_sim = period_and_amplitude(time, x)
    ref_same = ref[: len(x)]
    print(f"\n{'':24} {'period [Gyr]':>14} {'amplitudes [kpc]'}")
    print(f"{'static reference':24} {period_ref:>14.4f} {[round(a, 3) for a in amplitudes_ref[:5]]}")
    print(f"{'full simulation':24} {period_sim:>14.4f} {[round(a, 3) for a in amplitudes_sim[:5]]}")
    print(f"period difference simulation vs static reference: {100 * (period_sim / period_ref - 1):+.2f} %")
    print(f"max |x_sim - x_ref| / x0 over the run: {np.abs(x - ref_same).max() / x0:.3e}")

    amplitude_change = max(abs(a / x0 - 1.0) for a in amplitudes_sim[:-1])    # last half-oscillation may be cut
    assert abs(period_sim / period_ref - 1.0) < PERIOD_TOL, (
        f"period differs from the static reference by {100 * (period_sim / period_ref - 1):+.2f} %")
    assert amplitude_change < AMPLITUDE_TOL, f"oscillation amplitude changed by {100 * amplitude_change:.1f} %"
    print(f"PASS: the sink oscillates in the soliton (period within {100 * abs(period_sim / period_ref - 1):.2f} % of the static "
          f"reference, amplitude change {100 * amplitude_change:.1f} %)")


if __name__ == "__main__":
    main()
