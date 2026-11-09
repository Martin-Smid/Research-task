"""Test: absorption of ULDM by sinks (SinkNBody.accrete_from_uldm, enabled by Simulation_Class.sink_accretion_uldm).

dM/dt = 16 pi (G M / c^2)^2 c rho_hor (Bucciotti & Trincherini 2309.02482, Eq. 3.25; Unruh 1976), applied as an absorbing
factor psi -> psi sqrt(1 - f), f = 1 - exp(-K W dt), W = normalised Gaussian of width softening_bh around the sink.
a) plane wave psi = sqrt(rho0) exp(i k x), one sink with an enlarged cross section: dM = mass lost by psi (1e-12),
   dM = K rho0 dt for small K W dt, dP = dM hbar_tilde k.
b) huge K dt and two overlapping sinks: no NaN, rho >= 0 and never above the initial rho, the sinks gain what psi loses.
c) short evolve with the flag on: mass of psi + sinks + gas + particles conserved; with the flag off (or sink_accretion off)
   accrete_from_uldm is never called."""

from pathlib import Path
import sys
import io
import contextlib
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_function_class import Wave_function

N, BOX, RHO0, M_SINK, DT = 16, 8.0, 1.0e4, 1.0e9, 1.0e-3
L = 2.0 * BOX


def quiet():
    return contextlib.redirect_stdout(io.StringIO())    # the engine is chatty


class Field:
    """Minimal stand-in for a wave function: psi and its density."""

    def __init__(self, psi):
        self.psi = psi

    def calculate_density(self):
        return cp.abs(self.psi).astype(cp.float64) ** 2


def make_sim(total_time=DT):
    with quiet():
        return Simulation_Class(dim=3, boundaries=[(-BOX, BOX)] * 3, N=N, total_time=total_time, h=DT, order_of_evolution=2,
                                use_gravity=True, use_units=True, use_sponge=False, self_int=False)


def plane_wave(sim):
    k = 2.0 * np.pi / L
    x = cp.asarray(sim.grids[0])
    return k, (np.sqrt(RHO0) * cp.exp(1j * k * x)).astype(cp.complex128)


def mass_of(field, sim):
    return float(cp.sum(field.calculate_density())) * sim.dV


def scale_for(sim, sink, target):
    """Cross-section scale giving K * DT * W_max = target (W_max estimated from the isolated Gaussian)."""
    eps = float(sink.softening_bh)
    w_max = (2.0 * np.pi * eps ** 2) ** -1.5
    k_base = 16.0 * np.pi * (sim.G * float(sink.masses[0]) / sim.c ** 2) ** 2 * sim.c
    return target / (k_base * w_max * DT)


def check_plane_wave():
    sim = make_sim()
    sink = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[M_SINK], initial_positions=[[0.3, -0.2, 0.1]],
                     initial_velocities=[[0.0, 0.0, 0.0]])
    sim.add_baryons(sink)
    k, psi = plane_wave(sim)
    field = Field(psi)
    sink.uldm_cross_section_scale = scale_for(sim, sink, 1.0e-6)
    K = sink.uldm_cross_section_scale * 16.0 * np.pi * (sim.G * M_SINK / sim.c ** 2) ** 2 * sim.c

    dM = sink.accrete_from_uldm([field], DT)
    expected = K * RHO0 * DT
    print(f"plane wave: dM {dM:.6e}, K rho0 dt {expected:.6e}")
    assert abs(dM / expected - 1.0) < 1.0e-5, "dM differs from K rho_W dt"

    # mass balance with a stronger absorption (the lost mass is then far above the rounding of the sums)
    sink_b = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[M_SINK], initial_positions=[[0.3, -0.2, 0.1]],
                       initial_velocities=[[0.0, 0.0, 0.0]])
    sink_b.uldm_cross_section_scale = scale_for(sim, sink_b, 0.1)
    field_b = Field(plane_wave(sim)[1])
    before = mass_of(field_b, sim)
    dM_b = sink_b.accrete_from_uldm([field_b], DT)
    lost = before - mass_of(field_b, sim)
    print(f"strong absorption: dM {dM_b:.6e}, psi lost {lost:.6e}")
    assert dM_b > 0.0 and abs(dM_b - lost) <= 1.0e-12 * lost, "the sink gain differs from the mass lost by psi"
    assert abs(float(sink.mass_res[0]) - dM) <= 1.0e-12 * dM and abs(float(sink.masses[0]) - M_SINK - dM) <= 1.0e-6 * dM      # rounding of 1e9 + dM
    assert abs(sink.total_accreted_mass - dM) <= 1.0e-12 * dM and sink.accretion_history[-1][0] == "uldm_absorption"
    dP = (float(sink.masses[0]) * cp.asnumpy(sink.velocities)[0])        # started at rest
    expected_p = dM * sim.h_bar_tilde * k
    print(f"momentum: dP = {dP}, dM hbar k = {expected_p:.6e}")
    assert abs(dP[0] / expected_p - 1.0) < 1.0e-9 and abs(dP[1]) < 1.0e-9 * expected_p and abs(dP[2]) < 1.0e-9 * expected_p
    print("PASS: plane wave absorption (mass, rate, momentum)")


def check_large_kdt():
    sim = make_sim()
    sink = SinkNBody(simulation=sim, N_sinks=2, initial_masses=[M_SINK, 2.0 * M_SINK],
                     initial_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], initial_velocities=[[0.0] * 3, [0.0] * 3])
    sim.add_baryons(sink)
    _, psi = plane_wave(sim)
    field = Field(psi)
    sink.uldm_cross_section_scale = scale_for(sim, sink, 50.0)
    rho_before = field.calculate_density().copy()
    before = mass_of(field, sim)
    dM = sink.accrete_from_uldm([field], DT)
    rho = field.calculate_density()
    lost = before - mass_of(field, sim)
    gains = cp.asnumpy(sink.mass_res)
    print(f"large K dt: dM {dM:.6e}, psi lost {lost:.6e}, per sink {gains}, min rho {float(rho.min()):.3e}")
    assert bool(cp.isfinite(field.psi).all()) and bool(cp.isfinite(sink.velocities).all()) and np.isfinite(dM)
    assert float(rho.min()) >= 0.0 and bool((rho <= rho_before * (1.0 + 1.0e-12)).all())
    assert abs(dM - lost) <= 1.0e-12 * lost and abs(gains.sum() - lost) <= 1.0e-12 * lost and (gains > 0).all()
    print("PASS: saturated absorption stays finite, non-negative and conservative")


def build_evolve_case(uldm, sink_accretion=True):
    sim = make_sim(total_time=10 * DT)
    sim.sink_accretion = sink_accretion
    sim.sink_accretion_uldm = uldm
    with quiet():
        wf = Wave_function(sim, packet_type="blank", means=[0, 0, 0], st_deviations=[1.0, 1.0, 1.0], momenta=[0, 0, 0])
    _, wf.psi = plane_wave(sim)
    sim.wave_functions.append(wf)
    shape = (N,) * 3
    gas = NBodyGas(simulation=sim, rho=cp.full(shape, 1.0e3), vx=cp.zeros(shape), vy=cp.zeros(shape), vz=cp.zeros(shape),
                   cs=10.0, gamma=5.0 / 3.0, name="gas")
    cp.random.seed(0)
    with quiet():
        stars = Baryons(sim, N_particles=200, total_mass=1.0e6, init_profile="uniform")
    sink = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[M_SINK], initial_positions=[[0.3, -0.2, 0.1]],
                     initial_velocities=[[0.0, 0.0, 0.0]])
    sink.uldm_cross_section_scale = scale_for(sim, sink, 1.0e-2)
    for system in (gas, stars, sink):
        sim.add_baryons(system)
    with quiet():
        sim.initialize_simulation()
    return sim, wf, gas, stars, sink


def total_mass(sim, wf, gas, stars, sink):
    return (mass_of(wf, sim) + float(cp.sum(gas.rho)) * sim.dV + stars.N * stars.m_particle + float(sink.masses.sum()))


def run(sim):
    with quiet():
        sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)
    shutil.rmtree(sim.evolution.scribe.snapshot_directory, ignore_errors=True)       # the test leaves no output behind


def check_evolve():
    sim, wf, gas, stars, sink = build_evolve_case(uldm=True)
    m0, psi0, sink0 = total_mass(sim, wf, gas, stars, sink), mass_of(wf, sim), float(sink.masses.sum())
    run(sim)
    m1, psi1, sink1 = total_mass(sim, wf, gas, stars, sink), mass_of(wf, sim), float(sink.masses.sum())
    print(f"evolve, flag on: psi {psi0:.6e} -> {psi1:.6e}, sink {sink0:.6e} -> {sink1:.6e}, total {m0:.9e} -> {m1:.9e}")
    assert psi1 < psi0 * (1.0 - 1.0e-6), "psi did not lose mass: ULDM accretion is not called"
    assert abs(m1 / m0 - 1.0) < 1.0e-9, "the total mass is not conserved"

    for uldm, sink_accretion in ((False, True), (True, False)):
        sim, *_ = build_evolve_case(uldm=uldm, sink_accretion=sink_accretion)
        calls = []
        SinkNBody.accrete_from_uldm_original = SinkNBody.accrete_from_uldm
        SinkNBody.accrete_from_uldm = lambda self, *a, **k: calls.append(1) or 0.0
        try:
            run(sim)
        finally:
            SinkNBody.accrete_from_uldm = SinkNBody.accrete_from_uldm_original
            del SinkNBody.accrete_from_uldm_original
        assert not calls, f"accrete_from_uldm was called with sink_accretion_uldm={uldm}, sink_accretion={sink_accretion}"
    print("PASS: evolve with the flag on conserves mass; flag off / sink_accretion off never calls it")


def main():
    check_plane_wave()
    check_large_kdt()
    check_evolve()


if __name__ == "__main__":
    main()
