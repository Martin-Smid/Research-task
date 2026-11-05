"""Test: automatic sink formation (Simulation_Class(sink_formation={...}) -> Evolution._check_and_form_sinks).

Setup
    A cold uniform cloud (R = 4 kpc, 1e10 Msun, free-fall time 0.042 Gyr) falls onto its centre in a box [-10, 10]^3
    (N = 32, dx = 0.625 kpc), once as N-body particles and once as gas (R = 5 kpc, cs = 30 km/s). Both clouds start well
    below the formation threshold and pass it during the collapse. Every call of Evolution._check_and_form_sinks is
    recorded (the totals of mass, momentum and kinetic energy before and after).

Checks
    a) particles: the threshold is crossed -> exactly one sink appears, at the centre of the cloud, at the check number
       consecutive_steps after the first check that saw the density above the threshold (checks run every check_interval steps).
    b) particles: a threshold far above the peak density -> no sink, no particle consumed.
    c) particles: mass and momentum of particles + sinks are conserved at the formation; the kinetic energy that
       disappears is recorded in E_diss_formation_total.
    d) gas (enable_gas_sinks, gas_mode = "truelove"): the automatic threshold is pi cs^2 / (G (NJ dx)^2); one sink forms at
       the centre, mass and momentum of gas + sink are conserved, the specific energy of the gas cells is unchanged
       (as in accretion, E must scale with rho), the energy bookkeeping closes.
    e) the default of `source` is "both" (Evolution_Class and Simulation_Class agree); gas sinks need no extra switch.
    f) the sink appears at the centre of mass of the consumed particles / removed gas, with their mass and momentum.
    g) a cell skipped because it lies within the aggregation radius of an existing sink may form a sink later, once that
       sink has moved away (the tracker no longer blocks such cells for ever).
    i) with source "both" ONE sink consumes the particles and the gas of its region (mass and momentum conserved).
    j) the ULDM density counts in the criterion (include_uldm, default True): ULDM + baryons above the threshold forms a sink,
       baryons alone below it do not.
    m) Federrath checks (federrath_checks=True): converging + bound regions form a sink, expanding or unbound ones do not.
    n) dynamic threshold (dynamic_threshold=True): pi c^2 / (G (NJ dx)^2) with the LOCAL gas sound speed and the CURRENT particle velocity
       dispersion: a dense cell in hot gas does not form a sink, the same density in cold gas does.
    k) an explicit threshold (density_threshold or gas_density_threshold) overrides the automatic Truelove value.
    l) speed: a whole uniform gas sphere above the threshold (thousands of candidate cells) is processed in seconds.
    h) the automatic particle threshold pi sigma^2 / (G (NJ dx)^2) uses max(sigma, baryon_sigma_floor)."""

from pathlib import Path
import contextlib
import io
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (units: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # the box is [-BOX, BOX]^3 [kpc]
N_GRID = 32                         # cell 0.625 kpc
G_NEWTON = 4.498e-6                 # only to estimate the free-fall time (the simulation uses sim.G)
KM_S_TO_KPC_GYR = 1.0227121650537077
MASS = 1.0e10                       # mass of the cloud [Msun]
RADIUS_PARTICLES = 4.0              # [kpc]
RADIUS_GAS = 5.0                    # [kpc]
N_PARTICLES = 40000
CS_GAS = 30.0 * KM_S_TO_KPC_GYR     # gas sound speed [kpc/Gyr]; with NJ = 4 the automatic threshold is 1e8 Msun/kpc^3
THRESHOLD_PARTICLES = 1.0e9         # [Msun/kpc^3]; the initial peak is 8e7, the collapse reaches 2e10
THRESHOLD_FAR_ABOVE = 1.0e12        # b)
CHECK_INTERVAL = 5
CONSECUTIVE_STEPS = 3
MAX_CONSERVATION_ERROR = 1.0e-12    # relative; mass and momentum are sums of the same numbers, only rounding differs
MAX_FORMATION_RADIUS = 1.5          # the sink sits on a grid node within this many cells of the cloud centre (the densest node of a noisy cloud)


output_dirs = []                    # output folders of the simulations built below, removed at the end


def quiet():
    return contextlib.redirect_stdout(io.StringIO())    # the engine is chatty


def build(source_kind, sink_cfg, velocity=(0.0, 0.0, 0.0)):
    """Cloud + sink_formation config. Returns (sim, system); the free-fall time is sim.t_ff."""
    radius = RADIUS_PARTICLES if source_kind == "particles" else RADIUS_GAS
    t_ff = (3.0 * np.pi / (32.0 * G_NEWTON * MASS / (4.0 / 3.0 * np.pi * radius ** 3))) ** 0.5
    h = t_ff / 200.0
    with quiet():
        sim = Simulation_Class(dim=3, boundaries=[(-BOX, BOX)] * 3, N=N_GRID, total_time=1.3 * t_ff, h=h,
                               order_of_evolution=2, use_gravity=True, use_units=True, use_sponge=False, self_int=False,
                               sink_formation=sink_cfg)
        if source_kind == "particles":
            cp.random.seed(0)
            system = Baryons(sim, N_particles=N_PARTICLES, total_mass=MASS, init_profile="spherical_clump",
                             center=(0.0, 0.0, 0.0), radius=radius, velocity=velocity)
        else:
            x, y, z = (np.asarray(g) for g in sim.grids)
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            rho = np.where(r < radius, 1.0, 1e-3)
            rho *= MASS / (rho.sum() * sim.dV)
            v = [np.full_like(rho, c) for c in velocity]
            system = NBodyGas(simulation=sim, rho=rho, vx=v[0], vy=v[1], vz=v[2], cs=CS_GAS, gamma=5 / 3, name="gas")
        sim.add_baryons(system)
        sim.initialize_simulation()          # resolves the automatic thresholds and enables the formation
    sim.t_ff = t_ff
    return sim, system


def totals(sim):
    """(mass, momentum[3], kinetic energy, E_diss_formation) of all baryonic systems."""
    mass, momentum, kinetic, e_diss = 0.0, np.zeros(3), 0.0, 0.0
    for system in sim.baryonic_matter:
        if isinstance(system, SinkNBody):
            mass += float(system.masses.sum())
            momentum += cp.asnumpy((system.masses[:, None] * system.velocities).sum(axis=0))
            kinetic += float(system.kinetic_energy())
            e_diss += float(system.E_diss_formation_total)
        elif hasattr(system, "N"):
            mass += system.N * system.m_particle
            momentum += system.m_particle * cp.asnumpy(system.velocities.sum(axis=0))
            kinetic += 0.5 * system.m_particle * float((system.velocities ** 2).sum())
        else:
            mass += float(system.rho.sum()) * sim.dV
            momentum += np.array([float((system.rho * v).sum()) * sim.dV for v in (system.vx, system.vy, system.vz)])
            kinetic += float(system.kinetic_energy())
    return mass, momentum, kinetic, e_diss


def peak_density(sim):
    """Largest density of the non-sink baryons on the grid (what the formation criterion looks at)."""
    return max(float(system.deposit_to_grid().max()) for system in sim.baryonic_matter if not isinstance(system, SinkNBody))


def run_recording(sim):
    """Run the simulation and record every call of _check_and_form_sinks (a disabled call leaves the list empty)."""
    events = []
    original = sim.evolution._check_and_form_sinks

    def recording(step, wave_functions=None):
        before, rho_peak = totals(sim), peak_density(sim)
        u_before = sum(s.internal_energy() for s in sim.baryonic_matter if hasattr(s, "E"))
        specific = [s.E / s.rho for s in sim.baryonic_matter if hasattr(s, "E")]
        rho_before = [s.rho.copy() for s in sim.baryonic_matter if hasattr(s, "E")]
        created = original(step, wave_functions)
        sinks = getattr(sim.evolution, "sink_system", None)
        events.append(dict(step=step, rho_peak=rho_peak, created=bool(created), u_before=u_before,
                           u_after=sum(s.internal_energy() for s in sim.baryonic_matter if hasattr(s, "E")), before=before, after=totals(sim),
                           n_sinks=0 if sinks is None else sinks.N,
                           positions=None if sinks is None else cp.asnumpy(sinks.positions).copy(),
                           specific_before=specific, rho_before=rho_before,
                           rho_after=[s.rho.copy() for s in sim.baryonic_matter if hasattr(s, "E")], specific_after=[s.E / s.rho for s in sim.baryonic_matter if hasattr(s, "E")]))
        return created

    sim.evolution._check_and_form_sinks = recording
    with quiet():
        sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)
    output_dirs.append(sim.evolution.scribe.snapshot_directory)
    return events


def particle_config(threshold):
    return dict(density_threshold=threshold, consecutive_steps=CONSECUTIVE_STEPS, check_interval=CHECK_INTERVAL)


def check_particles_form_one_sink():
    """a) + c) the collapse crosses the threshold: one sink at the centre, at the right check, conserving mass and momentum."""
    sim, particles = build("particles", particle_config(THRESHOLD_PARTICLES), velocity=(3.0, -2.0, 1.0))
    events = run_recording(sim)
    assert events, "_check_and_form_sinks was never called (the call in Evolution.evolve is disabled)"

    formed = [e for e in events if e["created"]]
    assert formed, f"no sink formed although the density reached {max(e['rho_peak'] for e in events):.2e} > {THRESHOLD_PARTICLES:.1e}"
    first = formed[0]
    checks_over = [e for e in events if e["rho_peak"] > THRESHOLD_PARTICLES and e["step"] % CHECK_INTERVAL == 0]
    expected_step = checks_over[0]["step"] + (CONSECUTIVE_STEPS - 1) * CHECK_INTERVAL
    centre = np.array([0.0, 0.0, 0.0]) + np.array([3.0, -2.0, 1.0]) * (first["step"] * sim.h)     # the cloud moves with its bulk velocity
    distance = np.linalg.norm(first["positions"][0] - centre) / (2 * BOX / N_GRID)
    print(f"a) first sink at step {first['step']} (expected {expected_step}: first check over the threshold at step "
          f"{checks_over[0]['step']}, {CONSECUTIVE_STEPS} consecutive checks every {CHECK_INTERVAL} steps); "
          f"{first['n_sinks']} sink(s), distance from the cloud centre {distance:.2f} cells")
    assert first["n_sinks"] == 1, f"{first['n_sinks']} sinks formed at the first event, expected 1"
    assert first["step"] == expected_step, f"first sink at step {first['step']}, expected {expected_step}"
    assert distance < MAX_FORMATION_RADIUS, f"the sink is {distance:.2f} cells from the centre of the cloud"

    mass0, p0, kin0, diss0 = first["before"]
    mass1, p1, kin1, diss1 = first["after"]
    mass_error = abs(mass1 - mass0) / mass0
    momentum_error = np.abs(p1 - p0).max() / np.abs(p0).max()
    energy_error = abs((kin0 - kin1) - (diss1 - diss0)) / max(kin0, 1e-30)
    print(f"c) formation: mass error {mass_error:.1e}, momentum error {momentum_error:.1e}, kinetic energy {kin0:.4e} -> "
          f"{kin1:.4e}, recorded in E_diss_formation {diss1 - diss0:.4e} (bookkeeping error {energy_error:.1e} of K)")
    assert mass_error < MAX_CONSERVATION_ERROR, "mass of particles + sinks changed at the formation"
    assert momentum_error < MAX_CONSERVATION_ERROR, "momentum of particles + sinks changed at the formation"
    assert diss1 - diss0 >= 0.0 and energy_error < 1.0e-9, "the kinetic energy lost at the formation is not in E_diss_formation_total"
    assert kin1 <= kin0 * (1.0 + 1e-12), "the formation created kinetic energy"


def check_no_sink_below_threshold():
    """b) threshold far above the density that is ever reached: nothing happens."""
    sim, particles = build("particles", particle_config(THRESHOLD_FAR_ABOVE))
    events = run_recording(sim)
    assert events, "_check_and_form_sinks was never called (the call in Evolution.evolve is disabled)"
    rho_peak = max(e["rho_peak"] for e in events)
    n_sinks = [s for s in sim.baryonic_matter if isinstance(s, SinkNBody)]
    print(f"b) threshold {THRESHOLD_FAR_ABOVE:.0e}, peak density {rho_peak:.2e}: {len(events)} checks, "
          f"{len(n_sinks)} sink system(s), {particles.N} particles left")
    assert rho_peak > 1.0e10, "test set-up: the cloud did not collapse"
    assert not any(e["created"] for e in events) and not n_sinks, "a sink formed below the threshold"
    assert particles.N == N_PARTICLES, "particles were consumed although no sink formed"


def check_gas_sink():
    """d) gas sink with the automatic Truelove threshold."""
    cfg = dict(source="gas", gas_mode="truelove", check_interval=CHECK_INTERVAL, consecutive_steps=2)
    sim, gas = build("gas", cfg, velocity=(3.0, -2.0, 1.0))
    dx = 2 * BOX / N_GRID
    expected = np.pi * CS_GAS ** 2 / (float(sim.G) * (4 * dx) ** 2)
    threshold = sim.evolution.sink_density_threshold
    print(f"d) automatic gas threshold {threshold:.6e}, formula pi cs^2 / (G (NJ dx)^2) = {expected:.6e}")
    assert abs(threshold / expected - 1.0) < 1.0e-12, "automatic Truelove threshold differs from pi cs^2 / (G (NJ dx)^2)"
    assert float(gas.rho.max()) < threshold, "test set-up: the cloud starts above the threshold"

    events = run_recording(sim)
    assert events, "_check_and_form_sinks was never called (the call in Evolution.evolve is disabled)"
    formed = [e for e in events if e["created"]]
    assert formed, f"no gas sink formed although the density reached {max(e['rho_peak'] for e in events):.2e} > {threshold:.2e}"
    first = formed[0]
    centre = np.array([3.0, -2.0, 1.0]) * (first["step"] * sim.h)
    distance = np.linalg.norm(first["positions"][0] - centre) / dx
    print(f"   first gas sink at step {first['step']}: {first['n_sinks']} sink(s), {distance:.2f} cells from the cloud centre")
    assert first["n_sinks"] == 1, f"{first['n_sinks']} sinks formed at the first event, expected 1"
    assert distance < 2.0, f"the gas sink is {distance:.2f} cells from the centre of the cloud"

    mass0, p0, kin0, diss0 = first["before"]
    mass1, p1, kin1, diss1 = first["after"]
    mass_error = abs(mass1 - mass0) / mass0
    momentum_error = np.abs(p1 - p0).max() / np.abs(p0).max()
    energy_error = abs((kin0 - kin1) + (first["u_before"] - first["u_after"]) - (diss1 - diss0)) / kin0     # kinetic + internal energy removed
    spec_error = float(cp.abs(first["specific_after"][0] - first["specific_before"][0]).max() / first["specific_before"][0].max())
    print(f"   formation: mass error {mass_error:.1e}, momentum error {momentum_error:.1e}, kinetic bookkeeping error "
          f"{energy_error:.1e}, max change of the gas specific energy {spec_error:.1e}")
    assert mass_error < MAX_CONSERVATION_ERROR, "mass of gas + sink changed at the formation"
    assert momentum_error < 1.0e-9, "momentum of gas + sink changed at the formation"
    assert diss1 - diss0 >= 0.0 and energy_error < 1.0e-9, "the kinetic energy lost at the formation is not in E_diss_formation_total"
    assert spec_error < 1.0e-12, "the energy of the gas cells did not follow their mass (specific energy changed)"

    removed = cp.asnumpy(first["rho_before"][0] - first["rho_after"][0])
    grids = [np.asarray(g) for g in sim.grids]
    com = np.array([(removed * g).sum() / removed.sum() for g in grids])
    com_error = np.abs(first["positions"][0] - com).max()
    print(f"f) gas sink at {first['positions'][0]}, centre of mass of the removed gas {com}, difference {com_error:.1e} kpc")
    assert com_error < 1.0e-9, "the gas sink is not at the centre of mass of the gas it removed"


def build_few(positions, velocities, cfg, total_mass=3.0e9):
    """Simulation with a few particles of equal mass at the given positions."""
    with quiet():
        sim = Simulation_Class(dim=3, boundaries=[(-BOX, BOX)] * 3, N=N_GRID, total_time=0.1, h=1.0e-3,
                               order_of_evolution=2, use_gravity=True, use_units=True, use_sponge=False, self_int=False,
                               sink_formation=cfg)
        particles = Baryons(sim, N_particles=len(positions), total_mass=total_mass, init_profile="uniform")
        particles.positions = cp.asarray(positions, dtype=cp.float64)
        particles.velocities = cp.asarray(velocities, dtype=cp.float64)
        sim.add_baryons(particles)
        sim.initialize_simulation()
    return sim, particles


class UldmStub:
    """Stands in for a wave function: only calculate_density() is used by the formation check."""

    def __init__(self, density):
        self.density = density

    def calculate_density(self):
        return self.density


def build_dense_cell(cfg, rho_cell=1.0e7, particles=None):
    """Gas of 1e3 Msun/kpc^3 with one dense cell (16, 16, 16) = (0, 0, 0) kpc; optionally particles (positions, velocities, mass each)."""
    with quiet():
        sim = Simulation_Class(dim=3, boundaries=[(-BOX, BOX)] * 3, N=N_GRID, total_time=0.1, h=1.0e-3, order_of_evolution=2,
                               use_gravity=True, use_units=True, use_sponge=False, self_int=False, sink_formation=cfg)
        rho = np.full((N_GRID,) * 3, 1.0e3)
        rho[16, 16, 16] = rho_cell
        zeros = np.zeros_like(rho)
        gas = NBodyGas(simulation=sim, rho=rho, vx=zeros, vy=zeros, vz=zeros, cs=CS_GAS, gamma=5 / 3, name="gas")
        sim.add_baryons(gas)
        stars = None
        if particles is not None:
            positions, velocities, mass = particles
            stars = Baryons(sim, N_particles=len(positions), total_mass=mass * len(positions), init_profile="uniform")
            stars.positions = cp.asarray(positions, dtype=cp.float64)
            stars.velocities = cp.asarray(velocities, dtype=cp.float64)
            sim.add_baryons(stars)
        sim.initialize_simulation()
    return sim, gas, stars


def check_default_source():
    """e) the default `source` is "both"; a gas sink forms without any extra switch."""
    sim, gas, _ = build_dense_cell(dict(gas_density_threshold=1.0e5))
    default_source = sim.evolution.sink_source
    with quiet():
        created = sim.evolution._check_and_form_sinks(0)
    print(f"e) default source = {default_source!r}; gas-only configuration -> sink created: {created}")
    assert default_source == "both"
    assert created is True and any(isinstance(s, SinkNBody) for s in sim.baryonic_matter), "no gas sink was created"


def check_both_consumed():
    """i) particles + gas in one cell: one sink with the mass and momentum of both."""
    positions, velocities = [[0.05, 0.0, 0.0], [-0.05, 0.1, 0.0]], [[2.0, 0.0, 0.0], [0.0, 4.0, 1.0]]
    sim, gas, stars = build_dense_cell(dict(density_threshold=1.0e5), particles=(positions, velocities, 1.0e6))
    gas.vx[:] = 3.0                                              # the gas moves, too
    mass0, p0, _, _ = totals(sim)
    gas_before = float(gas.rho.sum()) * sim.dV
    with quiet():
        created = sim.evolution._check_and_form_sinks(0)
    sink = sim.evolution.sink_system
    mass1, p1, _, _ = totals(sim)
    gas_taken = gas_before - float(gas.rho.sum()) * sim.dV
    print(f"i) both: {sink.N} sink of {float(sink.masses[0]):.4e} Msun = particles {2.0e6:.1e} + gas {gas_taken:.4e}; "
          f"mass error {abs(mass1 - mass0) / mass0:.1e}, momentum error {np.abs(p1 - p0).max() / np.abs(p0).max():.1e}")
    assert created and sink.N == 1 and stars.N == 0 and gas_taken > 0.0, "the sink did not consume both particles and gas"
    assert abs(float(sink.masses[0]) - (2.0e6 + gas_taken)) < 1.0e-9 * float(sink.masses[0])
    assert abs(mass1 - mass0) / mass0 < 1.0e-12 and np.abs(p1 - p0).max() / np.abs(p0).max() < 1.0e-12


def check_uldm_density():
    """j) 1e4 of gas + 2e5 of ULDM in one cell is above the threshold 1e5, the gas alone is not."""
    uldm = np.zeros((N_GRID,) * 3)
    uldm[16, 16, 16] = 2.0e5
    results = {}
    for flag in (True, False):
        sim, gas, _ = build_dense_cell(dict(density_threshold=1.0e5, include_uldm=flag), rho_cell=1.0e4)
        with quiet():
            results[flag] = sim.evolution._check_and_form_sinks(0, [UldmStub(cp.asarray(uldm))])
    print(f"j) ULDM density in the criterion: include_uldm=True -> sink {results[True]}, False -> sink {results[False]}")
    assert results[True] is True and results[False] is False


def check_explicit_threshold():
    """k) explicit thresholds replace the automatic Truelove value."""
    auto = build_dense_cell(dict(source="gas"))[0].evolution.sink_density_threshold
    given = build_dense_cell(dict(source="gas", density_threshold=7.7e6))[0].evolution.sink_density_threshold
    given_gas = build_dense_cell(dict(source="gas", gas_density_threshold=5.5e6))[0].evolution.sink_density_threshold
    print(f"k) threshold: automatic {auto:.4e}, density_threshold=7.7e6 -> {given:.4e}, gas_density_threshold=5.5e6 -> {given_gas:.4e}")
    assert given == 7.7e6 and given_gas == 5.5e6 and auto not in (7.7e6, 5.5e6)


def check_federrath():
    """m) three particles around the origin: inward motion forms a sink, outward motion does not, inward but unbound does not."""
    positions = [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    results = {}
    for name, speed in (("inward", -1.0), ("outward", 1.0), ("inward, unbound", -500.0)):
        velocities = [[speed * p[0] / 0.5, speed * p[1] / 0.5, 0.0] for p in positions]
        for flag in (False, True):
            sim, _ = build_few(positions, velocities, dict(density_threshold=1.0, consecutive_steps=1, federrath_checks=flag))
            with quiet():
                results[(name, flag)] = sim.evolution._check_and_form_sinks(0)
    print("m) Federrath: " + ", ".join(f"{name}: off {results[(name, False)]} / on {results[(name, True)]}" for name in ("inward", "outward", "inward, unbound")))
    assert all(results[(name, False)] for name in ("inward", "outward", "inward, unbound")), "the density criterion alone must form a sink"
    assert results[("inward", True)] and not results[("outward", True)] and not results[("inward, unbound", True)]


def check_dynamic_threshold():
    """n) hot and cold gas, and a particle dispersion that changes in time."""
    with quiet():
        sim = Simulation_Class(dim=3, boundaries=[(-BOX, BOX)] * 3, N=N_GRID, total_time=0.1, h=1.0e-3, order_of_evolution=2,
                               use_gravity=True, use_units=True, use_sponge=False, self_int=False,
                               sink_formation=dict(source="gas", dynamic_threshold=True))
        rho_cold = np.pi * CS_GAS ** 2 / (float(sim.G) * (4 * 2 * BOX / N_GRID) ** 2)       # Truelove threshold of the cold gas
        rho = np.full((N_GRID,) * 3, 1.0e3)
        rho[8, 16, 16] = rho[24, 16, 16] = 2.0 * rho_cold
        zeros = np.zeros_like(rho)
        gas = NBodyGas(simulation=sim, rho=rho, vx=zeros, vy=zeros, vz=zeros, cs=CS_GAS, gamma=5 / 3, name="gas")
        gas.E[17:] *= 9.0                                                                  # hot half: cs is 3 times larger
        sim.add_baryons(gas)
        sim.initialize_simulation()
    grid = cp.asnumpy(sim.evolution._dynamic_sink_threshold())
    print(f"n) dynamic threshold: cold {grid[8, 16, 16]:.4e} (Truelove {rho_cold:.4e}), hot {grid[24, 16, 16]:.4e} (9x)")
    assert abs(grid[8, 16, 16] / rho_cold - 1.0) < 1.0e-9 and abs(grid[24, 16, 16] / (9.0 * rho_cold) - 1.0) < 1.0e-9
    with quiet():
        created = sim.evolution._check_and_form_sinks(0)
    sinks = sim.evolution.sink_system
    x = cp.asnumpy(sinks.positions)[:, 0]
    assert created and sinks.N == 1 and x[0] < 0.0, "only the dense cell in the cold gas should form a sink"

    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    sim, particles = build_few(positions, [[0.0, 0.0, 0.0]] * 4, dict(dynamic_threshold=True, baryon_sigma_floor=0.5))
    dx = 2 * BOX / N_GRID
    before = float(sim.evolution._dynamic_sink_threshold().min())
    velocities = np.random.default_rng(0).normal(size=(4, 3)) * 20.0
    particles.velocities = cp.asarray(velocities)
    after = float(sim.evolution._dynamic_sink_threshold().min())
    sigma = np.sqrt(((velocities - velocities.mean(axis=0)) ** 2).sum(axis=1).mean() / 3.0)
    expected = np.pi * np.array([0.5, sigma]) ** 2 / (float(sim.G) * (4 * dx) ** 2)
    print(f"   particles: threshold {before:.4e} (floor 0.5, expected {expected[0]:.4e}) -> {after:.4e} (sigma {sigma:.2f}, expected {expected[1]:.4e})")
    assert abs(before / expected[0] - 1.0) < 1.0e-9 and abs(after / expected[1] - 1.0) < 1.0e-9


def check_speed():
    """l) a uniform gas sphere (about 2000 cells above the threshold) is processed in seconds, mass is conserved."""
    from time import perf_counter
    sim, gas = build("gas", dict(source="gas", density_threshold=1.0e6, consecutive_steps=1))
    mass0 = totals(sim)[0]
    start = perf_counter()
    with quiet():
        created = sim.evolution._check_and_form_sinks(0)
    elapsed = perf_counter() - start
    mass1 = totals(sim)[0]
    print(f"l) uniform sphere above the threshold: {sim.evolution.sink_system.N} sinks in {elapsed:.2f} s, mass error {abs(mass1 - mass0) / mass0:.1e}")
    assert created and elapsed < 30.0, f"sink formation took {elapsed:.1f} s"
    assert abs(mass1 - mass0) / mass0 < 1.0e-12


def check_particle_centre_of_mass():
    """f) three particles of different velocity: one sink at their centre of mass, with their total mass and momentum."""
    positions = [[0.10, 0.0, 0.0], [0.30, 0.1, 0.0], [0.80, -0.1, 0.2]]
    velocities = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 6.0]]
    sim, particles = build_few(positions, velocities, dict(density_threshold=1.0, consecutive_steps=1))
    with quiet():
        created = sim.evolution._check_and_form_sinks(0)
    sink = sim.evolution.sink_system
    position, velocity = cp.asnumpy(sink.positions)[0], cp.asnumpy(sink.velocities)[0]
    print(f"f) particles -> {sink.N} sink at {position} (centre of mass {np.mean(positions, axis=0)}), mass {float(sink.masses[0]):.3e}, "
          f"velocity {velocity}")
    assert created and sink.N == 1 and particles.N == 0
    assert np.abs(position - np.mean(positions, axis=0)).max() < 1.0e-12, "the sink is not at the centre of mass"
    assert abs(float(sink.masses[0]) / 3.0e9 - 1.0) < 1.0e-14
    assert np.abs(velocity - np.mean(velocities, axis=0)).max() < 1.0e-12, "the sink does not carry the momentum"


def check_blocked_cell_forms_later():
    """g) a clump 1 kpc from an existing sink is excluded; after that sink moved away the clump forms its own sink."""
    positions = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    sim, particles = build_few(positions, [[0.0, 0.0, 0.0]] * 3, dict(density_threshold=1.0, consecutive_steps=1))
    with quiet():
        existing = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[1.0e8], initial_positions=[[1.0, 0.0, 0.0]],
                             initial_velocities=[[0.0, 0.0, 0.0]])
        sim.add_baryons(existing)
        sim.evolution.sink_system = existing
        first = sim.evolution._check_and_form_sinks(0)
        n_left = particles.N
        existing.positions = cp.asarray([[8.0, 0.0, 0.0]])
        second = sim.evolution._check_and_form_sinks(10)
    print(f"g) clump next to an existing sink: formed {first}; after the sink moved away: formed {second} ({existing.N} sinks)")
    assert first is False, f"first={first!r}"
    assert n_left == 3, "a sink formed inside the aggregation radius of an existing sink"
    assert second is True and existing.N == 2, "the cell stayed blocked after the existing sink moved away"


def check_sigma_floor():
    """h) automatic particle threshold = pi max(sigma, floor)^2 / (G (NJ dx)^2); a cold cloud (sigma = 0) needs the floor."""
    floor = 5.0
    sim, _ = build_few([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]] * 2,
                       dict(baryon_sigma_floor=floor, baryon_NJ=4))
    dx = 2 * BOX / N_GRID
    expected = np.pi * floor ** 2 / (float(sim.G) * (4 * dx) ** 2)
    got = sim.evolution.sink_density_threshold
    print(f"h) cold particles, sigma_floor {floor}: threshold {got:.6e}, formula {expected:.6e}")
    assert got is not None and abs(got / expected - 1.0) < 1.0e-12


def main():
    try:
        check_default_source()
        check_both_consumed()
        check_uldm_density()
        check_explicit_threshold()
        check_speed()
        check_federrath()
        check_dynamic_threshold()
        check_particle_centre_of_mass()
        check_blocked_cell_forms_later()
        check_sigma_floor()
        check_particles_form_one_sink()
        check_no_sink_below_threshold()
        check_gas_sink()
    finally:
        for directory in output_dirs:
            shutil.rmtree(directory, ignore_errors=True)
    print("PASS: sink formation (particles and gas: threshold, one sink at the centre, conservation, default source)")


if __name__ == "__main__":
    main()
