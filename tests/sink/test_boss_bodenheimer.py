"""Benchmark: Boss & Bodenheimer (1979) collapse of a rotating, m = 2 perturbed isothermal cloud -> binary formation with sinks.

Initial conditions (generated here, no external IC file)
    sphere R = 2139 AU, M = 1 Msun (mean density rho0), density perturbation rho = rho0 (1 + 0.5 cos 2 phi) inside the sphere
    (renormalised to M), solid-body rotation Omega = 1.56e-12 rad/s about z, T = 10 K (sound speed 0.166 km/s), embedded in
    a cold medium of rho0 / 100 at rest. All quantities are converted with astropy to the code units (kpc, Msun, Gyr).
Gas
    NBodyGas with gamma = 1.001: the pressure P = (gamma - 1) e = cs^2 rho / gamma, i.e. nearly isothermal (the engine has
    no isothermal closure).
Sinks
    Sink formation from the gas (Truelove threshold with N_J = 4, automatic) with the Federrath checks (converging flow, bound),
    through sim.sink_formation. Without the checks the collapsing cores keep re-forming sinks (7 at N = 64, 23 at N = 128).
Expected
    the m = 2 mode collapses into two dense cores on the free-fall time (~1.7e4 yr here) -> two sinks on opposite sides of the
    axis. Logged: the time at which the sinks form, their masses [Msun] and their separation [AU]."""

from pathlib import Path
import contextlib
import io
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import astropy.constants as const
import astropy.units as u
import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS (physical, converted to kpc, Msun, Gyr below)
# ---------------------------------------------------------------------------------------------------------
RADIUS = 2139 * u.AU
MASS = 1.0 * u.Msun
OMEGA = 1.56e-12 / u.s
SOUND_SPEED = 0.166 * u.km / u.s
TEMPERATURE = 10 * u.K
AMPLITUDE = 0.5                     # m = 2 density perturbation
AMBIENT_CONTRAST = 100.0            # rho_sphere / rho_ambient
N_GRID = 64                         # box 3 R, cell about 100 AU
BOX_IN_RADII = 1.5                  # the box is [-1.5 R, 1.5 R]^3
RUN_IN_T_FF = 1.5
GAMMA = 1.001                       # nearly isothermal


def code(quantity, unit):
    return float(quantity.to(unit).value)


def generate_ics(sim, radius, mass, omega, amplitude):
    """Density, velocity arrays on the simulation grid (code units: Msun/kpc^3, kpc/Gyr)."""
    x, y, z = (cp.asnumpy(cp.asarray(g)) for g in sim.grids)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    inside = r < radius
    rho = 1.0 + amplitude * np.cos(2.0 * phi)
    rho *= mass / (rho[inside].sum() * sim.dV)
    rho = np.where(inside, rho, mass / (4.0 / 3.0 * np.pi * radius ** 3) / AMBIENT_CONTRAST)
    vx = np.where(inside, -omega * y, 0.0)
    vy = np.where(inside, omega * x, 0.0)
    return rho, vx, vy, np.zeros_like(rho)


def main(n_grid=N_GRID, federrath=True, strict=True, keep_output=False, save_every=10 ** 9, diagnostics_every=10 ** 9, save_max_vals=False, on_step=None):
    """Run the benchmark. The extra arguments are for plotting (plot_boss_bodenheimer.py): keep the output folder, save snapshots
    and energies/max values every so many steps, call on_step(step, sim, gas) every step. Returns a dict with the results."""
    R = code(RADIUS, u.kpc)
    M = code(MASS, u.Msun)
    omega = code(OMEGA, 1 / u.Gyr)
    cs = code(SOUND_SPEED, u.kpc / u.Gyr)
    rho0 = (MASS / (4.0 / 3.0 * np.pi * RADIUS ** 3)).to(u.g / u.cm ** 3)
    t_ff = (3.0 * np.pi / (32.0 * const.G * rho0)) ** 0.5
    mu = (const.k_B * TEMPERATURE / (SOUND_SPEED ** 2 * const.m_p)).decompose().value
    t_ff_yr = t_ff.to(u.yr).value
    print(f"R = {R:.4e} kpc, M = {M} Msun, rho0 = {rho0:.3e}, t_ff = {t_ff_yr:.0f} yr, Omega t_ff = {(OMEGA * t_ff).decompose():.2f}, "
          f"cs = {SOUND_SPEED} (T = {TEMPERATURE} needs mu = {mu:.2f}), Jeans mass check M / M_J = "
          f"{(MASS / (0.5 * (np.pi * SOUND_SPEED ** 2 / const.G) ** 1.5 * (rho0) ** -0.5)).decompose().value:.2f}")

    h = code(t_ff, u.Gyr) / 200.0
    box = BOX_IN_RADII * R
    with contextlib.redirect_stdout(io.StringIO()):
        sim = Simulation_Class(dim=3, boundaries=[(-box, box)] * 3, N=n_grid, total_time=RUN_IN_T_FF * code(t_ff, u.Gyr), h=h,
                               order_of_evolution=2, use_gravity=True, use_units=True, use_sponge=False, self_int=False,
                               sink_formation=dict(source="gas", check_interval=2, consecutive_steps=2, federrath_checks=federrath), save_max_vals=save_max_vals)
        G_astropy = code(const.G, u.kpc ** 3 / (u.Msun * u.Gyr ** 2))
        rho, vx, vy, vz = generate_ics(sim, R, M, omega, AMPLITUDE)
        gas = NBodyGas(simulation=sim, rho=rho, vx=vx, vy=vy, vz=vz, cs=cs, gamma=GAMMA, name="gas", rho_floor=1.0e-6 * rho.min(),
                       e_floor=1.0e-12 * cs ** 2)
        sim.add_baryons(gas)
        sim.initialize_simulation()
    assert abs(float(sim.G) / G_astropy - 1.0) < 1.0e-3, f"engine G {float(sim.G)} differs from astropy {G_astropy}"
    dx = box * 2 / n_grid
    threshold = sim.evolution.sink_density_threshold
    print(f"grid {n_grid}^3, dx = {(dx * u.kpc).to(u.AU):.1f}, h = {h:.3e} Gyr, Truelove threshold {threshold:.3e} Msun/kpc^3 "
          f"= {(threshold * u.Msun / u.kpc ** 3).to(u.g / u.cm ** 3):.3e} = {threshold / (M / (4 / 3 * np.pi * R ** 3)):.1f} rho0; "
          f"initial mass in the box {float(gas.rho.sum()) * sim.dV:.4f} Msun")

    mass_initial = float(gas.rho.sum()) * sim.dV
    events = []
    original = sim.evolution._check_and_form_sinks

    def recording(step, wave_functions=None):
        if on_step is not None:
            on_step(step, sim, gas)
        created = original(step, wave_functions)
        if created:
            sinks = sim.evolution.sink_system
            events.append((step * h, sinks.N, cp.asnumpy(sinks.masses).copy(), cp.asnumpy(sinks.positions).copy(), float(gas.rho.max())))
        return created

    sim.evolution._check_and_form_sinks = recording
    with contextlib.redirect_stdout(io.StringIO()):
        sim.evolution.evolve(sim.wave_functions, save_every=save_every, diagnostics_every=diagnostics_every)
    output_dir = sim.evolution.scribe.snapshot_directory
    if not keep_output:
        shutil.rmtree(output_dir, ignore_errors=True)

    print(f"{len(events)} formation event(s) up to {RUN_IN_T_FF} t_ff")
    for time, n, masses, positions, rho_max in events:
        print(f"  t = {time / code(t_ff, u.Gyr):.3f} t_ff = {(time * u.Gyr).to(u.yr):.0f}: {n} sink(s) in total, "
              f"masses {np.round(masses, 5)} Msun, peak gas density {rho_max:.2e}")
    assert events, "no sink formed"

    def check(condition, message):
        if strict:
            assert condition, message
        elif not condition:
            print("NOT MET:", message)

    time, n, masses, positions, _ = events[0]
    separation = (np.linalg.norm(positions[0] - positions[-1]) * u.kpc).to(u.AU)
    azimuth = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))
    sinks = [s for s in sim.baryonic_matter if isinstance(s, SinkNBody)][0]
    total_mass = float(gas.rho.sum()) * sim.dV + float(sinks.masses.sum())
    print(f"first event: t = {time / code(t_ff, u.Gyr):.2f} t_ff = {(time * u.Gyr).to(u.yr):.0f}, {n} sinks of {masses} Msun, "
          f"separation {separation:.0f}, azimuths {azimuth} deg (m = 2: 180 deg apart); at the end {sinks.N} sinks "
          f"(accretion is off; without the Federrath checks the collapsing cores keep re-forming sinks); gas + sinks = {total_mass:.6f} Msun, initial {mass_initial:.6f}")
    check(n == 2, f"{n} sinks formed at the first event, expected the two cores of the m = 2 mode")
    check(0.8 < time / code(t_ff, u.Gyr) < 1.3, "the cores collapsed at a wrong time")
    check(n >= 2 and abs(masses[0] / masses[1] - 1.0) < 0.05, "the two cores differ in mass")
    check(n >= 2 and abs(abs(azimuth[0] - azimuth[1]) - 180.0) < 10.0, "the two cores are not on opposite sides of the axis")
    check(abs(total_mass / mass_initial - 1.0) < 1.0e-9, "mass is not conserved")
    check(not federrath or sinks.N == 2, f"{sinks.N} sinks at the end, the binary should stay two sinks with the Federrath checks")
    print("PASS: Boss & Bodenheimer: the m = 2 perturbation forms two sinks")
    return dict(sim=sim, gas=gas, sinks=sinks, events=events, output_dir=output_dir, t_ff=code(t_ff, u.Gyr), h=h, R=R)


if __name__ == "__main__":
    main()
