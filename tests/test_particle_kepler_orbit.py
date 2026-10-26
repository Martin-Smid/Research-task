"""Test A: časová přesnost integrátoru N-body částic (NBody.drift) na kruhové dráze kolem sinku.

Uspořádání
    Jedna testovací částice (zanedbatelné hmotnosti) obíhá kolem sinku uprostřed boxu. Pole sinku je statické,
    takže se měří jen CHYBA ČASOVÉ INTEGRACE: dráha z NBody.drift se porovná s téměř přesnou dráhou (RK4 s krokem
    64x menším než nejmenší testované h) v TOMTÉŽ interpolovaném poli. Chyba pole samotného (periodický box,
    rozlišení, změkčení sinku) se sem nepromítá, patří do samostatného testu.

Tři kontroly
    1) NBody.drift s jedním podkrokem na okno je přesně učebnicové DKD (leapfrog).
    2) Čisté DKD konverguje s krokem h průměrně aspoň řádem MIN_MEAN_ORDER (teorie: 2). Konkrétní řád mezi
       dvěma sousedními h kolísá (1.7-2.6), protože interpolovaná síla je jen po částech lineární (C0).
    3) Produkční nastavení (NBody.drift si sám volí počet podkroků) drží chybu polohy pod MAX_ERROR_PRODUCTION * r0.
       Poznámka: chyba v produkci skoro nezávisí na h, protože počet podkroků určuje kritérium
       0.25 * dx / v, ne délka okna.

Sink v Poissonově rovnici nevystupuje jako hustota, jeho potenciál je analytický (k-prostor), takže částice cítí
přesně totéž pole jako ve skutečné simulaci (forces_total = prostředí + sink)."""

from contextlib import contextmanager
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# PARAMETRY TESTU (jednotky: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # box je [-BOX, BOX]^3 [kpc]
N_GRID = 128                        # rozlišení mřížky (buňka 0.156 kpc)
M_SINK = 5.0e10                     # hmotnost centrálního sinku [Msun]
M_PARTICLE = 1.0                    # hmotnost testovací částice [Msun] (zanedbatelná, aby netáhla zdroj)
R0 = 5.0                            # poloměr startu [kpc] (daleko od stěn boxu i od změkčení sinku)
N_ORBITS = 3                        # kolik oběhů se integruje
STEPS_PER_ORBIT = [20, 40, 80, 160, 320]    # kroky na oběh pro testované h (každé je polovina předchozího)
REF_STEPS_PER_ORBIT = 2560                  # krok referenčního RK4 (dělitelný všemi výše)
MIN_MEAN_ORDER = 1.7                # nejmenší povolený průměrný řád čistého DKD
MAX_ERROR_PRODUCTION = 2.0e-3       # největší povolená chyba polohy v produkci [násobky r0]
MAX_DIFF_TEXTBOOK = 1.0e-9          # největší povolený rozdíl drift() proti učebnicovému DKD [kpc]


def build_system():
    """Sink uprostřed + silová mřížka jeho pole. Vrací (sim, (Fx, Fy, Fz))."""

    sim = Simulation_Class(

        dim=3,                              # 3D simulace
        boundaries=[(-BOX, BOX)] * 3,       # box, střed je v (0, 0, 0)
        N=N_GRID,                           # rozlišení mřížky
        total_time=0.01,                    # nepoužije se, evolve() nevoláme
        h=0.001,                            # nepoužije se
        order_of_evolution=2,               # řád evoluce
        use_gravity=True,                   # zapnutá gravitace
        use_units=True,                     # kpc, Msun, Gyr
        use_sponge=False,                   # bez houby
        self_int=False,                     # bez samointerakce

    )

    sink = SinkNBody(

        simulation=sim,                             # simulace, do které sink patří
        N_sinks=1,                                  # jeden sink
        initial_masses=[M_SINK],                    # jeho hmotnost [Msun]
        initial_positions=[[0.0, 0.0, 0.0]],        # uprostřed boxu
        initial_velocities=[[0.0, 0.0, 0.0]],       # v klidu

    )
    sim.add_baryons(sink)
    sim.initialize_simulation()

    phi_sink = sim.evolution._compute_sink_potential_analytic_kspace()      # analytický potenciál sinku
    forces = sim.propagator.compute_force_grids_from_potential(phi_sink)    # a = -grad(phi) na mřížce
    return sim, forces


def build_particle(sim, position, velocity):
    """Jedna testovací částice na dané poloze a s danou rychlostí."""

    particle = Baryons(

        simulation=sim,                 # simulace (mřížka, hranice)
        N_particles=1,                  # jedna částice
        total_mass=M_PARTICLE,          # zanedbatelná hmotnost
        init_profile="uniform",         # profil nehraje roli, polohu i rychlost hned přepíšeme

    )
    particle.positions = cp.asarray([position], dtype=cp.float64)
    particle.velocities = cp.asarray([velocity], dtype=cp.float64)
    particle.invalidate_density_cache()
    return particle


def acceleration(particle, forces, position):
    """Zrychlení v bodě `position` stejnou trilineární interpolací, jakou používá NBody.drift."""

    particle.positions = cp.asarray([position], dtype=cp.float64)
    return cp.asnumpy(particle._interpolate_force_trilinear(*forces))[0]


def circular_velocity(particle, forces, radius):
    """Kruhová rychlost podle síly, kterou částice na mřížce SKUTEČNĚ cítí (ne podle analytického GM/r)."""

    a = acceleration(particle, forces, [radius, 0.0, 0.0])
    return float(np.sqrt(radius * abs(a[0])))


@contextmanager
def single_substep(sim):
    """NBody.drift dělí okno na podkroky podle kritéria 0.25 * min(dx) / v. S obřím dx vyjde vždy jeden podkrok,
    takže drift() provede přesně jeden DKD krok délky h (jen tak jde měřit řád v h)."""

    real_dx = sim.dx
    sim.dx = [1.0e3] * 3
    try:
        yield
    finally:
        sim.dx = real_dx


def reference_orbit(sim, forces, v0, dt, n_steps):
    """Téměř přesná dráha: klasický RK4 ve stejném interpolovaném poli. Vrací polohy (n_steps + 1, 3)."""

    probe = build_particle(sim, [R0, 0.0, 0.0], [0.0, v0, 0.0])

    def derivative(state):
        return np.concatenate([state[3:], acceleration(probe, forces, state[:3])])

    state = np.array([R0, 0.0, 0.0, 0.0, v0, 0.0])
    positions = [state[:3].copy()]
    for _ in range(n_steps):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        positions.append(state[:3].copy())
    return np.array(positions)


def textbook_dkd_orbit(sim, forces, v0, h, n_steps):
    """Učebnicové DKD (půl driftu, celý kick, půl driftu) na CPU. Vrací polohy (n_steps + 1, 3)."""

    probe = build_particle(sim, [R0, 0.0, 0.0], [0.0, 0.0, 0.0])
    x = np.array([R0, 0.0, 0.0])
    v = np.array([0.0, v0, 0.0])
    positions = [x.copy()]
    for _ in range(n_steps):
        x = x + v * h / 2.0
        v = v + acceleration(probe, forces, x) * h
        x = x + v * h / 2.0
        positions.append(x.copy())
    return np.array(positions)


def code_orbit(sim, forces, v0, h, n_steps):
    """Dráha z NBody.drift: n_steps oken po h. Vrací polohy (n_steps + 1, 3)."""

    particle = build_particle(sim, [R0, 0.0, 0.0], [0.0, v0, 0.0])
    positions = [cp.asnumpy(particle.positions)[0].copy()]
    for step in range(n_steps):
        particle.drift(dt=h, potential_grid=forces, first_step=(step == 0), last_step=(step == n_steps - 1))
        positions.append(cp.asnumpy(particle.positions)[0].copy())
    return np.array(positions)


def max_error(orbit, reference, steps_per_orbit):
    """Největší vzdálenost od reference (v násobcích r0); reference se vzorkuje ve stejných časech jako orbit."""

    stride = REF_STEPS_PER_ORBIT // steps_per_orbit
    return float(np.linalg.norm(orbit - reference[::stride][: len(orbit)], axis=1).max() / R0)


def main():
    sim, forces = build_system()
    probe = build_particle(sim, [R0, 0.0, 0.0], [0.0, 0.0, 0.0])

    v0 = circular_velocity(probe, forces, R0)
    period = 2.0 * np.pi * R0 / v0
    print(f"sink {M_SINK:.1e} Msun, r0 = {R0} kpc, N = {N_GRID}, v_circ(mřížka) = {v0:.2f} kpc/Gyr, "
          f"perioda = {period:.5f} Gyr, {N_ORBITS} oběhů")

    reference = reference_orbit(sim, forces, v0, period / REF_STEPS_PER_ORBIT, N_ORBITS * REF_STEPS_PER_ORBIT)
    radius = np.linalg.norm(reference, axis=1)
    print(f"referenční dráha: r v rozsahu [{radius.min():.4f}, {radius.max():.4f}] kpc "
          f"(pole na mřížce není přesně centrální, dráha tedy není přesný kruh)")

    # --- 1) + 2) čisté DKD: jeden podkrok na okno
    print("\nČisté DKD (kritérium podkroků vypnuto):")
    print(f"{'kroků/oběh':>11} {'h [Gyr]':>10} {'max chyba polohy / r0':>24} {'řád':>6}")
    errors = []
    with single_substep(sim):
        for steps_per_orbit in STEPS_PER_ORBIT:
            h = period / steps_per_orbit
            n_steps = N_ORBITS * steps_per_orbit
            orbit = code_orbit(sim, forces, v0, h, n_steps)
            errors.append(max_error(orbit, reference, steps_per_orbit))
            order = np.log2(errors[-2] / errors[-1]) if len(errors) > 1 else float("nan")
            print(f"{steps_per_orbit:>11} {h:>10.6f} {errors[-1]:>24.3e} {order:>6.2f}")

        check_steps = 80
        textbook = textbook_dkd_orbit(sim, forces, v0, period / check_steps, N_ORBITS * check_steps)
        from_code = code_orbit(sim, forces, v0, period / check_steps, N_ORBITS * check_steps)
    diff = float(np.abs(textbook - from_code).max())
    mean_order = float(np.log2(errors[0] / errors[-1]) / (len(errors) - 1))
    print(f"průměrný řád přes {len(errors) - 1} půlení kroku: {mean_order:.2f}")
    print(f"největší rozdíl drift() (1 podkrok) vs učebnicové DKD: {diff:.2e} kpc")

    # --- 3) produkční nastavení
    print("\nProdukční NBody.drift (počet podkroků volí kritérium):")
    print(f"{'kroků/oběh':>11} {'h [Gyr]':>10} {'max chyba polohy / r0':>24}")
    production = []
    for steps_per_orbit in (40, 80, 160):
        h = period / steps_per_orbit
        orbit = code_orbit(sim, forces, v0, h, N_ORBITS * steps_per_orbit)
        production.append(max_error(orbit, reference, steps_per_orbit))
        print(f"{steps_per_orbit:>11} {h:>10.6f} {production[-1]:>24.3e}")

    assert diff < MAX_DIFF_TEXTBOOK, f"drift() s jedním podkrokem není učebnicové DKD (rozdíl {diff:.2e} kpc)"
    assert mean_order >= MIN_MEAN_ORDER, f"průměrný řád čistého DKD je {mean_order:.2f}, očekává se >= {MIN_MEAN_ORDER}"
    assert max(production) < MAX_ERROR_PRODUCTION, (
        f"produkční chyba polohy {max(production):.2e} r0 je nad limitem {MAX_ERROR_PRODUCTION:.1e}")
    print(f"\nPASS: drift() je učebnicové DKD (rozdíl {diff:.1e} kpc), průměrný řád {mean_order:.2f}, "
          f"produkční chyba polohy {max(production):.1e} r0")


if __name__ == "__main__":
    main()
