"""Test C: testovací částice v časově proměnném zdroji (dýchající soliton), samokonvergence v kroku h.

Proč
    Částice se v jednom kroku pohybují v poli spočteném z hustoty na ZAČÁTKU kroku (zamrzlý potenciál, viz
    Evolution._evolve_order_2). Když se zdroj mění, tahle prodleva dává chybu prvního řádu v h. Přesné řešení
    neexistuje, proto se dráha při h porovná s dráhou při nejjemnějším h (Richardson).

Uspořádání
    Soliton (5.3e7 Msun) s vlnovou funkcí vynásobenou SCALE: není v rovnováze, dýchá. Za T_END = 0.2 Gyr se
    ρ_max zvětší asi 3x (první, stlačující fáze dýchání), zdroj se tedy výrazně mění.
    Testovací částice zanedbatelné hmotnosti obíhá uvnitř jádra (r0 = 1 kpc), kde se enclosed mass mění.
    Mimo sférický soliton by se pole neměnilo (větu o slupkách), proto uvnitř.

Kontroly
    1) Chyba dráhy klesá s h aspoň průměrným řádem MIN_MEAN_ORDER. Vůči nejjemnějšímu h vychází pro řád p = 1
       zdánlivý řád 1.2 a 1.6, pro p = 2 by to bylo 2.1 a 2.3 (finest sám není přesný).
    2) Chyba polohy při h = 4e-4 je pod MAX_ERROR (násobky r0).
    Naměřeno (N = 64, 0.2 Gyr): zdánlivý řád 1.2 a 1.6, chyba 1.3e-3 r0 při h = 4e-4. Zdroj se mění asi 3x; u skutečné
    galaxie (±13 % hustoty) čekáme chybu o řád menší (neměřeno)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from test_particle_kepler_orbit import circular_velocity


# ---------------------------------------------------------------------------------------------------------
# PARAMETRY TESTU (jednotky: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0 / 2.0                    # box je [-BOX, BOX]^3 [kpc]
N_GRID = 64                         # rozlišení mřížky (buňka 0.156 kpc)
M_SOLITON = 53090072.0              # hmotnost solitonu [Msun]
SCALE = 1.15                        # násobek vlnové funkce: soliton není v rovnováze a dýchá
R0 = 1.0                            # poloměr startu částice [kpc] (uvnitř jádra)
M_PARTICLE = 1.0                    # hmotnost testovací částice [Msun] (zanedbatelná)
T_END = 0.2                         # délka integrace [Gyr]
H_LIST = [4.0e-4, 2.0e-4, 1.0e-4]   # testovaná h (mezní stabilní h pro tuto mřížku je 6.6e-4)
H_REFERENCE = 5.0e-5                # nejjemnější h, vůči kterému se chyba měří
MIN_MEAN_ORDER = 0.9                # nejmenší povolený průměrný zdánlivý řád (zamrzlý potenciál: ~1.4)
MAX_ERROR = 5.0e-3                  # největší povolená chyba polohy při h = H_LIST[0] [násobky r0]


def run_orbit(h):
    """Spustí soliton + testovací částici s krokem h a vrátí polohy částice po každém kroku (n_steps + 1, 3)."""

    sim = Simulation_Class(

        dim=3,                              # 3D simulace
        boundaries=[(-BOX, BOX)] * 3,       # box, střed je v (0, 0, 0)
        N=N_GRID,                           # rozlišení mřížky
        total_time=T_END,                   # jak dlouho integrovat [Gyr]
        h=h,                                # časový krok [Gyr]
        order_of_evolution=2,               # řád evoluce
        use_gravity=True,                   # zapnutá gravitace
        use_units=True,                     # kpc, Msun, Gyr
        use_sponge=False,                   # bez houby
        self_int=False,                     # bez samointerakce
        save_max_vals=False,                # nic neukládat
        m_s=2.5e-22,                        # hmotnost částice ULDM [eV]

    )

    wave_vector = Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat",    # profil základního stavu solitonu
        means=[0, 0, 0],                                        # střed
        st_deviations=[0.5, 0.5, 0.5],                          # nepoužije se pro profil ze souboru
        simulation=sim,                                         # simulace
        mass=1,                                                 # hmotnost
        omega=1,                                                # frekvence
        momenta=[0, 0.0, 0],                                    # bez hybnosti
        spin=0,                                                 # skalární pole
        desired_soliton_mass=M_SOLITON,                         # hmotnost solitonu [Msun]

    )
    sim.add_wave_vector(wave_vector)

    tracer = Baryons(

        simulation=sim,                 # simulace (mřížka, hranice)
        N_particles=1,                  # jedna testovací částice
        total_mass=M_PARTICLE,          # zanedbatelná hmotnost
        init_profile="uniform",         # profil nehraje roli, polohu i rychlost hned přepíšeme

    )
    sim.add_baryons(tracer)
    sim.initialize_simulation()

    evolution = sim.evolution
    wave_function = sim.wave_functions[0]
    wave_function.psi = cp.asarray(wave_function.psi) * SCALE   # nerovnovážný soliton: dýchá

    # kruhová rychlost podle síly, kterou částice na mřížce skutečně cítí na začátku
    density = evolution._compute_total_density(sim.wave_functions)
    phi_environment, phi_sink = evolution._compute_gravity_fields(density)
    forces = sim.propagator.compute_force_grids_from_potential(phi_environment + phi_sink)
    v0 = circular_velocity(tracer, forces, R0)
    tracer.positions = cp.asarray([[R0, 0.0, 0.0]])
    tracer.velocities = cp.asarray([[0.0, v0, 0.0]])
    tracer.invalidate_density_cache()

    # polohy částice po každém kroku (obalení kroku evoluce, stejný trik jako v main.py)
    trajectory = [cp.asnumpy(tracer.positions)[0].copy()]
    original_step = evolution._perform_evolution_step

    def recorded_step(wave_functions, total_density, step, save_step):
        result = original_step(wave_functions, total_density, step, save_step)
        trajectory.append(cp.asnumpy(tracer.positions)[0].copy())
        return result

    evolution._perform_evolution_step = recorded_step
    evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=10 ** 9)
    return np.array(trajectory)


def main():
    reference = run_orbit(H_REFERENCE)
    print(f"soliton {M_SOLITON:.2e} Msun (psi x {SCALE}), r0 = {R0} kpc, {T_END} Gyr, "
          f"reference h = {H_REFERENCE:.1e} ({len(reference) - 1} kroků)")
    print(f"\n{'h [Gyr]':>10} {'kroků':>7} {'max chyba polohy / r0':>24} {'zdánlivý řád':>14}")

    errors = []
    for h in H_LIST:
        orbit = run_orbit(h)
        stride = int(round(h / H_REFERENCE))
        difference = np.linalg.norm(orbit - reference[::stride][: len(orbit)], axis=1)
        errors.append(float(difference.max() / R0))
        order = np.log2(errors[-2] / errors[-1]) if len(errors) > 1 else float("nan")
        print(f"{h:>10.1e} {len(orbit) - 1:>7} {errors[-1]:>24.3e} {order:>14.2f}")

    mean_order = float(np.log2(errors[0] / errors[-1]) / (len(errors) - 1))
    print(f"průměrný zdánlivý řád: {mean_order:.2f} (pro skutečný řád 1 vychází ~1.4, pro řád 2 ~2.2)")

    assert mean_order >= MIN_MEAN_ORDER, f"chyba částice neklesá s h: zdánlivý řád {mean_order:.2f}"
    assert errors[0] < MAX_ERROR, f"chyba polohy {errors[0]:.2e} r0 při h = {H_LIST[0]:.0e} je nad limitem {MAX_ERROR}"
    print(f"PASS: částice v dýchajícím solitonu konverguje (zdánlivý řád {mean_order:.2f}), "
          f"chyba při h = {H_LIST[0]:.0e} je {errors[0]:.1e} r0")


if __name__ == "__main__":
    main()
