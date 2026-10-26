"""Test B: síla sinku na mřížce proti analytickému vzorci (kde je pole věrohodné?).

Co se měří
    Sink uprostřed boxu vytváří potenciál analyticky v k-prostoru (hmotnost rozmazaná Gaussiánem šířky
    sigma = softening_bh = 1.5 dx, periodická krabice, nulový mód odstraněn). Síla na mřížce a = -grad(phi) se
    z něj počítá FFT derivací a pro částice se interpoluje trilineárně (NBody._interpolate_force_trilinear).
    Tady se tato síla porovná s přesným vzorcem na sférách různého poloměru a v mnoha směrech.

Analytická reference (zdroj je Gaussián v periodickém boxu s homogenním neutralizujícím pozadím)
    a_r(r) = G M / r^2 * [erf(x) - 2 x exp(-x^2) / sqrt(pi)]  -  (4 pi / 3) G M r / L^3,    x = r / (sqrt(2) sigma)
    První člen je síla rozmazané hmoty (pro r >> sigma je to Newton GM/r^2), druhý je vliv periodické krabice
    (homogenní pozadí, které odstraňuje k = 0; obrazy zdroje ho pro kubickou mříž dál nemění do řádu (r/L)^5).
    Je to 6.5 % Newtonovy síly při r = L/4, proto se nesmí měřit proti holému GM/r^2.

Co se kontroluje
    Pro poloměry mezi R_MIN_CELLS buňkami a R_MAX_FRACTION * L musí být síla na mřížce v mezích MAX_ERROR
    (největší relativní chyba přes všechny směry) a odchylka od radiálního směru (anizotropie mřížky) pod
    MAX_TANGENTIAL. Navíc se přes všechny směry průměrná chyba (systematický posun síly) musí vejít do MAX_BIAS.
    Vypisuje se i chyba oproti holému Newtonovi, ať je vidět, o kolik se krabice liší.

Co z měření vyšlo (N = 64, 128, 256)
    - Průměrná chyba je ~1e-5 až 1e-4: vzorec je správně a síla sinku na mřížce nemá systematický posun.
    - Největší chyba přes směry u malých r klesá s rozlišením (mřížka, změkčení), od r ~ 4 kpc ale na N nezávisí
      a roste jako (r/L)^4: to je kubická anizotropie periodické mříže (síla míří k rohům boxu), ne chyba kódu.
      Při r = 6 kpc je 3.5 %, při 8 kpc 19 %, takže dál od středu než L/4 už pole sinku není izotropní."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import numpy as np
from scipy.special import erf

from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
from resources.Classes.Simulation_Class import Simulation_Class


# ---------------------------------------------------------------------------------------------------------
# PARAMETRY TESTU (jednotky: kpc, Msun, Gyr)
# ---------------------------------------------------------------------------------------------------------
BOX = 10.0                          # box je [-BOX, BOX]^3 [kpc], L = 2 BOX
M_SINK = 5.0e10                     # hmotnost sinku [Msun]
GRIDS = [64, 128, 256]              # testovaná rozlišení
RADII_KPC = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]    # poloměry sfér, na kterých se síla měří [kpc]
N_DIRECTIONS = 200                  # kolik směrů na každé sféře (rovnoměrně, Fibonacciho spirála)
R_MIN_CELLS = 6.0                   # nejmenší kontrolovaný poloměr [počet buněk]
R_MAX_FRACTION = 0.25               # největší kontrolovaný poloměr [zlomek délky boxu L]
MAX_ERROR = 0.02                    # největší povolená relativní chyba radiální síly (přes všechny směry)
MAX_TANGENTIAL = 0.02               # největší povolený podíl tečné složky síly (anizotropie mřížky)
MAX_BIAS = 2.0e-3                   # největší povolená průměrná chyba přes směry (systematický posun síly)


def build_system(n_grid):
    """Sink uprostřed + silová mřížka jeho pole. Vrací (sim, (Fx, Fy, Fz))."""

    sim = Simulation_Class(

        dim=3,                              # 3D simulace
        boundaries=[(-BOX, BOX)] * 3,       # box, střed je v (0, 0, 0)
        N=n_grid,                           # rozlišení mřížky
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
    return sim, sink, forces


def sphere_directions(n_points):
    """Rovnoměrně rozložené jednotkové směry (Fibonacciho spirála). Tvar (n_points, 3)."""

    index = np.arange(n_points) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / n_points)
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * index
    return np.column_stack((np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)))


def analytic_inward_force(radius, sigma, g_const, box_length):
    """Dovnitř mířící velikost síly: rozmazaný zdroj (Gaussián) minus vliv homogenního pozadí periodického boxu."""

    x = radius / (np.sqrt(2.0) * sigma)
    gauss = g_const * M_SINK / radius ** 2 * (erf(x) - 2.0 * x * np.exp(-x ** 2) / np.sqrt(np.pi))
    background = 4.0 * np.pi / 3.0 * g_const * M_SINK * radius / box_length ** 3
    return gauss - background


def measure_radius(sim, forces, radius, directions):
    """Síla z mřížky na sféře daného poloměru. Vrací (radiální složka dovnitř, podíl tečné složky) pro každý směr."""

    particle = Baryons(

        simulation=sim,                 # simulace (mřížka, hranice)
        N_particles=len(directions),    # jedna testovací částice na směr
        total_mass=1.0,                 # hmotnost nehraje roli, jen čteme sílu
        init_profile="uniform",         # polohy hned přepíšeme

    )
    particle.positions = cp.asarray(radius * directions, dtype=cp.float64)
    force = cp.asnumpy(particle._interpolate_force_trilinear(*forces))
    inward = -np.einsum("ij,ij->i", force, directions)                       # -a . r_hat
    tangential = np.linalg.norm(force + inward[:, None] * directions, axis=1) / np.linalg.norm(force, axis=1)
    return inward, tangential


def main():
    directions = sphere_directions(N_DIRECTIONS)
    box_length = 2.0 * BOX
    worst_error, worst_tangential, worst_bias = 0.0, 0.0, 0.0

    for n_grid in GRIDS:
        sim, sink, forces = build_system(n_grid)
        dx = float(min(sim.dx))
        sigma = float(sink.softening_bh)
        g_const = float(sim.G)
        print(f"\nN = {n_grid}: dx = {dx:.4f} kpc, sigma (změkčení) = {sigma:.4f} kpc, L = {box_length} kpc")
        print(f"{'r [kpc]':>8} {'r/dx':>6} {'střed chyby':>12} {'max |chyba|':>12} {'tečná složka':>13}"
              f" {'proti Newtonovi':>16}  kontrola")
        for radius in RADII_KPC:
            inward, tangential = measure_radius(sim, forces, radius, directions)
            expected = analytic_inward_force(radius, sigma, g_const, box_length)
            error = inward / expected - 1.0
            vs_newton = float(np.mean(inward) / (g_const * M_SINK / radius ** 2) - 1.0)
            checked = R_MIN_CELLS * dx <= radius <= R_MAX_FRACTION * box_length
            print(f"{radius:>8.2f} {radius / dx:>6.1f} {np.mean(error):>+12.2e} {np.abs(error).max():>12.2e}"
                  f" {tangential.max():>13.2e} {vs_newton:>+16.2e}  {'ano' if checked else '-'}")
            if checked:
                worst_error = max(worst_error, float(np.abs(error).max()))
                worst_tangential = max(worst_tangential, float(tangential.max()))
                worst_bias = max(worst_bias, abs(float(np.mean(error))))

    assert worst_bias < MAX_BIAS, f"průměrná chyba síly {worst_bias:.2e} je nad limitem {MAX_BIAS} (systematický posun)"
    assert worst_error < MAX_ERROR, f"síla na mřížce se liší od analytiky až o {worst_error:.2e} (limit {MAX_ERROR})"
    assert worst_tangential < MAX_TANGENTIAL, (
        f"tečná složka síly je až {worst_tangential:.2e} její velikosti (limit {MAX_TANGENTIAL})")
    print(f"\nPASS: síla sinku na mřížce odpovídá analytice (průměrná chyba {worst_bias:.1e}, "
          f"největší chyba {worst_error:.1e}, největší tečná složka {worst_tangential:.1e})")


if __name__ == "__main__":
    main()
