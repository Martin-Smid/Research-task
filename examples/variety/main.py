import cupy as cp
import numpy as np

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody


# =====================================================================================================
# ZJEDNODUŠENÁ TRPASLIČÍ GALAXIE SE ČTYŘMI SLOŽKAMI (jednotky: kpc, Msun, Gyr), dvě fáze
#
#   FÁZE 1 - RELAXACE (RELAX_TIME):  soliton + hvězdy. Hvězdy se do solitonu zapínají POMALU (jejich hmotnost
#            roste z nuly na plnou hodnotu), aby soliton stihl sledovat měnící se potenciál. Kdyby hvězdy
#            byly od začátku v plné hmotnosti, soliton by "dýchal" (±13 % hustoty jádra s periodou ~0.75 Gyr)
#            a to dýchání nemá žádné tření, takže by se samo neutlumilo.
#   FÁZE 2 - GALAXIE (RUN_TIME):     z konce fáze 1 se udělá nová simulace (from_checkpoint + start_new_segment),
#            přidá se plyn a černá díra. Rotace plynu se spočte z gravitace, kterou plyn skutečně cítí.
#
# Hmotnosti všech složek jsou zadané JAKO POMĚR k hmotnosti solitonu.
# =====================================================================================================

# ---------------------------------------------------- hmotnosti (odpichuje se od solitonu)
M_SOLITON = 53090072.0            # hmotnost solitonu [Msun]  (z profilu GroundState(1).dat)
F_STARS = 0.10                    # hvězdy = 10 % hmotnosti solitonu
F_GAS = 0.10                      # plyn   = 10 % hmotnosti solitonu (plyn : hvězdy = 1 : 1, plynem bohatý trpaslík)
F_BH = 0.004                      # černá díra = 0.4 % hmotnosti solitonu (~4 % hvězdné hmotnosti, IMBH)

M_STARS = F_STARS * M_SOLITON     # [Msun]
M_GAS = F_GAS * M_SOLITON         # [Msun]
M_BH = F_BH * M_SOLITON           # [Msun]

# ---------------------------------------------------- struktura disků
N_STARS = 100_000                 # počet hvězdných částic (hmotnost částice = M_STARS / N_STARS)
R_STARS = 1.2                     # škálová délka hvězdného disku [kpc]
R_STARS_IN = 0.5                  # uvnitř tohoto poloměru nejsou hvězdy (jinak by je černá díra hned snědla) [kpc]
R_GAS = 2.0                       # škálová délka plynového disku [kpc] (plyn je rozlehlejší než hvězdy)
CS = 8.0                          # rychlost zvuku plynu [kpc/Gyr]  (8 kpc/Gyr = 7.8 km/s)
RHO_BG = 2.0                      # hustota řídkého plynu kolem disku [Msun/kpc^3] (absolutní vakuum solver nemá rád)
SIGMA_R_FRAC = 0.15               # radiální disperze hvězd jako zlomek kruhové rychlosti
SIGMA_Z_FRAC = 0.6                # svislá disperze hvězd jako zlomek radiální

# ---------------------------------------------------- simulace a fáze
N = 128                           # rozlišení mřížky (128 buněk na osu, buňka 0.156 kpc)
H = 5.0e-4                        # časový krok [Gyr]
RELAX_TIME = 2.0                  # FÁZE 1: délka relaxace [Gyr]
RAMP_STARS = True                 # hvězdy se do solitonu zapínají pomalu (True) / jsou od začátku v plné hmotnosti (False)
RAMP_TIME = 2.0                   # za jak dlouho hvězdy dorostou na plnou hmotnost [Gyr] (<= RELAX_TIME); delší = adiabatičtější
RUN_TIME = 1.0                    # FÁZE 2: délka simulace s plynem a černou dírou [Gyr]
RELAXED_DIR = None                # None = spustit fázi 1; nebo cesta ke hotové relaxaci "resources/data/simulation_..." (fáze 1 se přeskočí)
SAVE_EVERY_RELAX = 500            # snapshot ve fázi 1 každých 500 kroků = 0.25 Gyr
SAVE_EVERY_RUN = 200              # snapshot ve fázi 2 každých 200 kroků = 0.1 Gyr
DIAGNOSTICS_EVERY = 20            # energie se zapisuje každých 20 kroků
SEED = 42

rng = np.random.default_rng(SEED)


# =====================================================================================================
# POMOCNÉ FUNKCE (měření gravitace na mřížce, stavba hvězd a plynu)
# =====================================================================================================
def measure_gravity(sim, edges):
    """Změří v rovině disku (z = 0) z gravitace na mřížce: kruhovou rychlost v_c(R) a svislou frekvenci nu(R).
    Gravitaci dělá vše, co v simulaci právě je (soliton, hvězdy, případně plyn a černá díra)."""
    evo = sim.evolution
    gx, gy, _ = (cp.asnumpy(g) for g in sim.grids)
    Rg = np.hypot(gx, gy)
    k0 = sim.N // 2  # index roviny z = 0
    centres = 0.5 * (edges[1:] + edges[:-1])
    phi_env, phi_sink = evo._compute_gravity_fields(evo._compute_total_density(sim.wave_functions))
    fx, fy, fz = (cp.asnumpy(f) for f in sim.propagator.compute_force_grids_from_potential(phi_env + phi_sink))
    a_in = -(gx * fx + gy * fy)[:, :, k0] / np.maximum(Rg[:, :, k0], 1e-9)  # zrychlení směrem do středu
    nu2 = -(fz[:, :, k0 + 1] - fz[:, :, k0 - 1]) / (2.0 * sim.dx[2])  # nu^2 = -dFz/dz
    ring = np.digitize(Rg[:, :, k0], edges) - 1
    v_c, nu = np.zeros(len(centres)), np.zeros(len(centres))
    for i, r in enumerate(centres):
        sel = ring == i
        if np.any(sel):
            v_c[i] = np.sqrt(max(r * a_in[sel].mean(), 0.0))
            nu[i] = np.sqrt(max(nu2[sel].mean(), 1e-6))
    return centres, v_c, nu


def set_star_velocities(stars, z_unit, centres, v_c, nu):
    """Hvězdám nastaví rychlosti: kruhová rychlost zmenšená o asymetrický drift (Jeansova rovnice) + disperze;
    svislou šířku disku z rovnováhy sigma_z / nu."""
    pos = cp.asnumpy(stars.positions)
    Rs = np.hypot(pos[:, 0], pos[:, 1])
    vc_s = np.interp(Rs, centres, v_c)
    nu_s = np.interp(Rs, centres, nu)
    sigma_R = SIGMA_R_FRAC * vc_s
    sigma_z = SIGMA_Z_FRAC * sigma_R
    omega = vc_s / Rs
    dvc = np.gradient(v_c, centres)[np.clip(np.digitize(Rs, EDGES) - 1, 0, len(centres) - 1)]
    kappa2 = 2.0 * omega * (dvc + omega)
    v_phi = np.sqrt(np.maximum(vc_s ** 2 - sigma_R ** 2 * (Rs / R_STARS - 1.0 + kappa2 / (4.0 * omega ** 2)), 0.0))
    pos[:, 2] = z_unit * sigma_z / nu_s  # svislá šířka disku
    vel = np.column_stack((-v_phi * pos[:, 1] / Rs, v_phi * pos[:, 0] / Rs, np.zeros(len(Rs))))
    vel[:, :2] += rng.normal(size=(len(Rs), 2)) * sigma_R[:, None]  # radiální + azimutální disperze
    vel[:, 2] += rng.normal(size=len(Rs)) * sigma_z
    stars.positions = cp.asarray(pos)
    stars.velocities = cp.asarray(vel)


def smoothstep(x):
    """Hladký přechod 0 -> 1 s nulovou derivací na obou koncích."""
    x = min(max(x, 0.0), 1.0)
    return 3.0 * x ** 2 - 2.0 * x ** 3


EDGES = np.linspace(0.3, 9.5, 47)  # poloměrové slupky pro měření gravitace [kpc]


# =====================================================================================================
# FÁZE 1: RELAXACE (soliton + hvězdy)
# =====================================================================================================
if RELAXED_DIR is None:

    # --- 1) simulace + soliton
    sim = Simulation_Class(

        dim=3, # 3D simulace
        boundaries=[(-10, 10)]*3, # box 20 kpc, střed galaxie je v (0,0,0)
        N=N, # rozlišení mřížky
        total_time=RELAX_TIME, # jak dlouho relaxovat [Gyr]
        h=H, # časový krok [Gyr]
        order_of_evolution=2, # plyn podporuje jen řád 2
        use_gravity=True, # zapnutá gravitace
        static_potential=None,
        save_max_vals=True, # ukládá maximální hustotu v čase
        m_s=2.5e-22, # hmotnost částice ULDM [eV]
        use_sponge=False,
        self_int=False,

    )
    dx = min(sim.dx)  # velikost buňky [kpc]

    wave_vector = Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat", # profil základního stavu solitonu
        means=[0, 0, 0], # střed
        st_deviations=[0.5, 0.5, 0.5],
        simulation=sim,
        mass=1,
        omega=1,
        momenta=[0, 0.0, 0],
        spin=0,
        desired_soliton_mass=M_SOLITON, # hmotnost solitonu [Msun]

    )
    sim.add_wave_vector(wave_vector)

    # --- 2) hvězdy: polohy částic (rychlosti se doplní podle gravity na mřížce)
    #        poloměr: exponenciální disk Sigma ~ exp(-R/R_STARS), oříznutý zevnitř (R_STARS_IN) a zvenku (7 kpc)
    r_grid = np.linspace(R_STARS_IN, 7.0, 20001)
    cdf = np.cumsum(r_grid * np.exp(-r_grid / R_STARS))
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    R_p = np.interp(rng.uniform(size=N_STARS), cdf, r_grid)
    phi_p = rng.uniform(0.0, 2.0 * np.pi, N_STARS)
    z_unit = rng.normal(size=N_STARS)  # svislé polohy: normální rozdělení, jeho šířka se nastaví z gravitace

    stars = Baryons(

        simulation=sim,
        N_particles=N_STARS, # počet částic
        total_mass=M_STARS, # celková hmotnost hvězd
        init_profile="disk", # (polohy i rychlosti si stejně přepíšeme)
        radius=R_STARS,
        center=(0.0, 0.0, 0.0),

    )
    stars.positions = cp.asarray(np.column_stack((R_p * np.cos(phi_p), R_p * np.sin(phi_p), 0.3 * z_unit)))
    stars.invalidate_density_cache()

    sim.add_baryons(stars)
    sim.initialize_simulation()
    evo = sim.evolution
    for wf in sim.wave_functions:
        wf.psi = cp.asarray(wf.psi)  # vlnová funkce na GPU (evolve() to dělá taky; potřebujeme to už teď kvůli měření)

    # --- 3) rychlosti hvězd podle gravity na mřížce (soliton + hvězdy v plné hmotnosti)
    centres, v_c, nu = measure_gravity(sim, EDGES)
    set_star_velocities(stars, z_unit, centres, v_c, nu)

    # --- 4) pomalé zapínání hvězd: hmotnost částice roste z nuly na plnou hodnotu (hladce, přes smoothstep)
    if RAMP_STARS:
        m_full = M_STARS / N_STARS
        _step = evo._perform_evolution_step

        def step_with_growing_stars(wave_functions, total_density, step, save_step):
            stars.m_particle = m_full * smoothstep(step * H / RAMP_TIME)
            stars.invalidate_density_cache()
            return _step(wave_functions, evo._compute_total_density(wave_functions), step, save_step)

        evo._perform_evolution_step = step_with_growing_stars

    # --- 5) relaxace
    print(f"\nFÁZE 1: relaxace solitonu s hvězdami, {RELAX_TIME} Gyr ({'hvězdy pomalu rostou' if RAMP_STARS else 'hvězdy v plné hmotnosti od začátku'})")
    evo.evolve(sim.wave_functions, save_every=SAVE_EVERY_RELAX, diagnostics_every=DIAGNOSTICS_EVERY)
    RELAXED_DIR = sim.evolution.scribe.snapshot_directory
    print(f"Relaxace uložena v: {RELAXED_DIR}")


# =====================================================================================================
# FÁZE 2: GALAXIE (nová simulace z konce relaxace + plyn + černá díra)
# =====================================================================================================
sim = Simulation_Class.from_checkpoint(RELAXED_DIR)  # načte stav z konce fáze 1 (soliton i hvězdy)
sim.start_new_segment(total_time=RUN_TIME, h=H, save_max_vals=True)  # nová simulace, stav z konce relaxace je počátek
dx = min(sim.dx)
stars = sim.baryonic_matter[0]  # hvězdy načtené z relaxace

gx, gy, gz = (cp.asnumpy(g) for g in sim.grids)
Rg = np.hypot(gx, gy)

# --- 1) plyn: zatím prozatímní hustota (správná hydrostatická výška se spočte z gravity níže)
rho_gas = np.exp(-Rg / R_GAS) * np.exp(-0.5 * (gz / 0.3) ** 2)
rho_gas *= M_GAS / (rho_gas.sum() * sim.dV)
rho_gas += RHO_BG
zeros = np.zeros_like(rho_gas)

gas = NBodyGas(

    simulation=sim,
    rho=rho_gas, # hustota
    vx=zeros, # rychlosti se doplní později
    vy=zeros,
    vz=zeros,
    cs=CS, # rychlost zvuku
    gamma=5.0 / 3.0, # adiabatický index
    cfl=0.4, # CFL číslo plynu
    max_substeps=50, # kolik podkroků plynu smí vzniknout na jeden krok
    name="gas",

)

# --- 2) černá díra uprostřed; pojídá plyn a hvězdy v okolí capture_radius
sink = SinkNBody(

    simulation=sim,
    N_sinks=1, # jedna černá díra
    initial_masses=[M_BH], # její hmotnost
    initial_positions=[[0.0, 0.0, 0.0]], # střed galaxie (a solitonu)
    initial_velocities=[[0.0, 0.0, 0.0]], # na začátku stojí
    capture_radius=2.5 * dx, # do této vzdálenosti pojídá plyn a hvězdy
    softening_length=1.5 * dx, # změkčení gravitace

)

sim.add_baryons(gas)
sim.add_baryons(sink)
sim.sink_accretion = True  # černá díra pojídá plyn a hvězdy (ve výchozím stavu je akrece vypnutá)
sim.initialize_simulation()
evo = sim.evolution
for wf in sim.wave_functions:
    wf.psi = cp.asarray(wf.psi)

# --- 3) rovnováha plynu: změříme gravitaci (soliton po relaxaci + hvězdy + černá díra) a podle ní plyn roztočíme
centres, v_c, nu = measure_gravity(sim, EDGES)
c_iso2 = CS ** 2 / (5.0 / 3.0)  # izotermní rychlost zvuku na druhou
height = lambda r: np.maximum(np.sqrt(c_iso2) / np.interp(r, centres, nu), 1.5 * dx)  # hydrostatická výška H = c / nu
Hz = height(Rg)
rho_new = np.exp(-Rg / R_GAS) / Hz * np.exp(-0.5 * (gz / Hz) ** 2)
rho_new *= M_GAS / (rho_new.sum() * sim.dV)
gas.rho = cp.asarray(rho_new + RHO_BG)
centres, v_c, nu = measure_gravity(sim, EDGES)  # znovu, už s konečnou hustotou plynu

# rotace plynu = kruhová rychlost upravená o radiální gradient tlaku, v^2 = v_c^2 + c^2 dln(rho)/dln(R)
Rf = np.linspace(0.05, 12.0, 4000)
dlnrho_dlnR = Rf * np.gradient(-Rf / R_GAS - np.log(height(Rf)), Rf)
v_gas = np.sqrt(np.maximum(np.interp(Rg, centres, v_c) ** 2 + c_iso2 * np.interp(Rg, Rf, dlnrho_dlnR), 0.0))
Rn = np.maximum(Rg, 1e-6)
gas.vx = cp.asarray(-v_gas * gy / Rn)
gas.vy = cp.asarray(v_gas * gx / Rn)
gas.vz = cp.zeros_like(gas.rho)
gas.vel = [gas.vx, gas.vy, gas.vz]
gas.E = gas.rho * CS ** 2 / (gas.gamma * (gas.gamma - 1.0)) + 0.5 * gas.rho * (gas.vx ** 2 + gas.vy ** 2)

# --- 4) přehled, co jsme postavili
print("\n=========== SOUHRN POČÁTEČNÍCH PODMÍNEK FÁZE 2 ===========")
print(f"soliton {M_SOLITON:.4e} Msun | hvězdy {M_STARS:.3e} ({F_STARS:.0%}) | plyn {M_GAS:.3e} ({F_GAS:.0%}) | černá díra {M_BH:.3e} ({F_BH:.1%})")
print("R [kpc]   v_c [kpc/Gyr]   nu [1/Gyr]   Toomre Q plyn")
for r0 in (1.0, 2.0, 3.0, 4.0, 5.0):
    v, n_ = np.interp(r0, centres, v_c), np.interp(r0, centres, nu)
    kap = np.sqrt(max(2 * v / r0 * (np.interp(r0, centres, np.gradient(v_c, centres)) + v / r0), 1e-9))
    sig_g = M_GAS / (2 * np.pi * R_GAS ** 2) * np.exp(-r0 / R_GAS)
    print(f"{r0:5.1f}   {v:10.2f}   {n_:10.2f}   {np.sqrt(c_iso2) * kap / (np.pi * sim.G * sig_g):12.1f}")
print("==========================================================\n")

# --- 5) spuštění
print(f"FÁZE 2: galaxie se čtyřmi složkami, {RUN_TIME} Gyr")
evo.evolve(sim.wave_functions, save_every=SAVE_EVERY_RUN, diagnostics_every=DIAGNOSTICS_EVERY)

print(sim.evolution.scribe.snapshot_directory)
print(f"černá díra: {float(sink.masses.sum()):.4e} Msun (na začátku {M_BH:.4e}), plyn: {float(cp.sum(gas.rho)) * sim.dV:.4e} Msun (na začátku {M_GAS:.4e})")
