import argparse
import os, random, sys
import numpy as np
from scipy.integrate import odeint
import astropy.units as u
import astropy.constants as const
from multiprocessing import Process
import warnings

warnings.filterwarnings('ignore')

# --- 1. Načtení argumentů z wrapperu ---
parser = argparse.ArgumentParser()
parser.add_argument("--M_BH", type=float, default=1e4)
parser.add_argument("--m_s", type=float, default=1e-21)
args = parser.parse_args()

M_BH = args.M_BH
m_s_val = args.m_s
m_s_eV = args.m_s * u.eV

# --- 2. Fyzikální konstanty ---
mass_s = (m_s_eV / const.c**2).to(u.Msun)
G_KPC3_MSUN_GYR2 = 4.4986381856937695e-6

c = const.c.to(u.kpc / u.Gyr)
G = const.G.to(u.kpc**3 / (u.Msun * u.Gyr**2))
h_bar = const.hbar.to(u.Msun * u.kpc**2 / u.Gyr)

h_bar_tilde = (h_bar / mass_s).to(u.kpc**2 / u.Gyr)
HBAR_TILDE = h_bar_tilde.value

EPS_KPC = 1e-10
As = 0.0

# --- 3. Solver ---
def gpp(y, r, omega, a_s=As):
    Phi, U, A, B = y
    rr = r if r > EPS_KPC else EPS_KPC
    Ubh = -G_KPC3_MSUN_GYR2 * M_BH / (HBAR_TILDE**2 * rr)
    Utot = U + Ubh

    if r == 0:
        dydr = [A, B, 2*(Utot + a_s*Phi**2 - omega)*Phi/3.0, 4*np.pi*Phi**2/3.0]
    else:
        dydr = [A, B, 2*(Utot + a_s*Phi**2 - omega)*Phi - (2.0/r)*A, 4*np.pi*Phi**2 - (2.0/r)*B]
    return dydr

def Sol(r, U0, omega):
    y0 = [1.0, U0, 0.0, 0.0]
    sol = odeint(gpp, y0, r, args=(omega,), mxstep=5000)
    return sol

# --- 4. Hledání uzlů (Každé jádro dělá tohle) ---
def NodeFinder(U0, omega, rn, U0lim, omegalim, rnlims, filename):
    r = np.linspace(EPS_KPC, rn, num=1000)
    U0 = -1.0
    max_tries = 2000
    tries = 0

    Solution = Sol(r, U0, omega)
    Phi0 = abs(Solution[0,0])

    while ((np.abs(Solution[1:,0]) > 1.02*Phi0).any() or np.abs(Solution[-1,2]) > 1e-2) and tries < max_tries:
        rn = np.random.normal(rn, 0.1)
        while rn > rnlims[1] or rn < rnlims[0]:
            rn = np.random.normal(rn, 0.1)

        r = np.linspace(EPS_KPC, rn, num=1000)
        omega = np.random.normal(omega, 0.1)
        while omega > omegalim[1] or omega < omegalim[0]:
            omega = np.random.normal(omega, 0.1)

        Solution = Sol(r, U0, omega)
        Phi0 = abs(Solution[0,0])
        tries += 1

    if tries >= max_tries:
        return

    # Úspěch! Vytiskneme zprávu (kterou chytí tvůj wrapper) a zapíšeme do DYNAMICKÉHO souboru
    print(f"Hurrah! U0={U0:.3e}, omega={omega:.3e}, rn={rn:.3e}", flush=True)
    with open(filename, "a") as f:
        f.write(f"{U0:.16e} {omega:.16e} {rn:.16e}\n")


# --- 5. Hlavní blok (Pouští se jen jednou v hlavním programu) ---
if __name__ == '__main__':
    Ncores = os.cpu_count() or 8
    rn_lims = [0.2, 4.0]
    U0_lims = [-20, -4]
    omega_lims = [0.0, 0.5]

    filename = f"Sols_{M_BH:.1e}_{m_s_val:.1e}.txt"
    
    # 5A. HLAVIČKA
    with open(filename, "w") as f:
        f.write("#U0 omega rn\n")
        
    print(f"Hledam pro: M_BH={M_BH:.1e}, m_s={m_s_val:.1e} eV, HBAR={HBAR_TILDE:.3e}", flush=True)

    MAX_LOOPS = 20
    TARGET_SOLUTIONS = 20  # <--- TADY JE TO KOUZLO! Jakmile najde 20 řešení, končí.
    
    for _ in range(MAX_LOOPS):
        # --- CHYTRÉ UKONČENÍ ---
        # Zkontrolujeme, jestli už v souboru nemáme dost nalezených bodů
        try:
            with open(filename, "r") as f:
                found_count = len(f.readlines()) - 1  # -1 kvůli hlavičce
            if found_count >= TARGET_SOLUTIONS:
                print(f"-> Skvělé, už máme {found_count} řešení! To pro DBSCAN stačí. Končím hledání.", flush=True)
                break
        except FileNotFoundError:
            pass
        # -----------------------

        processes = []
        for i in range(Ncores):
            random.seed(os.getpid() + i * 1000)
            
            U0 = np.random.uniform(U0_lims[0], U0_lims[1])
            omega = np.random.uniform(omega_lims[0], omega_lims[1])
            rn = np.random.uniform(rn_lims[0], rn_lims[1])
            
            p = Process(target=NodeFinder, args=(U0, omega, rn, U0_lims, omega_lims, rn_lims, filename))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

    print("Hledani pro tuto kombinaci ukonceno.", flush=True)