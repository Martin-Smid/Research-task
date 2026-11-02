"""Tables and C(t) plots: jaxion (out_main, out_eps) vs the reference code (read-only npz in research_task)."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REF = os.path.expanduser("~/projects/research_task/resources/data")
HOME = os.path.expanduser("~/projects/jaxion")
OUT = f"{HOME}/results"
os.makedirs(OUT, exist_ok=True)
LAMBDA = 0.2614


def ref_dir(n, eps, smooth, mq):
    """Directory of the reference npz for this case; the reference used eps = 0.075 (0.29 lambda), 0.0375, 0.01875 kpc."""
    s = "smooth_" if smooth else ""
    suf = f"_s{1 if smooth else 0}"
    if abs(eps - 0.075) < 1e-6:
        tag = {128: "N128_H4e-4", 192: "N192", 256: "N256"}.get(n)
        return f"{REF}/sink_friction_scan_t2_{s}{tag}{suf}" if tag else None
    e = "0.0375" if abs(eps - 0.0366) < 1e-3 else "0.01875"
    if n == 256:
        return f"{REF}/sink_friction_scan_t2_{s}eps{e}{suf}"
    if n == 320:
        q = "q03" if abs(mq - 0.3) < 1e-9 else ""
        return f"{REF}/sink_friction_scan_t2_{s}N320{q}_eps{e}{suf}"
    return None


def ref_case(n, eps, smooth, mq):
    d = ref_dir(n, eps, smooth, mq)
    if d is None:
        return None
    files = sorted(glob.glob(f"{d}/case_MQ{mq:g}_N{n}_*.npz"))
    if not files:
        return None
    z = np.load(files[0], allow_pickle=True)
    return dict(t=z["t_sim"], c=z["c_sim"], mean=float(z["mean_sim"]), bal=float(z["momentum_error_late"]),
                slow=float(z["slowdown"]), file=files[0])


def osc(t1, c1, t2, c2, b1, lam_over_v):
    """Phase lag (Gyr) and correlation of the detrended oscillations of two C(t) curves for v t / lambda > 7."""
    m = t2 > 7 * lam_over_v
    tt = t2[m]
    a = np.interp(tt, t1, c1)
    b = c2[m]
    p = np.polyfit(tt, a, 2)
    q = np.polyfit(tt, b, 2)
    a = a - np.polyval(p, tt)
    b = b - np.polyval(q, tt)
    if a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, np.nan
    dt = np.mean(np.diff(tt))
    best = (-2, 0)
    for lag in range(-int(0.03 / dt), int(0.03 / dt) + 1):
        x, y = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        r = np.corrcoef(x, y)[0, 1]
        if r > best[0]:
            best = (r, lag * dt)
    return best[1], best[0], np.corrcoef(a, b)[0, 1]


rows = []
for f in sorted(glob.glob(f"{HOME}/out_main/jx_*.npz") + glob.glob(f"{HOME}/out_eps/jx_*.npz")):
    d = np.load(f, allow_pickle=True)
    r = dict(mode=str(d["mode"]), n=int(d["n"]), eps=float(d["eps"]), mq=float(d["mq"]), mean=float(d["c_mean"]),
             bal=float(d["bal_late"]), slow=float(d["slowdown"]), t=d["t_c"], c=d["c"], wall=float(d["wall"]),
             nt=int(d["nt"]), dx=float(d["dx"]), peak=float(d["peak_gb"]) if "peak_gb" in d else np.nan,
             dt=float(d["dt_gyr"]) if "dt_gyr" in d else np.nan)
    r["el"] = r["eps"] / LAMBDA
    r["ec"] = r["eps"] / r["dx"]
    rows.append(r)
# native ignores eps: keep one copy; drop the under-resolved eps = 0.021 (0.14 cell at N = 128) rows from the comparison
rows = [r for r in rows if not (r["mode"] == "native" and abs(r["eps"] - 0.021) < 1e-9)]
rows = [r for r in rows if abs(r["eps"] - 0.021) > 1e-9]
rows.sort(key=lambda r: (r["mq"], r["mode"], r["n"], -r["eps"]))

L = ["## Výsledky: průměrné C pro v t / lambda > 7 (jaxion vs. náš kód, stejné N, eps a typ přijímače)", "",
     "| kód | přijímač | N | eps [kpc] | eps/lambda | eps/buňka | M_Q | zpomalení % | <C> jaxion | <C> náš kód | rozdíl % | P_psi/P_ztracená-1 pozdě % | náš kód: bilance % |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    ref = None
    if r["mode"] != "native":
        ref = ref_case(r["n"], r["eps"], r["mode"] == "smooth", r["mq"])
    refm = f"{ref['mean']:.3f}" if ref else "-"
    diff = f"{100 * (r['mean'] / ref['mean'] - 1):+.1f}" if ref else "-"
    rb = f"{100 * ref['bal']:+.2f}" if ref else "-"
    recv = {"native": "vestavěná částice (CIC)", "point": "bodový", "smooth": "vyhlazený"}[r["mode"]]
    L.append(f"| jaxion | {recv} | {r['n']} | {r['eps']:g} | {r['el']:.2f} | {r['ec']:.2f} | {r['mq']:g} | {100 * r['slow']:.2f} | "
             f"{r['mean']:.3f} | {refm} | {diff} | {100 * r['bal']:+.2f} | {rb} |")
L += ["", "Rozdíl = jaxion / náš kód − 1. U vestavěné částice nemá eps smysl (změkčení je dané buňkou), proto nemá referenci. "
      "Reference pro eps = 0,0366 / 0,0183 kpc jsou naše běhy s eps = 0,0375 / 0,01875 kpc (o 2,4 % širší jádro). "
      "Naše H: 4e-4 (N=128), 2,5e-4 (192), 1,4e-4 (256), 9e-5 (320) Gyr.", ""]

# oscillation phase
L += ["## Oscilace C(t) pro M_Q <= 1 (v t / lambda > 7, odečten kvadratický trend)", "",
      "| přijímač | N | eps/lambda | M_Q | posun fáze jaxion vůči našemu kódu [Gyr] | korelace při tom posunu | korelace bez posunu |", "|---|---|---|---|---|---|---|"]
for r in rows:
    if r["mode"] == "native" or r["mq"] > 1:
        continue
    ref = ref_case(r["n"], r["eps"], r["mode"] == "smooth", r["mq"])
    if not ref:
        continue
    lag, rbest, r0 = osc(r["t"], r["c"], ref["t"], ref["c"], 0, LAMBDA / 30.0)
    L.append(f"| {r['mode']} | {r['n']} | {r['el']:.2f} | {r['mq']:g} | {lag:+.4f} | {rbest:.2f} | {r0:.2f} |")
L += ["", "Kladný posun = oscilace jaxionu jsou oproti našemu kódu zpožděné (korelace hledána v rozsahu ±0,03 Gyr).", ""]

# cost
L += ["## Čas a paměť (jeden případ; GPU P100 je sdílená, čas tedy mohl být zpomalen cizími joby)", "",
      "| N | přijímač | kroků | dt [Gyr] | dt / (limit jaxionu) | čas běhu [s] | max. paměť GPU [GiB] |", "|---|---|---|---|---|---|---|"]
seen = set()
for r in sorted(rows, key=lambda r: (r["n"], r["mode"])):
    key = (r["n"], r["mode"])
    if key in seen:
        continue
    seen.add(key)
    dt_lim = 1.0 / 6.0 / 7.8428 * r["dx"] ** 2
    dtv = r["dt"] if not np.isnan(r["dt"]) else 0.32 / r["nt"]
    L.append(f"| {r['n']} | {r['mode']} | {r['nt']} | {dtv:.2e} | {dtv / dt_lim:.2f} | {r['wall']:.0f} | "
             f"{'%.2f' % r['peak'] if not np.isnan(r['peak']) else 'neměřeno'} |")
L.append("")
open(f"{OUT}/table_results.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# C(t) plots for M_Q = 1 and 0.3
n_pref = {1.0: None, 0.3: None}
for mq in (1.0, 0.3):
    cases = sorted({(r["n"], r["eps"]) for r in rows if r["mq"] == mq and r["mode"] != "native"}, key=lambda x: (-x[0], -x[1]))
    eps_list = sorted({e for n, e in cases}, reverse=True)
    fig, axes = plt.subplots(len(eps_list), 1, figsize=(9, 3.6 * len(eps_list)), squeeze=False)
    for ax, eps in zip(axes[:, 0], eps_list):
        n = max(nn for nn, e in cases if e == eps)
        for mode, col in (("point", "tab:red"), ("smooth", "tab:blue")):
            for r in rows:
                if r["mq"] == mq and r["n"] == n and r["eps"] == eps and r["mode"] == mode:
                    ax.plot(r["t"], r["c"], "-", color=col, label=f"jaxion {mode}")
            ref = ref_case(n, eps, mode == "smooth", mq)
            if ref:
                ax.plot(ref["t"], ref["c"], "--", color=col, alpha=0.8, label=f"náš kód {mode}")
        ax.set_title(f"M_Q = {mq:g}, N = {n}, eps = {eps:g} kpc = {eps / LAMBDA:.2f} lambda")
        ax.set_xlabel("t [Gyr]")
        ax.set_ylabel("C(t)")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/C_of_t_MQ{mq:g}.png", dpi=120)
