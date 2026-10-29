from pathlib import Path
import glob
import os
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================================================
# T2: collects the runs of tests/sink_friction_scan.py with SCAN=t2 (resources/data/sink_friction_scan_t2*_N*_H*_s*/case_*.npz),
# prints the convergence tables (mean drag coefficient, momentum balance, energy drift against N, H, travel distance) and
# saves resources/data/sink_friction_t2_summary/{t2_curves.png, t2_means.png}.
# =====================================================================================================

OUT = os.path.join("resources", "data", "sink_friction_t2_summary")
os.makedirs(OUT, exist_ok=True)

runs = []
for path in sorted(glob.glob(os.path.join("resources", "data", "sink_friction_scan_t2*", "case_*.npz"))):
    folder = os.path.basename(os.path.dirname(path))
    smooth = "_smooth" in folder
    m = re.search(r"_N(\d+)(?:_(?:H[\d.e+-]+|tr(\d+)))?_s(\d)", folder)
    if m is None:
        continue
    d = np.load(path)
    runs.append(dict(
        n=int(m.group(1)), h=re.search(r"_H([\d.e+-]+)\.npz", path).group(1), travel=("0.25" if m.group(2) else "0.5"), smooth=smooth, mach_q=float(d["mach_q"]),
        mean_sim=float(d["mean_sim"]), mean_theory=float(d["mean_theory"]), mean_exact=float(d["mean_exact"]),
        momentum=float(d["momentum_error_late"]), energy=float(d["energy_drift"]), slowdown=float(d["slowdown"]),
        t=d["t_sim"], c=d["c_sim"], b=d["b_tilde"], theory=d["c_theory"], exact=d["c_exact"], eps_over_lambda=float(d["eps_over_lambda"])))

if not runs:
    sys.exit("no T2 runs found")


def label(r):
    return f"N={r['n']}, H={r['h'] + (' tr' + r['travel'] if r['travel'] != '0.5' else '')}"


print(f"{'M_Q':>5} {'smooth':>6} {'N':>4} {'H':>7} {'travel':>6} {'C sim':>7} {'C lin':>7} {'C exact':>8} {'sim/lin-1':>9} {'P/J-1':>7} {'dE/K_lost':>9} {'slow':>6}")
for q in sorted({r["mach_q"] for r in runs}, reverse=True):
    for smooth in (False, True):
        for r in sorted([r for r in runs if r["mach_q"] == q and r["smooth"] == smooth], key=lambda r: (r["travel"], r["n"], float(r["h"]))):
            print(f"{q:5g} {str(smooth):>6} {r['n']:4d} {r['h']:>7} {r['travel']:>6} {r['mean_sim']:7.3f} {r['mean_theory']:7.3f} "
                  f"{r['mean_exact']:8.3f} {100 * (r['mean_sim'] / r['mean_theory'] - 1):+8.1f}% {100 * r['momentum']:+6.2f}% "
                  f"{100 * r['energy']:8.2f}% {100 * r['slowdown']:5.2f}%")
        print()

# ---- curves: one panel per (smooth, M_Q), N-runs at the stable step and the N=128 step series
qs = sorted({r["mach_q"] for r in runs}, reverse=True)
fig, axes = plt.subplots(2, len(qs), figsize=(5.2 * len(qs), 7), squeeze=False)
for row, smooth in enumerate((False, True)):
    for col, q in enumerate(qs):
        ax = axes[row, col]
        sel = [r for r in runs if r["mach_q"] == q and r["smooth"] == smooth and r["travel"] == "0.5"]
        for r in sorted(sel, key=lambda r: (r["n"], float(r["h"]))):
            ax.plot(r["b"], r["c"], lw=1.0, label=label(r))
        if sel:
            ax.plot(sel[0]["b"], sel[0]["exact"], ":", color="gray", label="exact steady state (point source)")
        ax.set_title(f"M_Q = {q:g}, {'smooth_force' if smooth else 'point receiver'}")
        ax.set_xlabel("v t / lambda")
        ax.set_ylabel("C_rel")
        ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t2_curves.png"), dpi=130)

# ---- means against N (at the largest stable step) and against H (N = 128)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for q in qs:
    for smooth, style in ((False, "o-"), (True, "s--")):
        by_n = sorted([r for r in runs if r["mach_q"] == q and r["smooth"] == smooth and r["travel"] == "0.5" and r["n"] != 128
                       or (r["mach_q"] == q and r["smooth"] == smooth and r["n"] == 128 and r["travel"] == "0.5" and abs(float(r["h"]) - 4e-4) < 1e-9)], key=lambda r: r["n"])
        axes[0].plot([r["n"] for r in by_n], [r["mean_sim"] for r in by_n], style, label=f"M_Q={q:g} {'smooth' if smooth else 'point'}")
        by_h = sorted([r for r in runs if r["mach_q"] == q and r["smooth"] == smooth and r["n"] == 128 and r["travel"] == "0.5" ],
                      key=lambda r: float(r["h"]))
        axes[1].plot([float(r["h"]) for r in by_h], [r["mean_sim"] for r in by_h], style, label=f"M_Q={q:g} {'smooth' if smooth else 'point'}")
axes[0].set_xlabel("N (largest stable H, fixed physical softening 0.29 lambda)")
axes[0].set_ylabel("mean C_rel (v t / lambda > 7)")
axes[1].set_xlabel("time step H [Gyr] (N = 128)")
axes[1].set_xscale("log")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t2_means.png"), dpi=130)
print("plots in", OUT)
