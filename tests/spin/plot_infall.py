"""Distance of the sink from the soliton peak versus time, from the output of sink_infall.py.

Usage: python tests/spin/plot_infall.py <label>=<simulation folder> [<label>=<folder> ...] [--box 9.6] [--out file.png]
Reads particle_trajectory.csv (sink) and max_locations.txt (density peak) in every folder. Distances use the periodic
minimum image. Prints the first time the separation drops below 1.0 / 0.5 / 0.25 kpc and the separation at the end."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
parser.add_argument("--box", type=float, default=9.6)
parser.add_argument("--out", default="sink_infall.png")
args = parser.parse_args()
L = 2 * args.box

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for item in args.runs:
    label, folder = item.split("=", 1)
    sink = np.genfromtxt(os.path.join(folder, "particle_trajectory.csv"), delimiter=",", names=True)
    peak = np.loadtxt(os.path.join(folder, "max_locations.txt"), delimiter=",")
    t = sink["time"]
    # the peak is logged every step including two extra rows at the start; align by time
    pt, pxyz = peak[:, 0], peak[:, 4:7]
    pxyz = np.array([np.interp(t, pt, pxyz[:, i]) for i in range(3)]).T
    d = np.stack([sink["x"], sink["y"], sink["z"]], axis=1) - pxyz
    d = (d + L / 2) % L - L / 2
    r = np.linalg.norm(d, axis=1)
    axes[0].plot(t, r, label=label)
    steps = np.diff(np.stack([sink["x"], sink["y"], sink["z"]], axis=1), axis=0)
    steps = (steps + L / 2) % L - L / 2   # undo the periodic wrap of the stored positions
    axes[1].plot(t, np.linalg.norm(np.vstack([np.zeros(3), np.cumsum(steps, axis=0)]), axis=1), label=label)
    firsts = {R: (t[np.argmax(r < R)] if np.any(r < R) else np.nan) for R in (1.0, 0.5, 0.25)}
    print(f"{label}: r(0) = {r[0]:.2f} kpc, first time r < 1 / 0.5 / 0.25 kpc: " + " / ".join(f"{firsts[R]:.3f}" for R in firsts)
          + f" Gyr, r(end) = {r[-1]:.2f} kpc at t = {t[-1]:.2f}, min r = {r.min():.3f} at t = {t[np.argmin(r)]:.3f}")
axes[0].set(xlabel="t [Gyr]", ylabel="|sink - soliton peak| [kpc]")
axes[1].set(xlabel="t [Gyr]", ylabel="sink displacement from start [kpc]")
for a in axes:
    a.grid(alpha=0.3)
    a.legend()
fig.tight_layout()
fig.savefig(args.out, dpi=130)
print("saved", args.out)
