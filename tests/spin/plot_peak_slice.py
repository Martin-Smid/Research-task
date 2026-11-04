"""Plot the density peak of a saved run: slice through the maximum, zoom, phase of psi and the radial profile vs the soliton formula.

Usage (repository root, project venv):
    python tests/spin/plot_peak_slice.py resources/data/simulation_20261102_xxxxxx [--time 10] [--half 12] [--out file.png]

Reads the snapshots wf_<m>_snapshot_at_time_<t>.npy of the run folder (all 2s+1 polarizations; the spin is the number of files at
the chosen time), the grid size from the snapshot and the box and the particle mass from run_config.json. --time picks the snapshot
closest to that time [Gyr] (default: the last one), --half is the half width of the zoom in cells. The figure is saved into the
run folder (peak_slice_t<time>.png) unless --out is given.

Panels: (1) log10 rho in the slice through the maximum, (2) zoom around the maximum with the soliton half-density radius x_c of the
formula rho_c = 1.9e7 m22^-2 x_c^-4 Msun/kpc^3 (Schive et al. 2014, Eq. 3; m22 = m / 1e-22 eV) for the measured rho_max,
(3) phase of the strongest polarization in the zoom (coherent = soliton, random = granules), (4) angular-averaged profile around the
maximum divided by rho_max against the soliton profile 1 / (1 + 0.091 (r / x_c)^2)^8."""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_key(obj, key):
    """First value of `key` in a nested dict / list (run_config.json)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--half", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.folder, "wf_0_snapshot_at_time_*.npy"))
    times = {float(re.search(r"time_([0-9.]+)\.npy", f).group(1)): f for f in files}
    t = max(times) if args.time is None else min(times, key=lambda x: abs(x - args.time))
    first = times[t]
    n_comp = len(glob.glob(os.path.join(args.folder, f"wf_*_snapshot_at_time_{os.path.basename(first).split('time_')[1]}")))
    psi = [np.load(first.replace("wf_0_", f"wf_{m}_")) for m in range(n_comp)]
    N = psi[0].shape[0]

    config = json.load(open(os.path.join(args.folder, "run_config.json")))
    boundaries = find_key(config, "boundaries")
    box = float(boundaries[0][1] - boundaries[0][0])                 # box length [kpc]
    m22 = float(find_key(config, "m_s")) / 1.0e-22
    dx = box / N
    rho = sum(np.abs(p) ** 2 for p in psi)
    mean = rho.mean()
    ix, iy, iz = np.unravel_index(np.argmax(rho), rho.shape)
    x_c = (1.9e7 * m22 ** -2 / rho.max()) ** 0.25
    half = args.half
    around = lambda a: np.take(np.take(a[:, :, iz], np.arange(ix - half, ix + half + 1), mode="wrap", axis=0),
                               np.arange(iy - half, iy + half + 1), mode="wrap", axis=1)
    strongest = psi[int(np.argmax([np.sum(np.abs(p) ** 2) for p in psi]))]

    index = [np.minimum(np.abs(np.arange(N) - c), N - np.abs(np.arange(N) - c)) * dx for c in (ix, iy, iz)]
    r = np.sqrt(index[0][:, None, None] ** 2 + index[1][None, :, None] ** 2 + index[2][None, None, :] ** 2)
    edges = np.arange(0.0, min(3.0, box / 2), dx)
    bins = np.digitize(r.ravel(), edges) - 1
    counts = np.maximum(np.bincount(bins, minlength=len(edges) + 1)[:len(edges)], 1)
    profile = np.bincount(bins, weights=rho.ravel(), minlength=len(edges) + 1)[:len(edges)] / counts
    centers = edges + dx / 2

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    im = axes[0].imshow(np.log10(rho[:, :, iz].T), origin="lower", extent=[0, box, 0, box], cmap="magma")
    axes[0].plot([(ix + 0.5) * dx], [(iy + 0.5) * dx], "c+", ms=14)
    axes[0].set_title(f"slice through the maximum\nt = {t:.2f} Gyr, spin {(n_comp - 1) // 2}, N = {N}")
    plt.colorbar(im, ax=axes[0], label="log10 rho [Msun/kpc^3]")
    extent = [-half * dx, half * dx, -half * dx, half * dx]
    im = axes[1].imshow(np.log10(around(rho).T), origin="lower", extent=extent, cmap="magma")
    axes[1].add_patch(plt.Circle((0, 0), x_c, fill=False, color="c", ls="--"))
    axes[1].set_title(f"zoom; cyan: soliton x_c = {x_c:.2f} kpc ({x_c / dx:.1f} cells)")
    plt.colorbar(im, ax=axes[1], label="log10 rho")
    im = axes[2].imshow(np.angle(around(strongest)).T, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[2].set_title("phase of the strongest polarization")
    plt.colorbar(im, ax=axes[2], label="arg psi")
    axes[3].plot(centers, profile / rho.max(), "o-", label="measured")
    axes[3].plot(centers, 1.0 / (1.0 + 0.091 * (centers / x_c) ** 2) ** 8, "--", label=f"soliton, x_c = {x_c:.2f} kpc")
    axes[3].set(xlabel="r [kpc]", ylabel="rho / rho_max", yscale="log", title=f"rho_max / rho_mean = {rho.max() / mean:.0f}")
    axes[3].legend()
    for ax in axes[1:3]:
        ax.set_xlabel("kpc")
        ax.set_ylabel("kpc")
    axes[0].set_xlabel("kpc")
    axes[0].set_ylabel("kpc")
    fig.tight_layout()
    out = args.out or os.path.join(args.folder, f"peak_slice_t{t:.2f}.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}: t = {t:.2f} Gyr, rho_max / rho_mean = {rho.max() / mean:.0f}, x_c = {x_c:.3f} kpc = {x_c / dx:.1f} cells")


if __name__ == "__main__":
    main()
