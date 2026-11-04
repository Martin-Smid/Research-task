"""Run the Boss & Bodenheimer benchmark (test_boss_bodenheimer.py) with the output kept and make the figures.

Usage: plot_boss_bodenheimer.py [N_GRID=128] [federrath=1|0]. Figures go to plots/boss_bodenheimer/N<N>_<federrath|no_federrath>/:
    slices.png        gas density in the z = 0 plane (units of rho0) at chosen times, sinks marked
    history.png       peak gas density vs time (threshold, formation events), sink masses, mass balance, angular momentum
    engine_output.png the same gas density read back from the snapshot files the engine itself saved (save_every steps)
    engine_energy.png energy.csv and max_values.csv written by the engine
The output folder of the run (snapshots, energy.csv, max_values.csv) is printed at the end; delete it when done."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import re

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import test_boss_bodenheimer as bb

SAVE_EVERY = 26            # snapshot file names carry the time with 6 decimals in Gyr (1 kyr) while a step is 0.09 kyr: save at least every 12 steps
RECORD_EVERY = 4           # in-memory record (slices, sinks) every 4 steps
N_GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 128
FEDERRATH = (sys.argv[2] != "0") if len(sys.argv) > 2 else True
OUT = PROJECT_ROOT / "plots" / "boss_bodenheimer" / f"N{N_GRID}_{'federrath' if FEDERRATH else 'no_federrath'}"

records = []


def record(step, sim, gas):
    if step % RECORD_EVERY:
        return
    rho = cp.asnumpy(gas.rho)
    sinks = getattr(sim.evolution, "sink_system", None)
    x, y = (cp.asnumpy(cp.asarray(g)) for g in sim.grids[:2])
    vx, vy = cp.asnumpy(gas.vx), cp.asnumpy(gas.vy)
    lz = float((rho * (x * vy - y * vx)).sum()) * sim.dV
    records.append(dict(step=step, time=step * sim.h, slice=rho[:, :, rho.shape[2] // 2].copy(), rho_max=float(rho.max()),
                        gas_mass=float(rho.sum()) * sim.dV, lz=lz,
                        sink_pos=None if sinks is None else cp.asnumpy(sinks.positions).copy(),
                        sink_mass=None if sinks is None else cp.asnumpy(sinks.masses).copy()))


def main():
    result = bb.main(n_grid=N_GRID, federrath=FEDERRATH, strict=False, keep_output=True, save_every=SAVE_EVERY, diagnostics_every=5, save_max_vals=True, on_step=record)
    sim, t_ff, R = result["sim"], result["t_ff"], result["R"]
    rho0 = 1.0 / (4.0 / 3.0 * np.pi * R ** 3)
    AU = bb.code(1 * bb.u.AU, bb.u.kpc)
    box = bb.BOX_IN_RADII * R
    extent = np.array([-box, box, -box, box]) / AU
    threshold = sim.evolution.sink_density_threshold
    OUT.mkdir(parents=True, exist_ok=True)
    tau = np.array([r["time"] for r in records]) / t_ff

    # --- 1) slices
    wanted = [0.0, 0.5, 0.8, 0.95, 1.02, 1.1, 1.3, 1.5]
    picks = [int(np.argmin(np.abs(tau - w))) for w in wanted]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), constrained_layout=True)
    for ax, k in zip(axes.ravel(), picks):
        r = records[k]
        image = ax.imshow(np.log10(r["slice"].T / rho0), origin="lower", extent=extent, vmin=-2, vmax=2.5, cmap="viridis")
        if r["sink_pos"] is not None:
            ax.plot(r["sink_pos"][:, 0] / AU, r["sink_pos"][:, 1] / AU, "r+", ms=14, mew=2)
        ax.set_title(f"t = {tau[k]:.2f} t_ff, peak {r['rho_max'] / rho0:.0f} rho0, {0 if r['sink_pos'] is None else len(r['sink_pos'])} sinks")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
    fig.colorbar(image, ax=axes, label="log10(gas density / rho0), plane z = 0", shrink=0.8)
    fig.savefig(OUT / "slices.png", dpi=110)
    plt.close(fig)

    # --- 2) history
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogy(tau, [r["rho_max"] / rho0 for r in records], label="peak gas density")
    ax.axhline(threshold / rho0, color="k", ls="--", label=f"Truelove threshold ({threshold / rho0:.0f} rho0)")
    for time, n, masses, positions, _ in result["events"]:
        ax.axvline(time / t_ff, color="r", alpha=0.4)
    ax.set_xlabel("t / t_ff")
    ax.set_ylabel("rho_max / rho0")
    ax.set_title("red lines: sink formation events")
    ax.legend()
    ax = axes[0, 1]
    for time, n, masses, positions, _ in result["events"]:
        ax.plot([time / t_ff] * len(masses), masses, "o")
    ax.set_yscale("log")
    ax.set_xlabel("t / t_ff")
    ax.set_ylabel("sink masses [Msun]")
    ax.set_title("masses of all sinks after each event (no accretion: formation only)")
    ax = axes[1, 0]
    total = np.array([r["gas_mass"] + (0.0 if r["sink_mass"] is None else r["sink_mass"].sum()) for r in records])
    ax.plot(tau, total / total[0] - 1.0)
    ax.set_xlabel("t / t_ff")
    ax.set_ylabel("(gas + sinks) / initial - 1")
    ax.set_title("mass balance (periodic box, no outflow)")
    ax = axes[1, 1]
    ax.plot(tau, np.array([r["lz"] for r in records]) / records[0]["lz"])
    ax.set_xlabel("t / t_ff")
    ax.set_ylabel("gas L_z / initial")
    ax.set_title("gas angular momentum (falls as gas goes into sinks / numerical viscosity)")
    fig.savefig(OUT / "history.png", dpi=110)
    plt.close(fig)

    # --- 3) what the engine saved
    folder = Path(result["output_dir"])
    stamp = lambda f: float(re.findall(r"time_([0-9.]+)\.npy", f.name)[0])
    files = sorted(folder.glob("baryons_snapshot_at_time_*.npy"), key=stamp)
    print(f"engine output folder: {folder}")
    print(f"  {len(files)} baryon-density snapshots (gas + sinks deposit), times [Gyr]: {[stamp(f) for f in files]}")
    print("  other files:", sorted(f.name for f in folder.iterdir() if not f.name.endswith(".npy")))
    if files:
        chosen = [files[i] for i in np.linspace(0, len(files) - 1, 8).astype(int)]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), constrained_layout=True)
        for ax, f in zip(axes.ravel(), chosen):
            rho = np.load(f)
            image = ax.imshow(np.log10(rho[:, :, rho.shape[2] // 2].T / rho0), origin="lower", extent=extent, vmin=-2, vmax=2.5)
            ax.set_title(f"t = {stamp(f) / t_ff:.2f} t_ff")
        fig.colorbar(image, ax=axes, label="log10(rho / rho0), from baryons_snapshot_*.npy", shrink=0.8)
        fig.savefig(OUT / "engine_output.png", dpi=110)
        plt.close(fig)

    energy = np.genfromtxt(folder / "energy.csv", delimiter=",", names=True)
    peak = np.loadtxt(folder / "max_values.csv", delimiter=",", skiprows=3)      # step index, max density
    print(f"  energy.csv: {len(energy)} rows every 5 steps; max_values.csv: {len(peak)} rows (every step)")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    t_e = energy["time"] / t_ff
    axes[0].plot(t_e, energy["K_total"], label="kinetic")
    axes[0].plot(t_e, -energy["W"], label="|gravitational W|")
    axes[0].set_yscale("log")
    axes[0].set_title("energies [Msun kpc^2/Gyr^2]; U_iso is 400x larger (gamma = 1.001)")
    axes[1].plot(t_e, (energy["E_tot_cons"] - energy["E_tot_cons"][0]) / np.abs(energy["W"]).max(), label="(E_tot + E_diss - E_0) / max|W|")
    axes[1].set_title("energy drift in units of the gravitational energy")
    axes[2].semilogy(peak[:, 0] * sim.h / t_ff, peak[:, 1] / rho0)
    axes[2].set_title("max_values.csv: peak total density / rho0 (every step)")
    for ax in axes:
        ax.set_xlabel("t / t_ff")
    axes[0].legend()
    axes[1].legend()
    fig.savefig(OUT / "engine_energy.png", dpi=110)
    plt.close(fig)
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
