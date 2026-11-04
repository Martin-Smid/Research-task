"""Plot the latest rotation curves from one or more simulation directories."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIRECTORY = Path(__file__).resolve().parent / "resources/data"


def _latest_simulation_directory():
    candidates = [p for p in DATA_DIRECTORY.glob("simulation_*")
                  if p.is_dir() and (p / "rotational_velocity.dat").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No saved rotation curves under {DATA_DIRECTORY}")
    for directory in sorted(candidates, key=lambda p: p.name, reverse=True):
        if not pd.read_csv(directory / "rotational_velocity.dat", nrows=1).empty:
            return directory
    raise FileNotFoundError(f"No recorded rotation curves under {DATA_DIRECTORY}")


def _component_kinds(directory):
    path = directory / "run_config.json"
    if not path.exists():
        return {}
    components = json.loads(path.read_text(encoding="utf-8"))["components"]
    names = [c.get("name") or f"baryons_{i}" for i, c in enumerate(components)]
    return {(name if names.count(name) == 1 else f"{name}_{i}"):
            ("gas" if "gas" in c["type"].lower() else "nbody")
            for i, (name, c) in enumerate(zip(names, components))}


def plot_rotation_curves(*simulation_directories, gas=True, nbody=True, time=None,
                         components=None, output=None, km_s=True, show=True):
    """Plot each run's latest (or nearest requested) time; select gas and/or N-body."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directories = [Path(p) for p in simulation_directories] if simulation_directories else [_latest_simulation_directory()]
    directories = [DATA_DIRECTORY / p if not p.is_dir() and p.parent == Path(".") else p for p in directories]
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted, display_units = False, None
    for directory in directories:
        curves = pd.read_csv(directory / "rotational_velocity.dat")
        frame_path = directory / "rotation_frames.csv"
        frames = pd.read_csv(frame_path) if frame_path.exists() else pd.DataFrame()
        times = np.sort((frames if not frames.empty else curves)["time"].unique())
        if len(times) == 0:
            print(f"{directory.name}: no saved rotation diagnostics")
            continue
        selected_time = times[-1] if time is None else times[np.argmin(abs(times - time))]
        curves = curves[np.isclose(curves["time"], selected_time, rtol=0, atol=1e-12)]
        curves = curves.drop_duplicates(["component", "R"], keep="last")
        if not frames.empty:
            frames = frames[np.isclose(frames["time"], selected_time, rtol=0, atol=1e-12)].drop_duplicates("component", keep="last")
        kinds = _component_kinds(directory)
        for name, rows in curves.groupby("component", sort=False):
            kind = kinds.get(name, "gas" if "gas" in str(name).lower() else "nbody")
            if (kind == "gas" and not gas) or (kind == "nbody" and not nbody) or (components and name not in components):
                continue
            rows = rows.sort_values("R")
            rows = rows[(rows["weight"] > 0) & np.isfinite(rows["vphi_mean"]) & np.isfinite(rows["vphi_std"])]
            if rows.empty:
                continue
            metadata = frames[frames["component"] == name] if not frames.empty else pd.DataFrame()
            length_unit, velocity_unit, factor = "simulation units", "simulation units", 1.0
            if not metadata.empty:
                row = metadata.iloc[-1]
                length_unit, velocity_unit = row["length_unit"], row["velocity_unit"]
                print(f"{directory.name}/{name}: {row['status']}; axis=({row.axis_x:.3f}, {row.axis_y:.3f}, {row.axis_z:.3f})")
                if km_s and velocity_unit != "simulation":
                    from astropy import units
                    factor = units.Unit(velocity_unit).to(units.km / units.s)
                    velocity_unit = "km/s"
            current_units = (length_unit, velocity_unit)
            if display_units is not None and current_units != display_units:
                raise ValueError("Runs have different units; convert them before plotting together")
            display_units = current_units
            radius = rows["R"].to_numpy()
            mean, sigma = rows["vphi_mean"].to_numpy() * factor, rows["vphi_std"].to_numpy() * factor
            label = f"{directory.name}: {name}, t={selected_time:g}"
            if len(radius) == 1:
                ax.errorbar(radius, mean, yerr=sigma, fmt="o", capsize=3, label=label)
            else:
                line, = ax.plot(radius, mean, marker="o", markersize=3, label=label)
                ax.fill_between(radius, mean - sigma, mean + sigma, color=line.get_color(), alpha=0.15)
            plotted = True
        if not frames.empty:
            for row in frames[frames["status"] != "ok"].itertuples():
                print(f"{directory.name}/{row.component}: {row.status}; no curve")
    if not plotted:
        plt.close(fig)
        raise ValueError("No usable curves for this selection; inspect frame status or choose another run/time")
    ax.set(xlabel=f"Cylindrical radius [{display_units[0]}]",
           ylabel=f"Mean azimuthal velocity [{display_units[1]}]",
           title="Rotation curves\nBands/bars: velocity dispersion")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = Path(output) if output else directories[0] / "rotation_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument("-g", "--gas", action="store_true", help="Plot gas only")
    parser.add_argument("-N", "--nbody", action="store_true", help="Plot N-body only")
    parser.add_argument("--time", type=float, help="Nearest saved time; defaults to latest in each run")
    parser.add_argument("--component", action="append", dest="components")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--km-s", dest="km_s", action="store_true", default=True)
    parser.add_argument("--native-units", dest="km_s", action="store_false")
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window")
    args = parser.parse_args()
    plot_rotation_curves(*args.directories, gas=args.gas or not args.nbody,
                         nbody=args.nbody or not args.gas, time=args.time, components=args.components,
                         output=args.output, km_s=args.km_s, show=not args.no_show)


if __name__ == "__main__":
    main()