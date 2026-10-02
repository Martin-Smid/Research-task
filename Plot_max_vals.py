import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIRECTORY = Path("resources/data")


def _latest_simulation_directory():
    candidates = [
        path
        for path in DATA_DIRECTORY.glob("simulation_*")
        if path.is_dir() and (path / "max_values.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No simulation directory containing max_values.csv found under {DATA_DIRECTORY}"
        )
    return max(candidates, key=lambda path: path.name)


def plot_max_values(*simulation_directories):
    """Plot normalized maxima from zero or more simulation directories.

    With no paths, the newest simulation directory containing max_values.csv is
    used. Each supplied path must point to a simulation directory.
    """
    directories = (
        [Path(path) for path in simulation_directories]
        if simulation_directories
        else [_latest_simulation_directory()]
    )

    plt.figure(figsize=(10, 6))

    for directory in directories:
        filename = directory / "max_values.csv"
        if not filename.is_file():
            raise FileNotFoundError(f"No max_values.csv found under {directory}")

        data = pd.read_csv(filename, comment="#", header=[0, 1], index_col=0)
        for n, spin in data.columns:
            values = data[(n, spin)]
            normalized = (values / values.iloc[0]) ** 0.25
            label = (
                f"{directory.name}: N = {n.replace('N', '')}, "
                f"spin = {spin.replace('s=', '')}"
            )
            plt.plot(data.index, normalized, label=label)

    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("Normalized Max Values", fontsize=18)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_max_values(*sys.argv[1:])
