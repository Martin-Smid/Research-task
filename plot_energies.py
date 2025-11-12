import os
import pandas as pd
import matplotlib.pyplot as plt

# 🔧 MANUALLY SET YOUR DIRECTORIES HERE
simulation_dirs = [

'resources/data/simulation_20251112_092937'

]

def plot_energy_ratio(paths):
    plt.figure(figsize=(8, 4))
    for path in paths:
        energy_file = os.path.join(path, "energy.txt")
        if not os.path.isfile(energy_file):
            print(f"[!] Skipping: 'energy.txt' not found in {path}")
            continue

        try:
            df = pd.read_csv(energy_file)
            time = df["time"]
            ratio = df["W/|E|"]
        except Exception as e:
            print(f"[!] Failed to read {energy_file}: {e}")
            continue

        label = os.path.basename(os.path.normpath(path))
        plt.plot(time, ratio, label=label)

    plt.xlabel("Time")
    plt.ylabel("W / |E|")
    plt.title("Energy Ratio Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_total_energy(paths):
    plt.figure(figsize=(8, 4))
    for path in paths:
        energy_file = os.path.join(path, "energy.txt")
        if not os.path.isfile(energy_file):
            print(f"[!] Skipping: 'energy.txt' not found in {path}")
            continue

        try:
            df = pd.read_csv(energy_file)
            time = df["time"]
            energy = df["E_total"]
        except Exception as e:
            print(f"[!] Failed to read {energy_file}: {e}")
            continue

        if len(energy) == 0:
            print(f"[!] Skipping: 'energy.txt' in {path} is empty")
            continue

        E0 = energy.iloc[0]
        if E0 == 0:
            print(f"[!] Skipping: E0 = 0 in {path}, cannot normalize")
            continue

        delta_E_over_E0 = (energy - E0) / E0
        label = os.path.basename(os.path.normpath(path))
        plt.plot(time, delta_E_over_E0, label=f'N = {label}')

    plt.xlabel("Time [Gyr]")
    plt.ylabel(r"$\Delta E / E_0$")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kinetic_energy_components(paths):
    """
    Plot kinetic energy components (K_total, K_flow, U_quantum) over time
    from the detailed energy logs in the given directories.

    Parameters:
        paths (list[str]): List of directories containing 'energy.txt' files.
    """
    plt.figure(figsize=(9, 5))

    for path in paths:
        energy_file = os.path.join(path, "energy.txt")
        if not os.path.isfile(energy_file):
            print(f"[!] Skipping: 'energy.txt' not found in {path}")
            continue

        try:
            df = pd.read_csv(energy_file)
        except Exception as e:
            print(f"[!] Failed to read {energy_file}: {e}")
            continue

        # Check if required columns exist
        required_cols = {"time", "K_total", "K_flow", "U_quantum"}
        if not required_cols.issubset(df.columns):
            print(f"[!] Missing columns in {energy_file}, found: {list(df.columns)}")
            continue

        time = df["time"]
        K_total = df["K_total"]
        K_flow = df["K_flow"]
        U_quantum = df["U_quantum"]

        label = os.path.basename(os.path.normpath(path))

        plt.plot(time, K_total, label=f"{label} – K_total", lw=1.8)
        plt.plot(time, K_flow, '--', label=f"{label} – K_flow", lw=1.2)
        plt.plot(time, U_quantum, ':', label=f"{label} – U_quantum", lw=1.2)

    plt.xlabel("Time")
    plt.ylabel("Energy [units?]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_virial_check(paths):
    """
    Plot kinetic energy components and check the virial theorem:
        K_flow + U_quantum - W ≈ 0
    for each simulation folder containing 'energy.txt'.

    Parameters:
        paths (list[str]): List of directories containing energy.txt files.
    """
    plt.figure(figsize=(9, 5))

    for path in paths:
        energy_file = os.path.join(path, "energy.txt")
        if not os.path.isfile(energy_file):
            print(f"[!] Skipping: 'energy.txt' not found in {path}")
            continue

        try:
            df = pd.read_csv(energy_file)
        except Exception as e:
            print(f"[!] Failed to read {energy_file}: {e}")
            continue

        required_cols = {"time", "K_flow", "U_quantum", "W"}
        if not required_cols.issubset(df.columns):
            print(f"[!] Missing required columns in {energy_file}, found: {list(df.columns)}")
            continue

        time = df["time"]
        K_flow = df["K_flow"]
        U_quantum = df["U_quantum"]
        W = df["W"]

        # Virial theorem residual
        virial_residual = 2*K_flow + 2*U_quantum + W

        label = os.path.basename(os.path.normpath(path))

        plt.plot(time, virial_residual, label=f"{label}: K_flow + U_q - W")

        # Optionally print mean deviation
        mean_abs = abs(virial_residual).mean()
        print(f"[{label}] Mean |K_flow + U_q - W| = {mean_abs:.3e}")

    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel("K_flow + U_quantum - W")
    plt.title("Virial Theorem Check Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_energy_ratio(simulation_dirs)
    plot_total_energy(simulation_dirs)
    plot_kinetic_energy_components(simulation_dirs)
    plot_virial_check(simulation_dirs)
