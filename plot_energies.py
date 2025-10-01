import os
import pandas as pd
import matplotlib.pyplot as plt

# 🔧 MANUALLY SET YOUR DIRECTORIES HERE
simulation_dirs = [

'resources/data/spin=0_N=256',

    'resources/data/spin=1_N=256',
    'resources/data/spin=2_N=256',
    'resources/data/spin=3_N=256',

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
            energy = df["E"]
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


if __name__ == "__main__":
    plot_energy_ratio(simulation_dirs)
    plot_total_energy(simulation_dirs)
