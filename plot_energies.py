import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 🔧 MANUALLY SET YOUR DIRECTORIES HERE
simulation_dirs = [

"resources/data/simulation_20260409_085600"




]

def plot_energy_ratio(paths):
    plt.figure(figsize=(8, 4))
    for path in paths:
        energy_file = os.path.join(path, "energy.csv")
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
        energy_file = os.path.join(path, "energy.csv")
        if not os.path.isfile(energy_file):
            print(f"[!] Skipping: 'energy.txt' not found in {path}")
            continue

        try:
            df = pd.read_csv(energy_file)
            time = df["time"]
            energy = df["E_tot_cons"]
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
        energy_file = os.path.join(path, "energy.csv")
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
    Virial diagnostics for ULDM + baryons.

    Two useful dimensionless diagnostics:
      (1) Virial ratio:        2K / |W|      -> ~1 for relaxed bound system
      (2) Virial residual: (2K + W) / |W|   -> ~0 for relaxed bound system

    Here K := K_flow + U_quantum + K_baryons   (your stored decomposition)
         W := W_self
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))

    for path in paths:
        energy_file = os.path.join(path, "energy.csv")
        if not os.path.isfile(energy_file):
            print(f"[!] Skipping: energy.csv not found in {path}")
            continue

        df = pd.read_csv(energy_file)

        required_cols = {"time", "K_flow", "U_quantum", "K_baryons", "W_self"}
        if not required_cols.issubset(df.columns):
            print(f"[!] Missing columns in {energy_file}. Have: {list(df.columns)}")
            continue

        t = df["time"].to_numpy()
        K_flow = df["K_flow"].to_numpy()
        U_q    = df["U_quantum"].to_numpy()
        K_b    = df["K_baryons"].to_numpy()
        W      = df["W_self"].to_numpy()

        K_tot = K_flow + U_q + K_b
        Wabs  = np.maximum(np.abs(W), 1e-30)

        virial_ratio    = 2.0 * K_tot / Wabs                 # ~1
        virial_residual = (2.0 * K_tot + W) / Wabs            # ~0

        label = os.path.basename(os.path.normpath(path))
        plt.plot(t, virial_ratio, label=f"{label}: 2K/|W|")

        print(f"[{label}] mean(2K/|W|) = {virial_ratio.mean():.3f}, "
              f"mean((2K+W)/|W|) = {virial_residual.mean():.3f}")

    plt.axhline(1.0, linestyle="--", linewidth=0.8, label="virial ratio = 1")
    plt.xlabel("Time [Gyr]")
    plt.ylabel("Virial ratio  $2K/|W|$")
    plt.title("Virial diagnostics (ULDM + baryons)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # Optional: plot normalized residuals as a second figure
    plt.figure(figsize=(9, 5))
    for path in paths:
        energy_file = os.path.join(path, "energy.csv")
        if not os.path.isfile(energy_file):
            continue
        df = pd.read_csv(energy_file)
        if not {"time", "K_flow", "U_quantum", "K_baryons", "W_self"}.issubset(df.columns):
            continue

        t = df["time"].to_numpy()
        K_tot = (df["K_flow"] + df["U_quantum"] + df["K_baryons"]).to_numpy()
        W = df["W_self"].to_numpy()
        Wabs = np.maximum(np.abs(W), 1e-30)
        res = (2.0 * K_tot + W) / Wabs

        label = os.path.basename(os.path.normpath(path))
        plt.plot(t, res, label=f"{label}: (2K+W)/|W|")

    plt.axhline(0.0, linestyle="--", linewidth=0.8, label="residual = 0")
    plt.xlabel("Time [Gyr]")
    plt.ylabel("Normalized residual  $(2K+W)/|W|$")
    plt.title("Virial residual (normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()



def plot_kinetic_energy_components_1(paths):
    plt.figure(figsize=(9, 5))

    for path in paths:
        energy_file = os.path.join(path, "energy.csv")
        if not os.path.isfile(energy_file):
            print(f"[!] Skipping: 'energy.txt' not found in {path}")
            continue

        try:
            df = pd.read_csv(energy_file)
        except Exception as e:
            print(f"[!] Failed to read {energy_file}: {e}")
            continue

        required_cols = {"time", "K_total", "K_flow", "U_quantum", "K_baryons"}
        if not required_cols.issubset(df.columns):
            print(f"[!] Missing columns in {energy_file}, found: {list(df.columns)}")
            continue

        time      = df["time"]
        K_total   = df["K_total"]
        K_flow    = df["K_flow"]
        U_quantum = df["U_quantum"]
        K_baryons = df["K_baryons"]

        label = os.path.basename(os.path.normpath(path))

        plt.plot(time, K_total,   label=f"{label} – K_total",   lw=1.8)
        plt.plot(time, K_flow,    "--", label=f"{label} – K_flow",    lw=1.2)
        plt.plot(time, U_quantum, ":",  label=f"{label} – U_quantum", lw=1.2)
        plt.plot(time, K_baryons, "-.", label=f"{label} – K_baryons", lw=1.2)

    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_energy_ratio(simulation_dirs)
    plot_total_energy(simulation_dirs)
    plot_kinetic_energy_components(simulation_dirs)
    plot_virial_check(simulation_dirs)
    #plot_kinetic_energy_components_1(simulation_dirs)