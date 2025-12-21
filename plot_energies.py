import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 🔧 MANUALLY SET YOUR DIRECTORIES HERE
simulation_dirs = [






"resources\data\simulation_20251221_211125"






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
    Virial theorem diagnostic for simulations with ULDM + N-body baryons.

    For self-gravity only (no external time-dependent potential),
    we expect approximately:

        2 (K_flow + U_quantum + K_baryons) + W_self ≈ 0

    where W_self is the self-gravitational energy from Poisson
    using the total density (waves + baryons).
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

        required_cols = {"time", "K_flow", "U_quantum", "K_baryons", "W_self"}
        if not required_cols.issubset(df.columns):
            print(f"[!] Missing required columns in {energy_file}, found: {list(df.columns)}")
            continue

        time       = df["time"]
        K_flow     = df["K_flow"]
        U_quantum  = df["U_quantum"]
        K_baryons  = df["K_baryons"]
        W_self     = df["W_self"]

        # Total kinetic (waves + baryons)


        # Virial residual: should be ~0 for a relaxed, self-gravitating system
        virial_residual = 2 * (K_flow + U_quantum + K_baryons) / np.abs(W_self)

        label = os.path.basename(os.path.normpath(path))
        plt.plot(time, virial_residual, label=f"{label}: 2K_tot + W_self")

        mean_abs = abs(virial_residual).mean()
        mean_norm = mean_abs / max(1.0, abs(W_self).max())
        print(
            f"[{label}] <|2K_tot+W_self|> = {mean_abs:.3e} "
            f"({mean_norm:.3e} × max|W_self|)"
        )

    plt.axhline(0, linestyle='--', linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel(r"$2(K_{\rm flow}+U_q+K_b) + W_{\rm self}$")
    plt.title("Virial Theorem Residual (ULDM + baryons)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    virial_residual_waves = 2.0 * (K_flow + U_quantum) + W_self
    plt.plot(time, virial_residual_waves, "--", alpha=0.6,
             label=f"{label}: waves only")


def plot_kinetic_energy_components_1(paths):
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
    #plot_energy_ratio(simulation_dirs)
    plot_total_energy(simulation_dirs)
    #plot_kinetic_energy_components(simulation_dirs)
    plot_virial_check(simulation_dirs)
    #plot_kinetic_energy_components_1(simulation_dirs)