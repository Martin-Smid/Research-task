import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




simulation_dirs = [

    'resources/data/simulation_20250805_214819',

]

specific_times = [0, 10, 15]



def nfw_profile(r, rho_0, r_s):
    """NFW profile function for fitting."""
    return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2)


def plot_density_profiles(simulation_dirs, times=None):
    """
    Plot radial density profiles for multiple simulation directories.
    If multiple directories are provided, plots are shown on the same canvas.
    If times are specified, creates separate plots for each time.

    Parameters:
        simulation_dirs (list): List of simulation directory paths
        times (list, optional): List of specific times to plot. If None, plots all available times.
    """
    # Dictionary to store data grouped by time
    time_data = {}

    for sim_dir in simulation_dirs:
        density_profiles_dir = os.path.join(sim_dir, "density_profiles")

        if not os.path.exists(density_profiles_dir):
            print(f"[!] Skipping: density_profiles directory not found in {sim_dir}")
            continue

        # Get all CSV files in the density_profiles directory
        csv_files = [f for f in os.listdir(density_profiles_dir) if f.endswith('.csv')]

        if not csv_files:
            print(f"[!] No CSV files found in {density_profiles_dir}")
            continue

        # Sort files by time if possible
        csv_files.sort()

        # Filter by specific times if provided
        if times is not None:
            filtered_files = []
            for time_val in times:
                # Look for files matching the time (with some tolerance for floating point)
                for csv_file in csv_files:
                    if f"time_{time_val:.6f}" in csv_file:
                        filtered_files.append(csv_file)
                        break
            csv_files = filtered_files

        # Process each density profile
        for csv_file in csv_files:
            file_path = os.path.join(density_profiles_dir, csv_file)

            # Extract time from filename
            try:
                time_str = csv_file.split("time_")[1].split(".csv")[0]
                current_time = float(time_str)
            except (IndexError, ValueError):
                current_time = 0.0

            # Read density data
            try:
                data = np.loadtxt(file_path, delimiter=",", skiprows=1)
                bin_centers = data[:, 0]
                rho_avg = data[:, 1]
            except Exception as e:
                print(f"[!] Failed to read {file_path}: {e}")
                continue

            # Group data by time
            if current_time not in time_data:
                time_data[current_time] = []

            sim_name = os.path.basename(os.path.normpath(sim_dir))
            time_data[current_time].append({
                'sim_name': sim_name,
                'bin_centers': bin_centers,
                'rho_avg': rho_avg
            })

    # Create plots for each time
    for current_time, sim_data_list in time_data.items():
        plt.figure(figsize=(8, 6))

        # Define colors for different simulations
        colors = plt.cm.tab10(np.linspace(0, 1, len(sim_data_list)))

        for i, sim_data in enumerate(sim_data_list):
            sim_name = sim_data['sim_name']
            bin_centers = sim_data['bin_centers']
            rho_avg = sim_data['rho_avg']
            color = colors[i]

            # Plot data points
            plt.scatter(bin_centers, rho_avg, s=15, color=color,
                        label=f"{sim_name} - Data", alpha=0.7)

            # Attempt NFW fit
            fit_mask = (rho_avg > 0) & (bin_centers > 0)

            if np.sum(fit_mask) > 2:  # Need at least 3 points for fitting
                try:
                    fitted_r, fitted_density, popt = fit_soliton_nfw_profile(bin_centers[fit_mask], rho_avg[fit_mask])
                    reps_fit, rc_fit, rs_fit, logrhoc_fit = popt

                    plt.plot(fitted_r, fitted_density, '--', color=color, lw=2, alpha=0.8,
                             label=f"{sim_name} - Sol+NFW (rₑ={reps_fit:.2f}, r_c={rc_fit:.2f}, rₛ={rs_fit:.2f})")

                except RuntimeError as e:
                    print(f"[!] Soliton+NFW fit failed for {sim_name} at t={current_time:.2f}: {e}")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Radius r [kpc]")
        plt.ylabel("Density ρ(r) [M☉/kpc³]")

        # Set title based on number of simulations
        if len(sim_data_list) == 1:
            plt.title(f"Radial Density Profile at t = {current_time:.2f}")
        else:
            plt.title(f"Radial Density Profiles Comparison at t = {current_time:.2f}")

        plt.legend()
        plt.tight_layout()
        plt.show()

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def fit_soliton_nfw_profile(r, rho):
    """
    Fits the log-log soliton + NFW hybrid profile to radial density data.
    Returns the fitted radius grid and fitted log-density.
    """
    def log_sol_nfw(log_r, reps, rc, rs, logrhoc):
        r = 10 ** log_r
        beta = 3
        rhos = 10 ** logrhoc * (reps / rs) * (1 + reps / rs) ** (beta - 1)
        rhos /= (1 + 0.091 * (reps / rc) ** 2) ** 8

        rho_out = np.where(
            r <= reps,
            10 ** logrhoc / (1 + 0.091 * (r / rc) ** 2) ** 8,
            rhos / (((r + 1e-99) / rs) * (1 + (r / rs)) ** (beta - 1))
        )
        return np.log10(rho_out)

    # Initial guess
    p0 = [2.85, 1.5, 15, 9]  # [reps, rc, rs, log10(rhoc)]
    bounds = (
        [0.0, 0.0, 1, 5],
        [3.0, 3.0, 200, 25]
    )

    # Filter valid values
    mask = (r > 0) & (rho > 0)
    r_valid = r[mask]
    rho_valid = rho[mask]

    if len(r_valid) < 4:
        raise RuntimeError("Not enough valid points for soliton+NFW fit.")

    # Interpolate to uniform log-log spacing
    log_r = np.log10(r_valid)
    log_rho = np.log10(rho_valid)
    interp_func = interp1d(log_r, log_rho, kind='linear', fill_value="extrapolate")
    log_r_uniform = np.linspace(log_r.min(), log_r.max(), 100)
    log_rho_uniform = interp_func(log_r_uniform)

    # Fit
    popt, pcov = curve_fit(
        log_sol_nfw,
        log_r_uniform,
        log_rho_uniform,
        p0=p0,
        bounds=bounds,
        maxfev=200000
    )

    reps_fit, rc_fit, rs_fit, logrhoc_fit = popt
    fitted_r = 10 ** log_r_uniform
    fitted_log_rho = log_sol_nfw(log_r_uniform, *popt)

    return fitted_r, 10 ** fitted_log_rho, popt



plot_density_profiles(simulation_dirs, times=specific_times)