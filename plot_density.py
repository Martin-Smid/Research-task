import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




simulation_dirs = [
    'resources/data/simulation_20250710_173709',

]

specific_times = [0, 4, 8]



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
                    popt, _ = curve_fit(nfw_profile, bin_centers[fit_mask], rho_avg[fit_mask],
                                        p0=[1.0, 1.0], maxfev=10000)
                    fitted_r = np.logspace(np.log10(bin_centers[fit_mask].min()),
                                           np.log10(bin_centers[fit_mask].max()), 200)
                    fitted_density = nfw_profile(fitted_r, *popt)

                    # Plot NFW fit with same color but dashed line
                    plt.plot(fitted_r, fitted_density, '--', color=color, lw=2, alpha=0.8,
                             label=f"{sim_name} - NFW (ρ₀={popt[0]:.2f}, rₛ={popt[1]:.2f})")
                except RuntimeError:
                    print(f"[!] NFW fit failed for {sim_name} at t={current_time:.2f}")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Radius r [kpc]")
        plt.ylabel("Normalized Density ρ(r)")

        # Set title based on number of simulations
        if len(sim_data_list) == 1:
            plt.title(f"Radial Density Profile at t = {current_time:.2f}")
        else:
            plt.title(f"Radial Density Profiles Comparison at t = {current_time:.2f}")

        plt.legend()
        plt.tight_layout()
        plt.show()


plot_density_profiles(simulation_dirs, times=specific_times)