from resources.Functions.Schrodinger_eq_functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import glob
import re
from pathlib import Path

def calculate_errors_between_num_and_analytical_evolution(wave_function, time_step):
    """
    Compares the numerical and analytical solutions of a wave function at a given time step
    and computes the error in both the wave function and the probability density.

    Parameters:
        wave_function (Wave_function): Instance of the Wave_function class, representing the simulated wavefunction.
        time_step (int): The time step at which the comparison will be made.

    Returns:
        dict: A dictionary with the following keys:
            - "numerical_psi": Numerical wave function at this time step.
            - "analytical_psi": Analytical wave function at this time step.
            - "wave function_error": The error between analytical and numerical wave functions.
            - "probability_density_error": The error between the probability densities.
            - "analytical_norm": Norm of the analytical solution at this step.
            - "numerical_norm": Norm of the numerical solution at this step.
    """
    # Get the time corresponding to the time step
    t = time_step * wave_function.h

    # Analytical solution at time t
    analytical_wave = cp.asnumpy(
        wave_function.psi_0 * cp.exp(-1j * energy_nd([0] * wave_function.dim, omega=1, hbar=1) * t)
    )

    # Numerical solution at time t
    numerical_wave = cp.asnumpy(wave_function.wave_function_at_time(t))

    # Compute errors between analytical and numerical solutions
    wavefunction_error = np.linalg.norm(analytical_wave - numerical_wave) / np.linalg.norm(analytical_wave)

    # Compute probability densities
    prob_density_analytical = np.abs(analytical_wave) ** 2
    prob_density_numerical = np.abs(numerical_wave) ** 2

    # Compute error in probability density
    probability_density_error = np.linalg.norm(prob_density_analytical - prob_density_numerical) / np.linalg.norm(
        prob_density_analytical)

    # Compute the norms of the wavefunctions for diagnostics
    analytical_norm = np.linalg.norm(prob_density_analytical)
    numerical_norm = np.linalg.norm(prob_density_numerical)

    return {
        "numerical_psi": numerical_wave,
        "analytical_psi": analytical_wave,
        "wave_function_error": wavefunction_error,
        "probability_density_error": probability_density_error,
        "analytical_norm": analytical_norm,
        "numerical_norm": numerical_norm
    }


def plot_max_values_on_N(simulation_class_instance):
    import pandas as pd
    import matplotlib.pyplot as plt

    # File name
    filename = simulation_class_instance.max_vals_filename

    # Read the CSV file, skipping comment lines (lines starting with '#')
    data = pd.read_csv(filename, comment='#')

    # Extract the first row (N values) for column names
    n_values = data.columns[1:]  # Skip the first column (time step)
    n_values = [float(n) for n in n_values]  # Convert to float for proper naming

    # Rename columns for clarity
    data.columns = ['Time Step'] + [f"N = {n}" for n in n_values]

    # Normalize the data: (value / first_value) - 1
    for col in data.columns[1:]:
        first_val = data[col].iloc[0]
        data[col] = (data[col] / first_val) ** 0.25  # λρ(t) = (ρc^f / ρc^i)^{1/4}


    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot each column of data (except the first column, which is the x-axis)
    for col in data.columns[1:]:
        plt.plot(data['Time Step'], data[col], label=col)

    # Add labels, title, and legend
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("Normalized Max Values", fontsize=18)
    plt.legend(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    # Show the plot
    plt.savefig("resources/data/max_vals.png", dpi=300)





def plot_wave_function_snapshots(snapshot_dir, wf_number, z_index=None,
                                 save_plots=True, show_plots=False,
                                 x_range=None, y_range=None, grid_spacing=None):
    """
    Plot wave function snapshots from saved .npy files.

    Parameters:
    -----------
    snapshot_dir : str
        Path to the directory containing snapshot files
    wf_number : int
        Wave function number to plot (corresponds to wf_{number}_snapshot_*.npy)
    z_index : int, optional
        Z-slice index for 3D data. If None, assumes 2D data or takes middle slice
    save_plots : bool, default=True
        Whether to save plot images
    show_plots : bool, default=False
        Whether to display plots interactively
    x_range : tuple, optional
        (x_min, x_max) for spatial coordinates. If None, uses index coordinates
    y_range : tuple, optional
        (y_min, y_max) for spatial coordinates. If None, uses index coordinates
    grid_spacing : float, optional
        Spacing between grid points. If None, assumes unit spacing

    Returns:
    --------
    dict : Dictionary with times as keys and corresponding data arrays as values
    """

    # Find all snapshot files for the specified wave function
    pattern = f"wf_{wf_number}_snapshot_at_time_*.npy"
    snapshot_files = glob.glob(os.path.join(snapshot_dir, pattern))

    if not snapshot_files:
        raise FileNotFoundError(f"No snapshot files found for wave function {wf_number} in {snapshot_dir}")

    # Extract times and sort files by time
    time_file_pairs = []
    for file_path in snapshot_files:
        filename = os.path.basename(file_path)
        # Extract time from filename using regex
        time_match = re.search(r'time_([\d.]+)\.npy', filename)
        if time_match:
            time = float(time_match.group(1))
            time_file_pairs.append((time, file_path))

    # Sort by time
    time_file_pairs.sort(key=lambda x: x[0])

    if not time_file_pairs:
        raise ValueError("Could not extract time information from snapshot filenames")

    print(f"Found {len(time_file_pairs)} snapshots for wave function {wf_number}")
    print(f"Time range: {time_file_pairs[0][0]:.6f} to {time_file_pairs[-1][0]:.6f}")

    # Create output directory for plots if saving
    if save_plots:
        plot_dir = os.path.join(snapshot_dir, f"plots_wf_{wf_number}")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Plots will be saved to: {plot_dir}")

    # Store data for return
    wave_data = {}

    # Process each snapshot
    for i, (time, file_path) in enumerate(time_file_pairs):
        print(f"Processing snapshot {i + 1}/{len(time_file_pairs)}: t={time:.6f}")

        # Load the wave function data
        psi = np.load(file_path)

        # Calculate probability density |ψ|²
        wave_values = np.abs(psi) ** 2
        wave_data[time] = wave_values.copy()

        # Handle dimensionality
        if wave_values.ndim == 3:
            # 3D data - take a slice
            if z_index is None:
                z_index = wave_values.shape[2] // 2  # Middle slice
            wave_slice = wave_values[:, :, z_index]
            title_suffix = f" (z-slice at index {z_index})"
        elif wave_values.ndim == 2:
            # 2D data
            wave_slice = wave_values
            title_suffix = ""
        else:
            raise ValueError(f"Unsupported data dimensionality: {wave_values.ndim}D")

        # Set up coordinate meshes
        ny, nx = wave_slice.shape

        if x_range is not None and y_range is not None:
            x = np.linspace(x_range[0], x_range[1], nx)
            y = np.linspace(y_range[0], y_range[1], ny)
        elif grid_spacing is not None:
            x = np.arange(nx) * grid_spacing
            y = np.arange(ny) * grid_spacing
        else:
            x = np.arange(nx)
            y = np.arange(ny)

        x_mesh_2d, y_mesh_2d = np.meshgrid(x, y)

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Handle zero values for log scale
        nonzero_values = wave_slice[wave_slice > 0]
        if len(nonzero_values) == 0:
            print(f"Warning: All values are zero at time {time}")
            plt.imshow(wave_slice.T, origin="lower", cmap="inferno")
            plt.colorbar(label="|ψ|²")
        else:
            # Create logarithmic levels
            vmin = nonzero_values.min()
            vmax = wave_slice.max()
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 64)

            # Create contour plot
            contour = plt.contourf(x_mesh_2d, y_mesh_2d, wave_slice.T,
                                   levels=levels, cmap="inferno", norm=LogNorm())
            plt.colorbar(contour, label="|ψ|²", format="%.2e")

        plt.title(f"Wave Function {wf_number} Probability Density at Time {time:.6f}{title_suffix}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)

        # Save plot if requested
        if save_plots:
            plot_filename = f"wf_{wf_number}_timestep_{time:.6f}.jpg"
            plot_path = os.path.join(plot_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()

    print(f"Completed processing {len(time_file_pairs)} snapshots")
    return wave_data


def plot_multiple_wave_functions(snapshot_dir, wf_numbers, z_index=None,
                                 save_plots=True, show_plots=False):
    """
    Plot multiple wave functions for comparison.

    Parameters:
    -----------
    snapshot_dir : str
        Path to the directory containing snapshot files
    wf_numbers : list
        List of wave function numbers to plot
    z_index : int, optional
        Z-slice index for 3D data
    save_plots : bool
        Whether to save plot images
    show_plots : bool
        Whether to display plots
    """

    # Get all available times from metadata if it exists
    metadata_path = os.path.join(snapshot_dir, "metadata.txt")
    times = []

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            content = f.read()
            if "Accessible times:" in content:
                times_line = content.split("Accessible times:")[-1].strip()
                times = [float(t.strip()) for t in times_line.split(',')]

    if not times:
        # Fallback: get times from first wave function's files
        pattern = f"wf_{wf_numbers[0]}_snapshot_at_time_*.npy"
        snapshot_files = glob.glob(os.path.join(snapshot_dir, pattern))
        for file_path in snapshot_files:
            filename = os.path.basename(file_path)
            time_match = re.search(r'time_([\d.]+)\.npy', filename)
            if time_match:
                times.append(float(time_match.group(1)))
        times.sort()

    if save_plots:
        comparison_dir = os.path.join(snapshot_dir, "comparison_plots")
        os.makedirs(comparison_dir, exist_ok=True)

    # Plot each time step with all wave functions
    for time in times:
        fig, axes = plt.subplots(1, len(wf_numbers), figsize=(5 * len(wf_numbers), 4))
        if len(wf_numbers) == 1:
            axes = [axes]

        for i, wf_num in enumerate(wf_numbers):
            # Load wave function data
            pattern = f"wf_{wf_num}_snapshot_at_time_{time:.6f}.npy"
            if time == 0:
                pattern = f"wf_{wf_num}_snapshot_at_time_{time:.0f}.npy"


            file_path = os.path.join(snapshot_dir, pattern)

            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            psi = np.load(file_path)
            wave_values = np.abs(psi) ** 2

            # Handle dimensionality
            if wave_values.ndim == 3:
                if z_index is None:
                    z_index = wave_values.shape[2] // 2
                wave_slice = wave_values[:, :, z_index]
            else:
                wave_slice = wave_values

            # Plot
            nonzero_values = wave_slice[wave_slice > 0]
            if len(nonzero_values) > 0:
                vmin = nonzero_values.min()
                vmax = wave_slice.max()
                levels = np.logspace(np.log10(vmin), np.log10(vmax), 32)

                im = axes[i].contourf(wave_slice.T, levels=levels,
                                      cmap="inferno", norm=LogNorm())
                plt.colorbar(im, ax=axes[i], format="%.1e")
            else:
                axes[i].imshow(wave_slice.T, cmap="inferno")

            axes[i].set_title(f"WF {wf_num}")
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")

        plt.suptitle(f"Wave Functions Comparison at Time {time:.6f}")
        plt.tight_layout()

        if save_plots:
            plt.savefig(os.path.join(comparison_dir, f"comparison_t_{time:.6f}.jpg"),
                        dpi=150, bbox_inches='tight')

        if show_plots:
            plt.show()
        else:
            plt.close()



    plt.savefig("resources/data/max_vals.png", dpi=300)