from resources.Functions.Schrodinger_eq_functions import *
import pandas as pd
import matplotlib.pyplot as plt

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

    filename = simulation_class_instance.max_vals_filename

    # Načti s MultiIndex ve sloupcích (dva řádky záhlaví), ignoruj komentáře
    data = pd.read_csv(filename, comment='#', header=[0, 1], index_col=0)

    # Vytvoř seznam legendárních popisků (N a spin)
    labels = [f"N = {n.replace('N', '')}, spin = {s.replace('s=', '')}"
              for (n, s) in data.columns]

    # Normalizuj každou složku
    for col in data.columns:
        first_val = data[col].iloc[0]
        data[col] = (data[col] / first_val) ** 0.25

    # Plotting
    plt.figure(figsize=(10, 6))

    for col, label in zip(data.columns, labels):
        plt.plot(data.index, data[col], label=label)

    # Osa a legenda
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("Normalized Max Values", fontsize=18)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import itertools

def saturation_func(t, a, b, c):
    return a + b * t / (1 + c * t)

def compute_tau99(a, b, c):
    lambda_inf = a + b / c
    lambda_99 = 0.99 * lambda_inf
    A = lambda_99 - a
    denom = b - c * A
    tau_99 = A / denom if denom != 0 else np.nan
    return lambda_inf, tau_99


def plot_lambda_rho_evolution(file_name):
    # Read file exactly like in your original function
    data = pd.read_csv(file_name, comment='#', header=[0, 1], index_col=0)

    # Generate labels
    labels = [f"N = {n.replace('N', '')}, spin = {s.replace('s=', '')}"
              for (n, s) in data.columns]

    # Setup plotting
    plt.figure(figsize=(10, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = itertools.cycle(['o', 'x', '^', 's', 'D', 'v', 'P', '*', 'h', '+'])

    for col, label, color in zip(data.columns, labels, itertools.cycle(color_cycle)):
        y = data[col].astype(float)
        x = data.index.astype(float)

        # Normalize: λρ = (ρ_f / ρ_i)^0.25
        rho_i = y.iloc[0]
        lambda_rho = (y / rho_i) ** 0.25

        # Remove NaNs
        valid = lambda_rho.notnull()
        x_valid = x[valid]
        y_valid = lambda_rho[valid]

        # Fit
        try:
            popt, _ = curve_fit(saturation_func, x_valid, y_valid, maxfev=10000)
            a, b, c = popt
            lambda_inf, tau99 = compute_tau99(a, b, c)

            # Plot data
            plt.plot(x_valid, y_valid, marker=next(markers),markersize=2, linestyle='None',
                     color=color, label=f"{label}\nλ∞={lambda_inf:.2f}, τ₉₉={tau99:.2f}")

            # Plot fit
            plt.plot(x_valid, saturation_func(x_valid, *popt), linestyle='-', color='black')

            # Plot saturation line
            plt.axhline(lambda_inf, color=color, linestyle='--', linewidth=1)
        except RuntimeError:
            print(f"Fit failed for {label}")
            continue

    # Axes and labels
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel(r"$\lambda_\rho(t)$", fontsize=18)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


#plot_lambda_rho_evolution(r'C:\projekty\Research-task\resources\data\max_values.csv')

#plot_max_values('/home/martin/ploty/max_values_4.csv')



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import glob
import re
from pathlib import Path


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


import os, re, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

def plot_wave_function_panel(
    snapshot_dir,
    wf_number,
    times=None,                 # e.g. [0.100000, 0.200000, 0.500000, 1.000000]
    z_index=None,               # for 3D -> choose slice; None = middle
    ncols=2,
    log_scale=True,             # use LogNorm across all panels
    cmap="inferno",
    x_range=None, y_range=None, grid_spacing=None,
    figsize=(10, 8),
    fontsize=14,
    save_path=None,             # e.g. ".../wf0_panel.pdf" (great for LaTeX)
    dpi=300,
    show=False
):
    """
    Make a multi-panel figure of |ψ|^2 snapshots with a single shared colorbar.
    If `times` is None, uses the earliest N that fit the grid.
    """

    pattern = f"wf_{wf_number}_snapshot_at_time_*.npy"
    files = glob.glob(os.path.join(snapshot_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No snapshot files for wf_{wf_number} in {snapshot_dir}")

    # (time, path) pairs
    time_file_pairs = []
    for fp in files:
        m = re.search(r"time_([\d.]+)\.npy", os.path.basename(fp))
        if m:
            time_file_pairs.append((float(m.group(1)), fp))
    time_file_pairs.sort(key=lambda x: x[0])
    if not time_file_pairs:
        raise ValueError("No valid time tags in filenames.")

    # Filter to requested times (matching on exact float string is brittle; we allow tolerance)
    if times is not None:
        sel = []
        want = list(times)
        for t in want:
            # find closest available
            closest = min(time_file_pairs, key=lambda p: abs(p[0] - t))
            if abs(closest[0] - t) < 1e-9:  # exact (as your filenames are formatted)
                sel.append(closest)
            else:
                # If not exact, still take the closest within a small tolerance
                if abs(closest[0] - t) < 1e-6:
                    sel.append(closest)
                else:
                    raise ValueError(f"No snapshot matching time {t} (closest is {closest[0]}).")
        time_file_pairs = sel

    # Decide grid size
    n = len(time_file_pairs)
    if n == 0:
        raise ValueError("No snapshots selected.")
    if ncols < 1:
        ncols = 1
    nrows = int(np.ceil(n / ncols))

    # Preload data and compute global vmin/vmax from positive values for shared colorbar
    slices = []
    nonzero_min = np.inf
    global_max = 0.0

    for t, fp in time_file_pairs:
        psi = np.load(fp)
        vals = np.abs(psi) ** 2

        if vals.ndim == 3:
            zi = z_index if z_index is not None else vals.shape[2] // 2
            sl = vals[:, :, zi]
        elif vals.ndim == 2:
            sl = vals
        else:
            raise ValueError(f"Unsupported dimensionality: {vals.ndim}D")

        slices.append((t, sl))

        nz = sl[sl > 0]
        if nz.size:
            nonzero_min = min(nonzero_min, nz.min())
        global_max = max(global_max, sl.max())

    if not np.isfinite(nonzero_min) or global_max <= 0:
        # fallback if all zeros (unlikely but safe)
        nonzero_min, global_max = 1e-30, 1.0

    # Coordinate grids
    ny, nx = slices[0][1].shape
    if x_range is not None and y_range is not None:
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
    elif grid_spacing is not None:
        x = np.arange(nx) * grid_spacing
        y = np.arange(ny) * grid_spacing
    else:
        x = np.arange(nx)
        y = np.arange(ny)

    extent = [x.min(), x.max(), y.min(), y.max()]

    # Shared normalization
    if log_scale:
        norm = LogNorm(vmin=nonzero_min, vmax=global_max)
        cbar_fmt = "%.1e"
    else:
        norm = Normalize(vmin=0.0, vmax=global_max)
        cbar_fmt = None

    # Figure & axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(
        right=0.85,  # space for the colorbar
        hspace=0.35,  # vertical gap between rows (default is ~0.2)
        wspace=0.25  # optional: horizontal gap between columns
    )


    im = None
    for i, (t, sl) in enumerate(slices):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        im = ax.imshow(
            sl.T, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="auto"
        )
        ax.set_title(f"t = {t:.1f}", fontsize=fontsize)
        ax.set_xlabel("x", fontsize=fontsize)
        ax.set_ylabel("y", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize-2)

    # Hide any unused axes
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    # One shared colorbar
    #cax = fig.add_axes([0.92, 0.15, 0.2, 0.7])  # manual to keep right margin tight in LaTeX
    fig.subplots_adjust(right=0.85)
    cbar = fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.046,  # thickness of colorbar
        pad=0.04  # gap between plots and colorbar
    )
    cbar.ax.tick_params(labelsize=fontsize - 2)
    cbar.set_label(r"$|\psi|^2$", fontsize=fontsize)


    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved multi-panel figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
