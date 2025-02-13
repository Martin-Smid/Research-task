import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import hermite, factorial


def gaussian_packet(x, x_0, sigma_0):
    """Function which returns gaussian packet at position x.
    params are x, x_0 - mean, sigma_0 - standard deviation"""
    return cp.exp(-(x - x_0) ** 2 / (2 * sigma_0 ** 2)) / (cp.sqrt(2 * cp.pi) * sigma_0)

def normalize_wavefunction(psi, dx):
    """Normalize the wavefunction to ensure it stays normalized.
    intakes psi and dx"""
    dx_total = cp.prod(cp.array(dx))
    psi /= cp.sqrt(cp.sum(cp.abs(psi) ** 2) * dx_total)
    return psi

def quadratic_potential(wave_function_instance):
    """
    Compute the potential V(r) = 1/2 * omega^2 * r^2 for arbitrary dimensions
    using the properties of a Wave_function instance.

    Parameters:
        wave_function_instance (Wave_function): An instance of the Wave_function class.

    Returns:
        cp.ndarray: The potential computed on the spatial grid.
    """
    grids = wave_function_instance.grids  # Grids from the wave_function instance
    mass = wave_function_instance.mass  # Mass from the wave_function instance
    dim = wave_function_instance.dim
    omega = wave_function_instance.omega

    grid = cp.meshgrid(*grids, indexing='ij')

    # Compute r^2 from the multidimensional grid
    r2 = sum(g ** 2 for g in grid)

    # Return the computed quadratic potential
    return  dim * mass * omega ** 2 * r2



def coefficient_nd(n, m=1, omega=1, hbar=1):
    """Calculate the normalization constant for N dimensions."""

    beta = np.sqrt(m * omega / hbar)
    factor = (beta ** 2 / np.pi) ** (len(n) / 2)  # Normalization factor for N dimensions
    denom = np.sqrt(np.prod([2 ** ni * factorial(ni) for ni in n]))  # Hermite polynomial normalization
    return factor / denom


def energy_nd(n, omega=1, hbar=1):
    """
    Calculate the energy of an n-dimensional quantum state.

    Parameters:
        n (list): Quantum numbers for each dimension [nx, ny, nz, ...].
        omega (float): Oscillation frequency (default = 1).
        hbar (float): Reduced Planck's constant (default = 1).

    Returns:
        float: The total energy of the n-dimensional quantum state.
    """
    if not isinstance(n, list) or not all(isinstance(x, int) and x >= 0 for x in n):
        raise ValueError("n must be a list of non-negative integers representing quantum numbers.")

    return hbar * omega * (sum(n) + len(n) * 0.5)


def lin_harmonic_oscillator(wave_function_instance):
    """
    Create the wave function psi_0 for an N-dimensional linear harmonic oscillator.

    Parameters:
        wave_function_instance: Wave function parameters from class Wave_function.

    Returns:
        psi_0: The N-dimensional wave function.
    """
    # Extract information from Wave_function params
    grids = wave_function_instance.grids  # CuPy 1D grids for each dimension
    dim = wave_function_instance.dim
    means = wave_function_instance.means
    dx = wave_function_instance.dx
    mass = wave_function_instance.mass
    omega = wave_function_instance.omega
    h_bar = 1

    quantum_numbers = [0] * dim  # Quantum numbers for each dimension
    beta = np.sqrt((mass * omega) / h_bar)

    gaussian_factors = []
    hermite_polynomials = []

    # Loop over dimensions to compute shifted grid, Hermite polynomial, and Gaussian factor
    for i in range(dim):
        shifted_grid = grids[i] - means[i]
        shifted_grid_numpy = cp.asnumpy(shifted_grid)  # Convert to numpy array for Hermite computation
        H_numpy = hermite(quantum_numbers[i])(beta * shifted_grid_numpy)  # Compute Hermite polynomial
        H = cp.array(H_numpy)  # Convert back to CuPy array
        hermite_polynomials.append(H)
        gaussian_factors.append(cp.exp(-0.5 * (beta * shifted_grid) ** 2))

    # Compute the full wavefunction as a product of Gaussian factors and Hermite polynomials
    psi_0 = cp.ones_like(grids[0])
    for i in range(dim):
        psi_0 *= gaussian_factors[i] * hermite_polynomials[i]

    # Call coefficient_nd for normalization constant across dimensions
    coeff = coefficient_nd(quantum_numbers, mass, omega, h_bar)
    psi_0 *= coeff  # Apply the normalization constant to psi_0

    # Normalize the wavefunction using the provided normalize_wavefunction function
    dx_total = cp.prod(cp.array(dx))
    psi_0 = normalize_wavefunction(psi_0, dx_total)

    return psi_0



def plot_wave_function(wave_function_instance, time_step=None, dimension_slice=None):
    """
    Plots the magnitude of a wave function at a specified time step.

    Parameters:
        wave_function_instance (Wave_function): An instance of the Wave_function class.
        time_step (int, optional): Time step index to evolve and plot (default=None, current state of wave function).
        dimension_slice (tuple, optional): Used for 3D+ data. Tuple of dimension index and slice index.
                                           Example: (2, 50) would slice the 3rd dimension at index 50.

    Returns:
        None
    """
    # Ensure time_step is valid
    if time_step is not None:
        if not isinstance(time_step, int) or time_step < 0:
            raise ValueError("time_step must be a non-negative integer.")

        # Evolve to the requested time step if necessary
        current_step = wave_function_instance.num_steps_performed if hasattr(wave_function_instance,
                                                                             'num_steps_performed') else 0
        if time_step > current_step:
            # Evolve forward from the current state
            for _ in range(time_step - current_step):
                wave_function_instance.psi_0 = wave_function_instance.evolve_wavefunction_split_step(
                    wave_function_instance.psi_0)
            # Update the number of steps performed
            wave_function_instance.num_steps_performed = time_step
        elif time_step < current_step:
            raise ValueError("Cannot evolve backward. Reset the wave function, then re-evolve to the desired step.")

    # Use the current wave function state (already evolved to the requested step, if any)
    evolved_psi = wave_function_instance.psi_0

    # Convert to NumPy array for plotting
    psi_real = cp.asnumpy(np.abs(evolved_psi))  # |ψ| magnitude

    # Handle dimensional slicing for 3D+ data
    if wave_function_instance.dim > 2:
        if dimension_slice is None:
            raise ValueError(
                "For dimensions higher than 2, specify dimension_slice=(index, slice_position) to reduce dimensionality."
            )
        dim_idx, slice_idx = dimension_slice
        psi_real = psi_real.take(indices=slice_idx, axis=dim_idx)  # Extract slice along the provided dimension

    # Create the grid in host memory (NumPy format)
    grids = [cp.asnumpy(g) for g in wave_function_instance.grids]

    # 2D (or reduced to 2D after slicing) plots
    if wave_function_instance.dim == 2 or len(psi_real.shape) == 2:
        plt.pcolor(grids[0], grids[1], psi_real, shading='auto')
        plt.colorbar(label="|ψ| (Magnitude)")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title(f"Wave Function Magnitude at Time Step {time_step}")
    else:
        raise ValueError("Plotting for dimensions higher than 2 requires slicing to 2D with dimension_slice.")

    plt.show()






def plot_1D_wavefunction_evolution(wave_function, interval=20, save_file=None):
    """
    Animate the time evolution of a 1D wavefunction (integrated with Wave_function class),
    with an optional plot of the potential if present.

    Parameters:
        wave_function (Wave_function): Instance of the Wave_function class.
        interval (int): Interval between frames (in milliseconds).
        save_file (str, optional): Filepath to save the animation. Defaults to None.

    Returns:
        FuncAnimation: Animation object.
    """
    # Get grid and initial wavefunction
    x = wave_function.grids[0]  # 1D spatial grid
    x_numpy = cp.asnumpy(x)  # Convert to numpy for plotting
    psi = wave_function.psi_0.copy()  # Initial wavefunction

    # Compute the potential for plotting (if it's provided)
    if wave_function.potential is not None:
        V = wave_function.potential(wave_function)  # Pass the correct instance
        V_numpy = cp.asnumpy(V)  # Convert potential to numpy
    else:
        V_numpy = None  # No potential

    # Initialize the figure and plot elements
    fig, axs = plt.subplots(2, figsize=(10, 10))  # Only 2 subplots now
    fig.suptitle("1D Wavefunction Evolution", fontsize=14)

    # Configure subplots

    # Plot for |ψ|² (magnitude squared)
    axs[0].set_xlim(x_numpy[0], x_numpy[-1])
    axs[0].set_ylim(0, 7)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("|ψ|²")
    axs[0].legend()
    abs2_line, = axs[0].plot([], [], color="green", label="|ψ|²")
    axs[0].legend()

    # If a potential is provided, plot as dashed line on the |ψ|² subplot
    if V_numpy is not None:
        axs[0].plot(x_numpy, V_numpy, color="black", linestyle="--", label="V(x)")
        axs[0].legend()

    # Plot for Re(ψ) and Im(ψ)
    axs[1].set_xlim(x_numpy[0], x_numpy[-1])
    axs[1].set_ylim(-7, 7)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("ψ")
    real_line, = axs[1].plot([], [], color="blue", label="Re(ψ)")
    imag_line, = axs[1].plot([], [], color="red", label="Im(ψ)")
    axs[1].legend()

    def init():
        """Initialize the plot elements."""
        abs2_line.set_data([], [])
        real_line.set_data([], [])
        imag_line.set_data([], [])
        return abs2_line, real_line, imag_line

    def update(step):
        """Update the plot for each frame."""
        nonlocal psi
        # Pass step (as step_index) and total number of steps from wave_function
        psi = wave_function.evolve_wavefunction_split_step(psi, step, wave_function.num_steps)
        psi_real = cp.asnumpy(cp.real(psi))
        psi_imag = cp.asnumpy(cp.imag(psi))
        psi_abs2 = cp.asnumpy(cp.abs(psi) ** 2)

        # Update magnitude plot
        abs2_line.set_data(x_numpy, psi_abs2)

        # Update real and imaginary part plot
        real_line.set_data(x_numpy, psi_real)
        imag_line.set_data(x_numpy, psi_imag)

        fig.suptitle(f"1D Wavefunction Evolution - Step {step}")
        return abs2_line, real_line, imag_line

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=wave_function.num_steps, init_func=init, blit=True, interval=interval
    )

    # Save the animation if a save file is specified
    if save_file:
        anim.save(save_file, fps=20, extra_args=['-vcodec', 'libx264'])

    return anim





def plot_2D_wavefunction_evolution(wave_function, interval=20, save_file=None, N=1024):
    """
    Animate the time evolution of the wavefunction using the evolve method from the Wave_function class.

    Parameters:
        wave_function (Wave_function): Instance of the Wave_function class.
        interval (int): Interval between frames (in milliseconds).
        save_file (str, optional): Filepath to save the animation.
        N (int, optional): Grid size for visualization.

    Returns:
        FuncAnimation: Animation object displaying 2D wave evolution.
    """
    # Extract useful values from the Wave_function instance
    x_numpy = cp.asnumpy(wave_function.grids[0])  # Spatial grid in x
    y_numpy = cp.asnumpy(wave_function.grids[1])  # Spatial grid in y
    psi = wave_function.psi_0.copy()  # Initial wavefunction

    # Initialize the figure and plot elements
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Wavefunction Evolution", fontsize=14)

    # Configure plot elements
    ax.set_title('Wavefunction Magnitude Squared')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    im = ax.imshow(
        cp.asnumpy(cp.abs(psi) ** 2),  # Initial |ψ|²
        extent=[x_numpy[0], x_numpy[-1], y_numpy[0], y_numpy[-1]],
        origin='lower',
        cmap='viridis',
        vmin=0,  # Minimum fixed at 0
        vmax=1
    )
    fig.colorbar(im, ax=ax)

    def init():
        """Initialize the plot elements."""
        im.set_data(cp.asnumpy(cp.abs(psi) ** 2))  # Initial |ψ|²
        im.set_clim(vmin=0, vmax=cp.abs(psi).max() ** 2)  # Set initial color limits
        return [im]

    def update(step):
        """Update the plot for each frame."""
        nonlocal psi
        # Use evolve_wavefunction_split_step to evolve
        psi = wave_function.evolve_wavefunction_split_step(psi)
        psi_abs2 = cp.abs(psi) ** 2  # Compute |ψ|²

        # Update the heatmap with new |ψ|² data
        im.set_data(cp.asnumpy(psi_abs2))
        im.set_clim(vmin=0, vmax=float(1))  # Dynamically adjust vmax based on max(|ψ|²)
        fig.suptitle(f"Wavefunction Evolution - Step {step}")
        return [im]

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=wave_function.num_steps, init_func=init, blit=False, interval=interval
    )

    # Save the animation if a save file is specified
    if save_file:
        anim.save(save_file, fps=10, extra_args=['-vcodec', 'libx264'])

    return anim



def evolve_3D_wavefunction(psi, propagator_x, propagator_y, propagator_z, dx):
    """
    Perform the time evolution of a 3D wave function using cupy Fourier transforms.

    Parameters:
        psi (cp.ndarray): Initial wavefunction (3D array).
        propagator_x, propagator_y, propagator_z (cp.ndarray): K-space propagators.
        dx (float): Spatial step size.

    Returns:
        cp.ndarray: Updated 3D wavefunction after time evolution step.
    """
    # Perform the Fourier Transform of the wavefunction
    psi_k = cp.fft.fftn(psi)

    # Apply the time evolution propagators in k-space
    psi_k *= propagator_x
    psi_k *= propagator_y
    psi_k *= propagator_z

    # Transform back to real space
    psi = cp.fft.ifftn(psi)

    # Normalize the wavefunction
    psi = normalize_wavefunction(psi, dx)
    return psi



def plot_3D_wavefunction_evolution(x, y, z, psi_0, propagator_x, propagator_y, propagator_z, dx, num_steps,
                                   interval=500, save_file=None, max_val=None):
    """
    Animate the time evolution of a 3D wavefunction using a 3D scatter plot.
    The positions of points are fixed, and |ψ|² is displayed as color.

    Parameters:
        x, y, z (cp.ndarray): Spatial grids (x, y, z axes).
        psi_0 (cp.ndarray): Initial wavefunction (3D array, complex).
        propagator_x, propagator_y, propagator_z (cp.ndarray): K-space propagators.
        dx (float): Spatial step size (used for wavefunction normalization).
        num_steps (int): Total number of steps for the animation.
        interval (int): Interval (in ms) between animation frames.
        save_file (str, optional): Filepath for saving the animation.
        max_val (float, optional): Maximum value for |ψ|² colormap.

    Returns:
        FuncAnimation: Animated 3D scatter plot.
    """
    print("I dont work, so no point in trying")
    # Convert CuPy arrays to NumPy arrays for plotting
    x_numpy, y_numpy, z_numpy = cp.asnumpy(x), cp.asnumpy(y), cp.asnumpy(z)
    psi = psi_0.copy()

    # Compute the initial probability density (|ψ|²)
    psi_abs2 = cp.abs(psi) ** 2

    # Determine the global maximum value for color scaling
    if max_val is None:
        max_val = 0.1

    # Create the 3D grid coordinates (fixed points for the animation)
    X, Y, Z = np.meshgrid(x_numpy, y_numpy, z_numpy, indexing='ij')
    X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()

    # Flatten and convert the initial |ψ|² to NumPy
    probabilities = cp.asnumpy(psi_abs2).flatten()

    # Initialize the figure and 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Wavefunction Evolution (|ψ|²)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Initialize the scatter plot (positions are fixed, colors will change)
    scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=probabilities, cmap='viridis', alpha=0.8, s=1, vmax=max_val, vmin=0)

    # Set color bar to indicate |ψ|² values
    colorbar = fig.colorbar(scatter, ax=ax, label="|ψ|²")

    # Update function for the animation
    def update(step):
        nonlocal psi

        # Evolve the wavefunction in time
        psi = evolve_3D_wavefunction(psi, propagator_x, propagator_y, propagator_z, dx)
        psi_abs2 = cp.abs(psi) ** 2  # Compute the updated probability density

        # Update the scatter plot's colors (|ψ|² values)
        probabilities = cp.asnumpy(psi_abs2).flatten()
        scatter.set_array(probabilities)

        return scatter,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_steps, interval=interval, blit=False)

    # Save the animation to a file if requested
    if save_file:
        anim.save(save_file, writer='ffmpeg', fps=20)

    return anim




def plot_wave_equation_evolution(wave_function, interval, save_file=None, N=1024):
    """
    Plot the evolution of a wavefunction, supporting 1D, 2D, and 3D functions.

    Parameters:
        wave_function (Wave_function): Wave_function instance with simulation details.
        interval (int): Interval between frames in milliseconds.
        save_file (str, optional): Filepath to save the animation.
        N (int, optional): Grid size for visualization.

    Returns:
        FuncAnimation: Animation object for the wavefunction evolution.

    """
    sim_dimension = wave_function.dim

    if sim_dimension == 1:
        anim = plot_1D_wavefunction_evolution(
            wave_function=wave_function,
            interval=interval,
            save_file=save_file,
        )
        return anim
    elif sim_dimension == 2:
        anim = plot_2D_wavefunction_evolution(
            wave_function=wave_function,
            interval=interval,
            save_file=save_file,
            N=N,
        )
        return anim
    elif sim_dimension == 3:
        print(interval)
        anim = plot_3D_wavefunction_evolution(
            x=wave_function.grids[0],
            y=wave_function.grids[1],
            z=wave_function.grids[2],
            psi_0=wave_function.psi_0,
            propagator_x=wave_function.propagator,
            propagator_y=wave_function.propagator,
            propagator_z=wave_function.propagator,
            dx=wave_function.dx[0],
            num_steps=wave_function.num_steps,
            interval=interval,
            save_file=save_file,
        )
        return anim





