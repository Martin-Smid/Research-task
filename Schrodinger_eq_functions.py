import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from matplotlib.animation import FuncAnimation

def gaussian_packet(x, x_0, sigma_0):
    """Function which returns gaussian packet at position x.
    params are x, x_0 - mean, sigma_0 - standard deviation"""
    return cp.exp(-(x - x_0) ** 2 / (2 * sigma_0 ** 2)) / (cp.sqrt(2 * cp.pi) * sigma_0)

def normalize_wavefunction(psi, dx):
    """Normalize the wavefunction to ensure it stays normalized.
    intakes psi and dx"""
    psi /= cp.sqrt(cp.sum(cp.abs(psi) ** 2) * dx)
    return psi

def evolve_1D_wavefunction(psi, propagator, dx):
    """Perform the time evolution of the wavefunction using cupy Fourier transforms
    and evolving the wavefunction in k-space using its propagator."""
    # Perform the Fourier Transform of the wavefunction
    psi_k = cp.fft.fft(psi)

    # Apply the time evolution propagator in k-space
    psi_k *= propagator

    # Transform back to real space
    psi_x = cp.fft.ifft(psi_k)

    # Normalize to ensure the wavefunction stays normalized
    psi_x = normalize_wavefunction(psi_x, dx)
    return psi_x

def evolve_2D_wavefunction(psi, propagator_x, propagator_y, dx):
    """Perform the time evolution of a 2D wave function using cupy Fourier transforms"""
    psi_k = cp.fft.fft2(psi)

    # Apply the time evolution propagators in k-space
    psi_k *= propagator_x
    psi_k *= propagator_y

    # Transform back to real space
    psi = cp.fft.ifft2(psi_k)

    # Normalize to ensure the wavefunction stays normalized
    psi = normalize_wavefunction(psi, dx)
    return psi





def plot_1D_wavefunction_evolution(x, vlna,dx, num_steps, interval=20, save_file=None,):
    """
    Animate the time evolution of the wavefunction (integrated with Wave_function class).

    Parameters:
        x (cp.ndarray): Spatial grid.
        vlna (Wave_function): Instance of Wave_function class.
        num_steps (int): Number of time evolution steps.
        interval (int): Interval between frames (in milliseconds).
        save_file (str, optional): Filepath to save the animation. Defaults to None.

    Returns:
        FuncAnimation: Animation object.
    """
    x_numpy = cp.asnumpy(x)
    psi = vlna.psi_0.copy()

    # Initialize the figure and plot elements
    fig, axs = plt.subplots(3, figsize=(8, 8))
    fig.suptitle("Wavefunction Evolution", fontsize=14)

    # Configure subplots
    labels = ['|ψ|²', 'Re(ψ)', 'Im(ψ)']
    colors = ['green', 'blue', 'red']
    lines = []
    for ax, label, color in zip(axs, labels, colors):
        ax.set_xlim(x_numpy[0], x_numpy[-1])
        ax.set_ylim(-7, 7)
        ax.set_xlabel('x')
        ax.set_ylabel(label)
        line, = ax.plot([], [], color=color, label=label)
        ax.legend()
        lines.append(line)

    def init():
        """Initialize the plot elements."""
        for line in lines:
            line.set_data([], [])
        return lines

    def update(step):
        """Update the plot for each frame."""
        nonlocal psi
        psi = evolve_1D_wavefunction(psi, vlna.propagator, dx)
        psi_real = cp.asnumpy(cp.real(psi))
        psi_imag = cp.asnumpy(cp.imag(psi))
        psi_abs2 = cp.asnumpy(cp.abs(psi) ** 2)

        lines[0].set_data(x_numpy, psi_abs2)
        lines[1].set_data(x_numpy, psi_real)
        lines[2].set_data(x_numpy, psi_imag)

        fig.suptitle(f"Wavefunction Evolution - Step {step}")
        return lines

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=num_steps, init_func=init, blit=True, interval=interval
    )

    # Save the animation if a save file is specified
    if save_file:
        anim.save(save_file, fps=10, extra_args=['-vcodec', 'libx264'])

    return anim






def plot_2D_wavefunction_evolution(x, y, psi_0, propagator_x, propagator_y, dx, num_steps, interval=20, save_file=None,
                                   N=1024):
    """
    Animate the time evolution of the wavefunction.

    Parameters:
        x (cp.ndarray): Spatial grid.
        psi_0 (cp.ndarray): Initial wavefunction.
        propagator_x (cp.ndarray): Precomputed k-space propagator in x.
        propagator_y (cp.ndarray): Precomputed k-space propagator in y.
        dx (float): Spatial step size.
        num_steps (int): Number of time evolution steps.
        interval (int): Interval between frames (in milliseconds).
        save_file (str, optional): Filepath to save the animation.
        N (int, optional): Grid size for visualization.

    Returns:
        FuncAnimation: Animation object displaying 2D wave evolution.
    """
    x_numpy = cp.asnumpy(x)
    psi = psi_0.copy()

    # Initialize the figure and plot elements
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Wavefunction Evolution", fontsize=14)

    # Configure plot elements
    ax.set_title('Wavefunction Magnitude Squared')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    im = ax.imshow(
        cp.asnumpy(cp.abs(psi) ** 2),  # Initial |ψ|²
        extent=[x_numpy[0], x_numpy[-1], x_numpy[0], x_numpy[-1]],
        origin='lower',
        cmap='viridis',
        vmin=0,  # Minimum fixed at 0
        vmax=cp.abs(psi).max() ** 2  # Dynamically updated later
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
        psi = evolve_2D_wavefunction(psi, propagator_x, propagator_y, dx)
        psi_abs2 = cp.abs(psi) ** 2  # Compute |ψ|²

        im.set_data(cp.asnumpy(psi_abs2))  # Update heatmap with new |ψ|² data
        im.set_clim(vmin=0, vmax=float(0.15))  # Dynamically adjust vmax based on max(|ψ|²)

        fig.suptitle(f"Wavefunction Evolution - Step {step}")
        return [im]

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=num_steps, init_func=init, blit=False, interval=interval
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
            x=wave_function.grids[0],
            vlna=wave_function,
            dx=wave_function.dx[0],
            num_steps=wave_function.num_steps,
            interval=interval,
            save_file=save_file,
        )
        return anim
    elif sim_dimension == 2:
        anim = plot_2D_wavefunction_evolution(
            x=wave_function.grids[0],
            y=wave_function.grids[1],
            psi_0=wave_function.psi_0,
            propagator_x=wave_function.propagator,
            propagator_y=wave_function.propagator,
            dx=wave_function.dx[0],
            num_steps=wave_function.num_steps,
            interval=interval,
            save_file=save_file,
            N=N
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
