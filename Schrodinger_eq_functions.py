import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def normalize_wavefunction(psi, dx):
    # Normalize the wavefunction
    psi /= cp.sqrt(cp.sum(cp.abs(psi) ** 2) * dx)
    return psi

def evolve_1D_wavefunction(psi, propagator, dx):
    # Perform the Fourier Transform of the wavefunction
    psi_k = cp.fft.fft(psi)

    # Apply the time evolution propagator in k-space
    psi_k *= propagator

    # Transform back to real space
    psi_x = cp.fft.ifft(psi_k)

    # Normalize to ensure the wavefunction stays normalized
    psi_x = normalize_wavefunction(psi_x, dx)
    return psi_x


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
        anim.save(save_file, fps=30, extra_args=['-vcodec', 'libx264'])

    return anim


def gaussian_packet(x, x_0, sigma_0):
    return cp.exp(-(x - x_0) ** 2 / (2 * sigma_0 ** 2)) / (cp.sqrt(2 * cp.pi) * sigma_0)


def evolve_2D_wavefunction(psi, propagator_x, propagator_y, dx):
    # Perform the 2D Fourier Transform of the wavefunction
    psi_k = cp.fft.fft2(psi)

    # Apply the time evolution propagators in k-space
    psi_k *= propagator_x
    psi_k *= propagator_y

    # Transform back to real space
    psi = cp.fft.ifft2(psi_k)

    # Normalize to ensure the wavefunction stays normalized
    psi = normalize_wavefunction(psi, dx)
    return psi


def plot_2D_wavefunction_evolution(x, y, psi_0, propagator_x, propagator_y, dx, num_steps, interval=20, save_file=None,N=1024):
    """
    Animate the time evolution of the wavefunction.

    Parameters:
        x (cp.ndarray): Spatial grid.
        psi_0 (cp.ndarray): Initial wavefunction.
        propagator (cp.ndarray): Precomputed k-space propagator.
        dx (float): Spatial step size.
        num_steps (int): Number of time evolution steps.
        interval (int): Interval between frames (in milliseconds).
        save_file (str, optional): Filepath to save the animation. Defaults to None.

    Returns:
        FuncAnimation: Animation object displaying 2D wave evolution.
    """
    x_numpy = cp.asnumpy(x)
    psi = psi_0.copy()

    # Initialize the figure and plot elements
    fig, ax = plt.subplots( figsize=(8, 8))
    fig.suptitle("Wavefunction Evolution", fontsize=14)

    # Configure subplots
    titles = ['|ψ|²']
    ax.set_title('Wavefunction Magnitude Squared')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    im = ax.imshow(
        cp.asnumpy(cp.zeros((N, N))),  # Convert to NumPy array
        extent=[x_numpy[0], x_numpy[-1], x_numpy[0], x_numpy[-1]],
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=0.01
    )
    fig.colorbar(im, ax=ax)
    lines = [im]

    def init():
        """Initialize the plot elements."""
        lines[0].set_data(cp.asnumpy(cp.zeros((N, N))))  # Convert to NumPy array
        return lines

    def update(step):
        """Update the plot for each frame."""
        nonlocal psi
        psi = evolve_2D_wavefunction(psi, propagator_x, propagator_y, dx)
        psi_abs2 = cp.asnumpy(cp.abs(psi) ** 2)  # Convert to NumPy array

        lines[0].set_data(psi_abs2)  # Update heatmap with new magnitude squared data

        fig.suptitle(f"Wavefunction Evolution - Step {step}")
        return lines

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=num_steps, init_func=init, blit=True, interval=interval
    )

    # Save the animation if a save file is specified
    if save_file:
        anim.save(save_file, fps=30, extra_args=['-vcodec', 'libx264'])

    return anim