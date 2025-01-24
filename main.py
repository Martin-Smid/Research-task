import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def normalize_wavefunction(psi, dx):
    # Normalize the wavefunction
    psi /= cp.sqrt(cp.sum(cp.abs(psi) ** 2) * dx)
    return psi


def gaussian_packet(x, x_0, sigma_0):
    return cp.exp(-(x - x_0) ** 2 / (2 * sigma_0 ** 2)) / (cp.sqrt(2 * cp.pi) * sigma_0)


def evolve_wavefunction(psi, propagator, dx):
    # Perform the Fourier Transform of the wavefunction
    psi_k = cp.fft.fft(psi)

    # Apply the time evolution propagator in k-space
    psi_k *= propagator

    # Transform back to real space
    psi_x = cp.fft.ifft(psi_k)

    # Normalize to ensure the wavefunction stays normalized
    psi_x = normalize_wavefunction(psi_x, dx)
    return psi_x


def plot_wavefunction_evolution(x, psi_0, propagator, dx, num_steps, interval=20, save_file=None):
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
        FuncAnimation: Animation object.
    """
    x_numpy = cp.asnumpy(x)
    psi = psi_0.copy()

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
        psi = evolve_wavefunction(psi, propagator, dx)
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


# Setup parameters
a, b = -1, 1  # Domain boundaries
N = 1024  # Number of spatial points
dx = (b - a) / (N - 1)
x = cp.linspace(a, b, N)  # Spatial grid
total_time = 10
h = 0.01  # Propagation parameter (time step size)
num_steps = int(total_time / h)

# Initialize wavefunction
x_0 = 0
sigma_0 = 0.1
psi_0 = gaussian_packet(x, x_0, sigma_0)
psi_0 = normalize_wavefunction(psi_0, dx)

# Compute the wave number array
psi_k = cp.fft.fft(psi_0)
k = cp.fft.fftfreq(N, d=dx)

# Precompute the k-space propagator
propagator = cp.exp(-1j * (h / 2) * k ** 2)

# Create the animation
anim = plot_wavefunction_evolution(
    x=x,
    psi_0=psi_0,
    propagator=propagator,
    dx=dx,
    num_steps=num_steps,
    interval=20,  # Frame interval in milliseconds
    save_file="wavefunction_evolution.mp4"  # Optional: save the animation
)

# Display the animation
plt.show()
