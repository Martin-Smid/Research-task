import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def normalize_wavefunction(psi, dx):
    # Normalize the wavefunction
    psi /= cp.sqrt(cp.sum(cp.abs(psi) ** 2) * dx)
    return psi


def gaussian_packet(x, x_0, sigma_0):
    return cp.exp(-(x - x_0) ** 2 / (2 * sigma_0 ** 2)) / (cp.sqrt(2 * cp.pi) * sigma_0)


def evolve_wavefunction(psi, propagator_x, propagator_y, dx):
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


def plot_wavefunction_evolution(x, y, psi_0, propagator_x, propagator_y, dx, num_steps, interval=20, save_file=None):
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
    fig, axs = plt.subplots(3, figsize=(8, 8))
    fig.suptitle("Wavefunction Evolution", fontsize=14)

    # Configure subplots
    titles = ['|ψ|²']
    axs[0].set_title('Wavefunction Magnitude Squared')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    im = axs[0].imshow(
        cp.asnumpy(cp.zeros((N, N))),  # Convert to NumPy array
        extent=[x_numpy[0], x_numpy[-1], x_numpy[0], x_numpy[-1]],
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=0.01
    )
    fig.colorbar(im, ax=axs[0])
    lines = [im]

    def init():
        """Initialize the plot elements."""
        lines[0].set_data(cp.asnumpy(cp.zeros((N, N))))  # Convert to NumPy array
        return lines

    def update(step):
        """Update the plot for each frame."""
        nonlocal psi
        psi = evolve_wavefunction(psi, propagator_x, propagator_y, dx)
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
sigma_0 = 0.05
psi_0_x = gaussian_packet(x, x_0, sigma_0)
psi_0_x = normalize_wavefunction(psi_0_x, dx)
y = cp.linspace(a, b, N)  # Spatial grid for y-axis
psi_0_y = gaussian_packet(y, x_0, sigma_0)
psi_0_y = normalize_wavefunction(psi_0_y, dx)
psi_0 = psi_0_x[:, cp.newaxis] * psi_0_y[cp.newaxis, :]

# Compute the wave number array
psi_k = cp.fft.fft(psi_0)
k = cp.fft.fftfreq(N, d=dx)

# Precompute the k-space propagators
propagator_x = cp.exp(-1j * (h / 2) * k ** 2)
propagator_y = cp.exp(-1j * (h / 2) * k[:, cp.newaxis] ** 2)

# Create the animation
anim = plot_wavefunction_evolution(
    x=x,
    y=y,
    psi_0=psi_0,
    propagator_x=propagator_x,
    propagator_y=propagator_y,
    dx=dx,
    num_steps=num_steps,
    interval=20,  # Frame interval in milliseconds
    save_file="2D_wavefunction_evolution.mp4"  # Optional: save the animation
)

# Display the animation
plt.show()
