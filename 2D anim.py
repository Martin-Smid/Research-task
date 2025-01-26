import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Schrodinger_eq_functions import *




# Setup parameters
a, b = -1, 1  # Domain boundaries
N = 1024  # Number of spatial points
dx = (b - a) / (N - 1)
x = cp.linspace(a, b, N)  # Spatial grid for x-axis
y = cp.linspace(a, b, N)  # Spatial grid for y-axis
total_time = 10
h = 0.1  # Propagation parameter (time step size)
num_steps = int(total_time / h)

# Initialize wavefunction
#initial conditions of x-gaussian packet
init_mean_x = -0.2
init_std_x = 0.05
#initial conditions of y-gaussian packet
init_mean_y = 0.2
init_std_y = 0.05

psi_0_x = gaussian_packet(x, init_mean_x, init_std_x)
psi_0_x = normalize_wavefunction(psi_0_x, dx)


psi_0_y = gaussian_packet(y, init_mean_x, init_std_x)
psi_0_y = normalize_wavefunction(psi_0_y, dx)


psi_0 = psi_0_x[:, cp.newaxis] * psi_0_y[cp.newaxis, :]#cp.newaxis adds new dimension to array
#[:,cp.newaxis] creates 2D array composed of everything in one dimension : - slicing which takes everything and add new dim with cp.newaxis



# Compute the wave number array
psi_k = cp.fft.fft(psi_0)

# Define the 2D wave numbers
two_dim_k_x = cp.fft.fftfreq(N, d=dx)[:, cp.newaxis]  # Shape (N, 1)
two_dim_k_y = cp.fft.fftfreq(N, d=dx)[cp.newaxis, :]  # Shape (1, N)

# Compute the propagators
propagator_x = cp.exp(-1j * (h / 2) * two_dim_k_x ** 2)  # Shape (N, 1) - affects rows
propagator_y = cp.exp(-1j * (h / 2) * two_dim_k_y ** 2)  # Shape (1, N) - affects columns


# Create the animation
anim = plot_2D_wavefunction_evolution(
    x=x,
    y=y,
    psi_0=psi_0,
    propagator_x=propagator_x,
    propagator_y=propagator_y,
    dx=dx,
    num_steps=num_steps,
    interval=20,  # Frame interval in milliseconds
    save_file="2D_wavefunction_evolution1.mp4",  # Optional: save the animation
    N=N
)

# Display the animation
plt.show()
