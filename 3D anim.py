from resources.Classes.Wave_function_class import *



boundaries3D = [(-1,1), (-1,1), (-1,1)]
N = 128

vlna = Wave_function(
    packet_type="gaussian",
    potential=None,  # No static potential
    gravity_potential=True,
    means=[0, 0, 0],
    st_deviations=[0.1, 0.1, 0.1],
    mass=2,
    omega=1,
    dim=3,
    boundaries=boundaries3D,
    N=N,
    total_time=1,
    h=0.1,
)

wave_begin = cp.asnumpy(cp.abs(vlna.wave_values[0]))  # Initial snapshot
wave_middle = cp.asnumpy(cp.abs(vlna.wave_values[len(vlna.wave_values) // 2]))  # Middle snapshot
wave_end = cp.asnumpy(cp.abs(vlna.wave_values[-1]))  # Final snapshot

# Take a central slice for 2D visualization (slice at z = middle of the grid)
z_idx = vlna.N // 2  # Index for z=0 plane
wave_begin_slice = wave_begin[:, :, z_idx]
wave_middle_slice = wave_middle[:, :, z_idx]
wave_end_slice = wave_end[:, :, z_idx]

# Grids for plotting (create a uniform grid based on boundaries)
x, y = np.linspace(vlna.boundaries[0][0], vlna.boundaries[0][1], vlna.N), \
    np.linspace(vlna.boundaries[1][0], vlna.boundaries[1][1], vlna.N)
X, Y = np.meshgrid(x, y)

# Plot the wavefunction slices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["Wavefunction at Beginning", "Wavefunction at Middle", "Wavefunction at End"]
waves = [wave_begin_slice, wave_middle_slice, wave_end_slice]

for ax, wave, title in zip(axes, waves, titles):
    im = ax.pcolormesh(X, Y, wave, shading='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, orientation="vertical", label="|ψ|")

plt.tight_layout()
plt.show()

