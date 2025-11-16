import matplotlib.pyplot as plt

from resources.Classes.Wave_function_class import *
from resources.Functions.system_fucntions import *
from matplotlib.colors import LogNorm
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp


from resources.Classes.Baryonic_N_body import NBodyBaryons

sim = Simulation_Class(
    dim=3,                             # 2D simulation
    boundaries=[(-50, 50),(-50, 50),(-50, 50)], # Spatial boundaries
    N=64,                             # Grid resolution
    total_time=15,                   # Total simulation time
    h=0.01,                            # Time step
    order_of_evolution=2,
    use_gravity=True,  # Enable gravitational effects
    static_potential=None,
    save_max_vals=True,

)

baryons = NBodyBaryons(
    simulation=sim,
    N_particles=5000,
    total_mass=1e6,
    init_profile="hernquist",
    radius=10

)
wave_vector = Wave_vector_class(
    packet_type="resources/solitons/GroundState(1).dat",
    means=[0, 0, 0],
    st_deviations=[0.5, 0.5, 0.5],
    simulation=sim,
    mass=1,
    omega=1,
    momenta=[0, 0.0, 0],
    spin=0,
    desired_soliton_mass=53090068

)


sim.add_baryons(baryons)
sim.add_wave_vector(wave_vector)


rho = cp.asnumpy(sim.baryonic_matter.deposit_to_grid())
x = sim.grids[0][:, 0, 0]
y = sim.grids[1][0, :, 0]
z = sim.grids[2][0, 0, :]

# ---- Quick diagnostics ----
total_mass = np.sum(rho) * np.prod(sim.dx)
print(f"Integrated baryonic mass: {total_mass:.3e}")
print(f"Deviation: {(total_mass / 1e8 - 1) * 100:.3f}%")

# ---- 3D scatter plot (coarse sample) ----
skip = 4  # increase for lighter plot
X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
RHO = rho[::skip, ::skip, ::skip]

# Mask near-zero densities for better color scaling
mask = RHO > 0
Cf = np.log10(RHO[mask])

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X[mask], Y[mask], Z[mask], c=Cf, cmap='plasma', s=3, alpha=0.8)
fig.colorbar(p, ax=ax, label=r'$\log_{10}(\rho)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Hernquist baryon density')
plt.tight_layout()
plt.show()

# ---- 2D central slice ----
mid = len(z) // 2
X2D, Y2D = np.meshgrid(x, y, indexing='ij')
rho_slice = rho[:, :, mid]
log_rho = np.log10(rho_slice + 1e-30)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(log_rho.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
               cmap='plasma', aspect='equal')
fig.colorbar(im, ax=ax, label=r'$\log_{10}(\rho)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Hernquist baryon density slice (z=0)')
plt.tight_layout()
plt.show()

sim.evolve(save_every=100)

rho = cp.asnumpy(sim.baryonic_matter.deposit_to_grid())
x = sim.grids[0][:, 0, 0]
y = sim.grids[1][0, :, 0]
z = sim.grids[2][0, 0, :]

# ---- Quick diagnostics ----
total_mass = np.sum(rho) * np.prod(sim.dx)
print(f"Integrated baryonic mass: {total_mass:.3e}")
print(f"Deviation: {(total_mass / 1e8 - 1) * 100:.3f}%")

# ---- 3D scatter plot (coarse sample) ----
skip = 4  # increase for lighter plot
X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
RHO = rho[::skip, ::skip, ::skip]

# Mask near-zero densities for better color scaling
mask = RHO > 0
Cf = np.log10(RHO[mask])

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X[mask], Y[mask], Z[mask], c=Cf, cmap='plasma', s=3, alpha=0.8)
fig.colorbar(p, ax=ax, label=r'$\log_{10}(\rho)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Hernquist baryon density')
plt.tight_layout()
plt.show()

# ---- 2D central slice ----
mid = len(z) // 2
X2D, Y2D = np.meshgrid(x, y, indexing='ij')
rho_slice = rho[:, :, mid]
log_rho = np.log10(rho_slice + 1e-30)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(log_rho.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
               cmap='plasma', aspect='equal')
fig.colorbar(im, ax=ax, label=r'$\log_{10}(\rho)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Hernquist baryon density slice (z=0)')
plt.tight_layout()
plt.show()

''' works with baryonic liquid
rho = cp.asnumpy(sim.baryonic_matter.rho_b)
x = sim.grids[0][:,0,0]
y = sim.grids[1][0,:,0]
z = sim.grids[2][0,0,:]

# build a coarser grid to avoid too many points
skip = 4  # increase for lighter plot
X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
RHO = rho[::skip, ::skip, ::skip]

# flatten
Xf, Yf, Zf, Cf = X.ravel(), Y.ravel(), Z.ravel(), np.log10(RHO.ravel() + 1e-30)

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(Xf, Yf, Zf, c=Cf, cmap='plasma', s=3, alpha=0.8)
fig.colorbar(p, ax=ax, label='log10(ρ)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Hernquist density distribution')
plt.tight_layout()
plt.show()

x = sim.grids[0][:,0,0]
y = sim.grids[1][0,:,0]
z = sim.grids[2][0,0,:]

# take a central slice (z = midplane)
mid = len(z)//2
X, Y = np.meshgrid(x, y, indexing='ij')
rho_slice = rho[:, :, mid]

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.log10(rho_slice + 1e-30), cmap='plasma', rstride=2, cstride=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('log10(ρ)')
ax.set_title('Hernquist density slice (z=0)')
plt.tight_layout()
plt.show()

before = cp.sum(sim.baryonic_matter.rho_b)
#sim.evolve(save_every=10)
after = cp.sum(sim.baryonic_matter.rho_b)
print(before - after)


rho = cp.asnumpy(sim.baryonic_matter.rho_b)
x = sim.grids[0][:,0,0]
y = sim.grids[1][0,:,0]
z = sim.grids[2][0,0,:]

# build a coarser grid to avoid too many points
skip = 4  # increase for lighter plot
X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
RHO = rho[::skip, ::skip, ::skip]

# flatten
Xf, Yf, Zf, Cf = X.ravel(), Y.ravel(), Z.ravel(), np.log10(RHO.ravel() + 1e-30)

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(Xf, Yf, Zf, c=Cf, cmap='plasma', s=3, alpha=0.8)
fig.colorbar(p, ax=ax, label='log10(ρ)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Hernquist density distribution')
plt.tight_layout()
plt.show()

x = sim.grids[0][:,0,0]
y = sim.grids[1][0,:,0]
z = sim.grids[2][0,0,:]

# take a central slice (z = midplane)
mid = len(z)//2
X, Y = np.meshgrid(x, y, indexing='ij')
rho_slice = rho[:, :, mid]

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.log10(rho_slice + 1e-30), cmap='plasma', rstride=2, cstride=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('log10(ρ)')
ax.set_title('Hernquist density slice (z=0)')
plt.tight_layout()
plt.show()

'''