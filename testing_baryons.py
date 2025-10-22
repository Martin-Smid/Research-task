from resources.Classes.Baryonic_Matter_Class import BaryonicMatter_Class
from resources.Classes.Simulation_Class import Simulation_Class
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

sim = Simulation_Class(
    dim=3,                             # 2D simulation
    boundaries=[(-10, 10),(-10, 10),(-10, 10)], # Spatial boundaries
    N=64,                             # Grid resolution
    total_time=0.2,                   # Total simulation time
    h=0.001,                            # Time step
    order_of_evolution=2,
    baryonic_model="two_clumps",
    use_gravity=True,
)


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
sim.evolve(save_every=10)
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