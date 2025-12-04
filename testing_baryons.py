from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp


from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons

sim = Simulation_Class(

    dim=3, # 2D simulation
    boundaries=[(-15, 15),(-15, 15),(-15, 15)], # Spatial boundaries
    N=128, # Grid resolution
    total_time=1, # Total simulation time
    h=0.0005, # Time step
    order_of_evolution=2,
    use_gravity=True, # Enable gravitational effects
    static_potential=None,
    save_max_vals=True,



)



bulge = Baryons(

    simulation=sim,
    N_particles=2_000_000,
    total_mass=1e10, # Msun
    init_profile="uniform",

    center=(0.0, 0.0, 0.0),


)

M=1e6

r = 5.0
d = 2.0 * r
v_orbit = np.sqrt(4.49 * M / (2.0 * d))
vel_sigma = 0.1 * v_orbit   # e.g. 10% of orbital speed

baryon1 = Baryons(
    simulation=sim,
    N_particles=20000,
    total_mass=M,
    init_profile="hernquist",
    scale_radius=0.5,
    truncation_radius=1.0,
    center=(0.0,  r, 0.0),
    velocity=( v_orbit, 0.0, 0.0),
    vel_sigma=vel_sigma,
)

baryon2 = Baryons(
    simulation=sim,
    N_particles=20000,
    total_mass=M,
    init_profile="hernquist",
    scale_radius=0.5,
    truncation_radius=1.0,
    center=(0.0, -r, 0.0),
    velocity=(-v_orbit, 0.0, 0.0),
    vel_sigma=vel_sigma,
)


disk = Baryons(

    simulation=sim,
    N_particles=5_00_00,
    total_mass=5e8, # Msun
    init_profile="disk",
    radius=1.50, # R_d in kpc
    height=0.3, # z_0 in kpc
    center=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0), # no bulk COM motion
    vel_sigma=20.0, # random dispersion (radial/vertical)
    circular_velocity = 220

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





#sim.add_baryons(baryon1)
sim.add_baryons(bulge)
#sim.add_wave_vector(wave_vector)

# BEFORE evolution - sum all baryonic systems
rho = cp.zeros((sim.N, sim.N, sim.N), dtype=cp.float64)
for baryons in sim.baryonic_matter:
    rho += baryons.deposit_to_grid()

rho = cp.asnumpy(rho)  # Move to CPU for plotting

x = sim.grids[0][:, 0, 0]
y = sim.grids[1][0, :, 0]
z = sim.grids[2][0, 0, :]

# ---- Quick diagnostics ----
total_mass = np.sum(rho) * np.prod(sim.dx)
print(f"Integrated baryonic mass (before): {total_mass:.3e}")
#print(f"Expected total mass: {sim.total_baryon_mass:.3e}")
#print(f"Deviation: {(total_mass / sim.total_baryon_mass - 1) * 100:.3f}%")

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
ax.set_title('3D Baryonic density (before evolution)')
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
ax.set_title('Baryonic density slice (z=0, before evolution)')
plt.tight_layout()
plt.show()


sim.evolve(save_every=10)

# AFTER evolution - sum all baryonic systems again
rho = cp.zeros((sim.N, sim.N, sim.N), dtype=cp.float64)
for baryons in sim.baryonic_matter:
    rho += baryons.deposit_to_grid()

rho = cp.asnumpy(rho)

x = sim.grids[0][:, 0, 0]
y = sim.grids[1][0, :, 0]
z = sim.grids[2][0, 0, :]

# ---- Quick diagnostics ----
total_mass = np.sum(rho) * np.prod(sim.dx)
print(f"Integrated baryonic mass (after): {total_mass:.3e}")
#print(f"Expected total mass: {sim.total_baryon_mass:.3e}")
#print(f"Deviation: {(total_mass / sim.total_baryon_mass - 1) * 100:.3f}%")

# ---- 3D scatter plot (coarse sample) ----
skip = 4
X, Y, Z = np.meshgrid(x[::skip], y[::skip], z[::skip], indexing='ij')
RHO = rho[::skip, ::skip, ::skip]

mask = RHO > 0
Cf = np.log10(RHO[mask])

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X[mask], Y[mask], Z[mask], c=Cf, cmap='plasma', s=3, alpha=0.8)
fig.colorbar(p, ax=ax, label=r'$\log_{10}(\rho)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Baryonic density (after evolution)')
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
ax.set_title('Baryonic density slice (z=0, after evolution)')
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