import cupy as cp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cupyx.scipy.ndimage as ndimage


class NBody:
    """
    Generic N-body base class: stores particle data and provides
    deposition, interpolation and leapfrog integration on the simulation grid.
    """

    def __init__(self, simulation, N_particles, total_mass):
        """
        Parameters
        ----------
        simulation : Simulation object
            The main simulation object containing grid info and boundaries.
        N_particles : int
            Number of particles to simulate.
        total_mass : float
            Total mass of all particles.
        """
        self.simulation = simulation
        self.N = N_particles
        self.m_particle = total_mass / N_particles

        # Particle data (N_particles, 3) - stored on GPU with CuPy
        self.positions = cp.zeros((N_particles, 3), dtype=cp.float64)
        self.velocities = cp.zeros((N_particles, 3), dtype=cp.float64)

    # ----------------------------
    # Generic grid-coupling methods
    # ----------------------------

    def deposit_to_grid(self):
        """
        Deposit particle masses to grid using Cloud-In-Cell (CIC) (GPU: CuPy).
        Uses sim.dx directly to ensure consistency with the fixed boundary logic.
        """
        sim = self.simulation

        # 1. Initialize density grid on GPU
        shape = (sim.N,) * sim.dim
        rho_grid = cp.zeros(shape, dtype=cp.float64)

        # 2. Get Grid Spacing & Boundaries directly from Simulation
        dx, dy, dz = sim.dx

        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

        # 3. Periodic Wrap of Positions (GPU)
        px = x_min + cp.mod(self.positions[:, 0] - x_min, Lx)
        py = y_min + cp.mod(self.positions[:, 1] - y_min, Ly)
        pz = z_min + cp.mod(self.positions[:, 2] - z_min, Lz)

        # 4. Convert to Fractional Grid Indices (GPU)
        fx = (px - x_min) / dx
        fy = (py - y_min) / dy
        fz = (pz - z_min) / dz

        # 5. Cloud-In-Cell (CIC) Interpolation Weights (GPU)
        i0 = cp.floor(fx).astype(cp.int32)
        j0 = cp.floor(fy).astype(cp.int32)
        k0 = cp.floor(fz).astype(cp.int32)

        tx = fx - i0
        ty = fy - j0
        tz = fz - k0

        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        # 6. Handle Periodic Wrapping for Indices (GPU)
        N = sim.N
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N
        i0 = i0 % N
        j0 = j0 % N
        k0 = k0 % N

        # 7. Mass Deposit
        cell_volume = dx * dy * dz
        mass_val = self.m_particle / cell_volume

        corners_weights = [
            (wx0 * wy0 * wz0, i0, j0, k0), (wx1 * wy0 * wz0, i1, j0, k0),
            (wx0 * wy1 * wz0, i0, j1, k0), (wx1 * wy1 * wz0, i1, j1, k0),
            (wx0 * wy0 * wz1, i0, j0, k1), (wx1 * wy0 * wz1, i1, j0, k1),
            (wx0 * wy1 * wz1, i0, j1, k1), (wx1 * wy1 * wz1, i1, j1, k1)
        ]

        N_sq = N * N

        for w, ii, jj, kk in corners_weights:
            flat_indices = ii * N_sq + jj * N + kk
            contribution = mass_val * w
            cp.add.at(rho_grid.ravel(), flat_indices, contribution)

        return rho_grid

    def interpolate_force_from_grid(self, potential_grid):
        """
        Calculate forces entirely on the GPU to avoid Host-Device transfers.
        Correctly calculates dx/dy/dz from boundaries and grid shape to ensure
        consistency with the FFT grid.
        """
        sim = self.simulation

        Nx, Ny, Nz = potential_grid.shape

        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dz = (z_max - z_min) / Nz

        # F = -∇Φ
        kx = 2 * cp.pi * cp.fft.fftfreq(Nx, d=dx)
        ky = 2 * cp.pi * cp.fft.fftfreq(Ny, d=dy)
        kz = 2 * cp.pi * cp.fft.fftfreq(Nz, d=dz)

        Phi_k = cp.fft.fftn(potential_grid)

        # force components in k-space: F_k = -ik * Phi_k
        fx_k = -1j * kx[:, None, None] * Phi_k
        fy_k = -1j * ky[None, :, None] * Phi_k
        fz_k = -1j * kz[None, None, :] * Phi_k

        # back to real space (Forces on grid)
        Fx_grid = cp.real(cp.fft.ifftn(fx_k))
        Fy_grid = cp.real(cp.fft.ifftn(fy_k))
        Fz_grid = cp.real(cp.fft.ifftn(fz_k))

        coords = cp.empty((3, self.N), dtype=cp.float64)
        coords[0] = (self.positions[:, 0] - x_min) / dx
        coords[1] = (self.positions[:, 1] - y_min) / dy
        coords[2] = (self.positions[:, 2] - z_min) / dz

        Fx = ndimage.map_coordinates(Fx_grid, coords, order=1, mode='wrap')
        Fy = ndimage.map_coordinates(Fy_grid, coords, order=1, mode='wrap')
        Fz = ndimage.map_coordinates(Fz_grid, coords, order=1, mode='wrap')

        forces = cp.stack((Fx, Fy, Fz), axis=1)
        return forces

    def integrate_leapfrog(self, force_computer, dt):
        """
        Leapfrog integration with proper force evaluation.

        Parameters:
            force_computer: callable that returns forces (N_particles, 3)
            dt: time step
        """
        # Half kick with current forces
        forces = force_computer()
        self.velocities += forces * (dt / 2)

        # Full drift
        self.positions += self.velocities * dt

        # Apply periodic boundary conditions
        for dim in range(3):
            low, high = self.simulation.boundaries[dim]
            width = high - low
            self.positions[:, dim] = ((self.positions[:, dim] - low) % width) + low

        # Half kick with NEW forces after drift
        forces = force_computer()
        self.velocities += forces * (dt / 2)

    def _circular_speed_from_grid(self, potential_grid, center_xyz):
        """Return v_circ at center_xyz from grid potential. (CPU/NumPy/SciPy due to complexity)"""
        sim = self.simulation

        # --- Transfer to CPU for NumPy/SciPy operations ---
        x = cp.asnumpy(sim.grids[0][:, 0, 0])
        y = cp.asnumpy(sim.grids[1][0, :, 0])
        z = cp.asnumpy(sim.grids[2][0, 0, :])

        Phi = cp.asnumpy(potential_grid)

        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        dz = float(z[1] - z[0])

        # NumPy FFT operations
        kx = 2 * np.pi * np.fft.fftfreq(Phi.shape[0], d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(Phi.shape[1], d=dy)
        kz = 2 * np.pi * np.fft.fftfreq(Phi.shape[2], d=dz)
        Phi_k = np.fft.fftn(Phi)
        dphidx = np.fft.ifftn(1j * kx[:, None, None] * Phi_k).real
        dphidy = np.fft.ifftn(1j * ky[None, :, None] * Phi_k).real
        dphidz = np.fft.ifftn(1j * kz[None, None, :] * Phi_k).real

        cx, cy, cz = center_xyz
        ix = np.argmin(np.abs(x - cx))
        iy = np.argmin(np.abs(y - cy))
        iz = np.argmin(np.abs(z - cz))

        xc = 0.5 * (sim.boundaries[0][0] + sim.boundaries[0][1])
        yc = 0.5 * (sim.boundaries[1][0] + sim.boundaries[1][1])
        zc = 0.5 * (sim.boundaries[2][0] + sim.boundaries[2][1])
        rx, ry, rz = cx - xc, cy - yc, cz - zc
        r = np.sqrt(rx * rx + ry * ry + rz * rz)
        if r == 0.0:
            return 0.0

        ux, uy, uz = rx / r, ry / r, rz / r
        dphidr = dphidx[ix, iy, iz] * ux + dphidy[ix, iy, iz] * uy + dphidz[ix, iy, iz] * uz
        return np.sqrt(abs(r * dphidr))