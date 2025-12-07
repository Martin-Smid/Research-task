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

    '''def deposit_to_grid(self):
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

        return rho_grid'''

    def interpolate_force_from_grid(self, potential_grid):
        """Calculate forces using CIC interpolation (matches deposition)."""
        sim = self.simulation
        Nx, Ny, Nz = potential_grid.shape

        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dz = (z_max - z_min) / Nz

        # Compute force grids via FFT
        kx = 2 * cp.pi * cp.fft.fftfreq(Nx, d=dx)
        ky = 2 * cp.pi * cp.fft.fftfreq(Ny, d=dy)
        kz = 2 * cp.pi * cp.fft.fftfreq(Nz, d=dz)

        Phi_k = cp.fft.fftn(potential_grid)

        fx_k = -1j * kx[:, None, None] * Phi_k
        fy_k = -1j * ky[None, :, None] * Phi_k
        fz_k = -1j * kz[None, None, :] * Phi_k

        Fx_grid = cp.real(cp.fft.ifftn(fx_k))
        Fy_grid = cp.real(cp.fft.ifftn(fy_k))
        Fz_grid = cp.real(cp.fft.ifftn(fz_k))

        # Use CIC interpolation instead of map_coordinates
        return self.interpolate_force_from_grid_CIC(Fx_grid, Fy_grid, Fz_grid)

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



    def deposit_to_grid(self):
        """Deposit with atomic operations via cupyx.scatter_add"""
        import cupyx
        sim = self.simulation
        shape = (sim.N,) * sim.dim
        rho_grid = cp.zeros(shape, dtype=cp.float64)

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
        rho_flat = rho_grid.ravel()

        for w, ii, jj, kk in corners_weights:
            flat_indices = ii * N_sq + jj * N + kk
            contribution = mass_val * w

            # Thread-safe scatter add
            cupyx.scatter_add(rho_flat, flat_indices, contribution)

        if cp.any(cp.isnan(rho_grid)) or cp.any(rho_grid < 0):
            print(f"⚠️  DENSITY CORRUPTION DETECTED!")
            print(f"   NaN count: {cp.sum(cp.isnan(rho_grid))}")
            print(f"   Negative count: {cp.sum(rho_grid < 0)}")
            print(f"   Max density: {cp.max(rho_grid):.3e}")

        total_deposited = cp.sum(rho_grid) * cell_volume
        expected_mass = self.N * self.m_particle
        error = abs(total_deposited - expected_mass) / expected_mass

        if error > 1e-6:
            print(f"⚠️  MASS CONSERVATION ERROR: {error:.3e}")

        return rho_grid

    def interpolate_force_from_grid_CIC(self, Fx_grid, Fy_grid, Fz_grid):
        """
        Interpolate forces using CIC (matching the deposition scheme).
        """
        sim = self.simulation
        dx, dy, dz = sim.dx
        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        Lx, Ly, Lz = x_max - x_min, y_max - y_min, z_max - z_min

        # Periodic wrap
        px = x_min + cp.mod(self.positions[:, 0] - x_min, Lx)
        py = y_min + cp.mod(self.positions[:, 1] - y_min, Ly)
        pz = z_min + cp.mod(self.positions[:, 2] - z_min, Lz)

        # Fractional indices
        fx = (px - x_min) / dx
        fy = (py - y_min) / dy
        fz = (pz - z_min) / dz

        i0 = cp.floor(fx).astype(cp.int32)
        j0 = cp.floor(fy).astype(cp.int32)
        k0 = cp.floor(fz).astype(cp.int32)

        tx, ty, tz = fx - i0, fy - j0, fz - k0
        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        N = sim.N
        i1, j1, k1 = (i0 + 1) % N, (j0 + 1) % N, (k0 + 1) % N
        i0, j0, k0 = i0 % N, j0 % N, k0 % N

        # Interpolate each force component using same CIC weights
        forces = cp.zeros((self.N, 3), dtype=cp.float64)

        corners = [
            (wx0 * wy0 * wz0, i0, j0, k0), (wx1 * wy0 * wz0, i1, j0, k0),
            (wx0 * wy1 * wz0, i0, j1, k0), (wx1 * wy1 * wz0, i1, j1, k0),
            (wx0 * wy0 * wz1, i0, j0, k1), (wx1 * wy0 * wz1, i1, j0, k1),
            (wx0 * wy1 * wz1, i0, j1, k1), (wx1 * wy1 * wz1, i1, j1, k1)
        ]

        for w, ii, jj, kk in corners:
            forces[:, 0] += w * Fx_grid[ii, jj, kk]
            forces[:, 1] += w * Fy_grid[ii, jj, kk]
            forces[:, 2] += w * Fz_grid[ii, jj, kk]

        return forces