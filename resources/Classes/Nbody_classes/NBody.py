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



    def interpolate_force_from_grid(self, potential_grid):
        """
        Compute forces from a potential grid via FFT, then interpolate them
        to particle positions using trilinear interpolation (CIC style),
        in the same coordinate convention as deposit_to_grid.
        """
        sim = self.simulation
        Nx, Ny, Nz = potential_grid.shape

        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dz = (z_max - z_min) / Nz

        # k-vectors (FFT frequencies) – double precision
        kx = 2 * cp.pi * cp.fft.fftfreq(Nx, d=dx)
        ky = 2 * cp.pi * cp.fft.fftfreq(Ny, d=dy)
        kz = 2 * cp.pi * cp.fft.fftfreq(Nz, d=dz)

        # FFT of potential (cast to complex128 for better phase accuracy)
        Phi_k = cp.fft.fftn(potential_grid.astype(cp.complex128))

        fx_k = -1j * kx[:, None, None] * Phi_k
        fy_k = -1j * ky[None, :, None] * Phi_k
        fz_k = -1j * kz[None, None, :] * Phi_k

        Fx_grid = cp.real(cp.fft.ifftn(fx_k)).astype(cp.float64)
        Fy_grid = cp.real(cp.fft.ifftn(fy_k)).astype(cp.float64)
        Fz_grid = cp.real(cp.fft.ifftn(fz_k)).astype(cp.float64)

        # Trilinear interpolate the three components
        return self._interpolate_force_trilinear(Fx_grid, Fy_grid, Fz_grid)

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

    def drift(self, dt, potential_grid, first_step=False, last_step=False):
        """
        Update baryonic particle positions/velocities with substeps.
        Uses DKD (Drift-Kick-Drift) scheme with single force evaluation per substep.
        """
        # --- 1. total dt window ---
        if first_step or last_step:
            total_dt_window = dt / 2.0
        else:
            total_dt_window = dt

        if isinstance(potential_grid, (tuple, list)) and len(potential_grid) == 3:
            Fx_grid, Fy_grid, Fz_grid = potential_grid

            def force_computer():
                return self._interpolate_force_trilinear(Fx_grid, Fy_grid, Fz_grid)
        else:
            def force_computer():
                return self.interpolate_force_from_grid(potential_grid)

        # --- 3. velocity-based criterion ---
        v_sq = cp.sum(self.velocities ** 2, axis=1)
        v_max = float(cp.sqrt(cp.max(v_sq)))

        if v_max < 1e-10:
            v_max = 1e-10

        min_dx = min(self.simulation.dx)
        f_v = 0.25
        dt_vel = f_v * (min_dx / v_max)

        # --- 4. acceleration-based criterion ---
        forces = force_computer()
        a_sq = cp.sum(forces ** 2, axis=1)
        a_max = float(cp.sqrt(cp.max(a_sq)))

        if a_max < 1e-10:
            a_max = 1e-10

        f_a = 0.20
        dt_acc = float(f_a * cp.sqrt(min_dx / a_max))

        # --- 5. determine substeps ---
        n_vel = int(np.ceil(total_dt_window / dt_vel))
        n_acc = int(np.ceil(total_dt_window / dt_acc))
        num_substeps = max(n_vel, n_acc)
        num_substeps = max(1, min(num_substeps, 50))
        dt_sub = total_dt_window / num_substeps

        # --- 6. DKD substep loop: ONLY ONE FORCE EVALUATION PER SUBSTEP ---
        for step_i in range(num_substeps):

            # SPECIAL CASE: First substep needs initial half-drift
            if step_i == 0:
                # Initial half-drift
                self.positions += self.velocities * (dt_sub / 2.0)

                # Periodic wrap
                for dim in range(3):
                    low, high = self.simulation.boundaries[dim]
                    width = high - low
                    self.positions[:, dim] = ((self.positions[:, dim] - low) % width) + low

            # C. Full kick with forces at current position
            forces = force_computer()
            self.velocities += forces * dt_sub

            # D. Full drift (except last substep does half-drift)
            drift_time = dt_sub if step_i < num_substeps - 1 else (dt_sub / 2.0)
            self.positions += self.velocities * drift_time

            # Periodic wrap
            for dim in range(3):
                low, high = self.simulation.boundaries[dim]
                width = high - low
                self.positions[:, dim] = ((self.positions[:, dim] - low) % width) + low

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
        N = sim.N
        dim = sim.dim

        # 1. Allocate density grid
        shape = (N,) * dim
        rho_grid = cp.zeros(shape, dtype=cp.float64)

        # 2. Boundaries and box sizes
        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

        # 3. Periodic-wrap particle positions into [x_min, x_max), etc.
        px = x_min + cp.mod(self.positions[:, 0] - x_min, Lx)
        py = y_min + cp.mod(self.positions[:, 1] - y_min, Ly)
        pz = z_min + cp.mod(self.positions[:, 2] - z_min, Lz)


        #    rx = (x / BoxSize) * N  → here BoxSize = Lx, but shifted by x_min
        rx = (px - x_min) / Lx * N
        ry = (py - y_min) / Ly * N
        rz = (pz - z_min) / Lz * N

        i0 = cp.floor(rx).astype(cp.int32)
        j0 = cp.floor(ry).astype(cp.int32)
        k0 = cp.floor(rz).astype(cp.int32)

        # fractional offsets inside the cell
        tx = rx - i0
        ty = ry - j0
        tz = rz - k0

        # CIC weights (same as your original code, just expressed like the C++)
        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        # periodic neighbour indices
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N

        i0 = i0 % N
        j0 = j0 % N
        k0 = k0 % N

        # 5. Mass per volume
        cell_volume = sim.dV
        mass_val = self.m_particle / cell_volume

        # 6. 8 corners and their weights
        corners_weights = [
            (wx0 * wy0 * wz0, i0, j0, k0),
            (wx1 * wy0 * wz0, i1, j0, k0),
            (wx0 * wy1 * wz0, i0, j1, k0),
            (wx1 * wy1 * wz0, i1, j1, k0),
            (wx0 * wy0 * wz1, i0, j0, k1),
            (wx1 * wy0 * wz1, i1, j0, k1),
            (wx0 * wy1 * wz1, i0, j1, k1),
            (wx1 * wy1 * wz1, i1, j1, k1),
        ]

        N_sq = N * N
        rho_flat = rho_grid.ravel()

        # 7. Thread-safe scatter-add onto flattened grid
        for w, ii, jj, kk in corners_weights:
            flat_indices = ii * N_sq + jj * N + kk
            contribution = mass_val * w
            cupyx.scatter_add(rho_flat, flat_indices, contribution)

        # optional sanity checks (you can keep these)
        total_deposited = cp.sum(rho_grid) * cell_volume
        expected_mass = self.N * self.m_particle
        error = abs(total_deposited - expected_mass) / expected_mass

        if error > 1e-6:
            print(f"⚠️  MASS CONSERVATION ERROR: {error:.3e}")

        return rho_grid

    def _interpolate_force_trilinear(self, Fx_grid, Fy_grid, Fz_grid):
        """
        Interpolate forces using CIC (matching the deposition scheme).
        """
        sim = self.simulation
        N = sim.N

        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

        # Periodic wrap of particle positions
        px = x_min + cp.mod(self.positions[:, 0] - x_min, Lx)
        py = y_min + cp.mod(self.positions[:, 1] - y_min, Ly)
        pz = z_min + cp.mod(self.positions[:, 2] - z_min, Lz)

        # Normalize to grid units (0..N) as for deposition
        rx = (px - x_min) / Lx * N
        ry = (py - y_min) / Ly * N
        rz = (pz - z_min) / Lz * N

        i0 = cp.floor(rx).astype(cp.int32)
        j0 = cp.floor(ry).astype(cp.int32)
        k0 = cp.floor(rz).astype(cp.int32)

        tx = rx - i0
        ty = ry - j0
        tz = rz - k0

        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N

        i0 = i0 % N
        j0 = j0 % N
        k0 = k0 % N

        # Helper to gather values at corners
        def gather(grid, ii, jj, kk):
            return grid[ii, jj, kk]

        # 8 corners for each component
        # Fx
        Fx000 = gather(Fx_grid, i0, j0, k0)
        Fx100 = gather(Fx_grid, i1, j0, k0)
        Fx010 = gather(Fx_grid, i0, j1, k0)
        Fx110 = gather(Fx_grid, i1, j1, k0)
        Fx001 = gather(Fx_grid, i0, j0, k1)
        Fx101 = gather(Fx_grid, i1, j0, k1)
        Fx011 = gather(Fx_grid, i0, j1, k1)
        Fx111 = gather(Fx_grid, i1, j1, k1)

        # Fy
        Fy000 = gather(Fy_grid, i0, j0, k0)
        Fy100 = gather(Fy_grid, i1, j0, k0)
        Fy010 = gather(Fy_grid, i0, j1, k0)
        Fy110 = gather(Fy_grid, i1, j1, k0)
        Fy001 = gather(Fy_grid, i0, j0, k1)
        Fy101 = gather(Fy_grid, i1, j0, k1)
        Fy011 = gather(Fy_grid, i0, j1, k1)
        Fy111 = gather(Fy_grid, i1, j1, k1)

        # Fz
        Fz000 = gather(Fz_grid, i0, j0, k0)
        Fz100 = gather(Fz_grid, i1, j0, k0)
        Fz010 = gather(Fz_grid, i0, j1, k0)
        Fz110 = gather(Fz_grid, i1, j1, k0)
        Fz001 = gather(Fz_grid, i0, j0, k1)
        Fz101 = gather(Fz_grid, i1, j0, k1)
        Fz011 = gather(Fz_grid, i0, j1, k1)
        Fz111 = gather(Fz_grid, i1, j1, k1)

        # Now trilinear interpolation for each component:
        # combine x, then y, then z as in the C++ code
        def trilinear(c000, c100, c010, c110, c001, c101, c011, c111):
            c00 = c000 * (1.0 - tx) + c100 * tx
            c01 = c001 * (1.0 - tx) + c101 * tx
            c10 = c010 * (1.0 - tx) + c110 * tx
            c11 = c011 * (1.0 - tx) + c111 * tx

            c0 = c00 * (1.0 - ty) + c10 * ty
            c1 = c01 * (1.0 - ty) + c11 * ty

            return c0 * (1.0 - tz) + c1 * tz

        Fx_part = trilinear(Fx000, Fx100, Fx010, Fx110, Fx001, Fx101, Fx011, Fx111)
        Fy_part = trilinear(Fy000, Fy100, Fy010, Fy110, Fy001, Fy101, Fy011, Fy111)
        Fz_part = trilinear(Fz000, Fz100, Fz010, Fz110, Fz001, Fz101, Fz011, Fz111)

        forces = cp.zeros((self.N, 3), dtype=cp.float64)
        forces[:, 0] = Fx_part
        forces[:, 1] = Fy_part
        forces[:, 2] = Fz_part

        return forces

    def kinetic_energy(self):
        """
        Default: constant per-particle mass (Baryons-style).
        Override in subclasses with variable particle masses.
        """
        v2 = (self.velocities ** 2).sum(axis=1)  # |v|^2 per particle
        return 0.5 * self.m_particle * v2.sum()
