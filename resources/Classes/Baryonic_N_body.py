import cupy as cp
import numpy as np


class NBodyBaryons:
    def __init__(self, simulation, N_particles, total_mass,
                 init_profile="hernquist", scale_radius=0.5):
        self.simulation = simulation
        self.N = N_particles
        self.m_particle = total_mass / N_particles
        self.scale_radius = scale_radius

        # Particle data (N_particles, 3)
        self.positions = cp.zeros((N_particles, 3), dtype=cp.float64)
        #self.velocities = cp.zeros((N_particles, 3), dtype=cp.float64)
        sigma = 0.1
        self.velocities = cp.random.normal(0, sigma, size=(N_particles, 3))

        print(self.positions)

        # Initialize spatial distribution
        if init_profile == "hernquist":
            self.initialize_hernquist()
        elif init_profile == "uniform":
            self.initialize_uniform()

        print(self.positions)

    def initialize_hernquist(self):
        """Sample particle positions from Hernquist profile"""
        # Use inverse transform sampling
        # Hernquist cumulative mass: M(r) = M * r^2 / (r + a)^2

        u = cp.random.uniform(0, 1, self.N)  # random [0,1]
        # Inverse: r = a * sqrt(u) / (1 - sqrt(u))
        r = self.scale_radius * cp.sqrt(u) / (1 - cp.sqrt(u))

        # Random directions (spherical coordinates)
        theta = cp.arccos(2 * cp.random.uniform(0, 1, self.N) - 1)
        phi = 2 * cp.pi * cp.random.uniform(0, 1, self.N)

        # Convert to Cartesian
        self.positions[:, 0] = r * cp.sin(theta) * cp.cos(phi)
        self.positions[:, 1] = r * cp.sin(theta) * cp.sin(phi)
        self.positions[:, 2] = r * cp.cos(theta)

        # Velocities: start at rest or use virial theorem
        self.velocities[:, :] = 0  # cold start

    def initialize_uniform(self):
        """Uniform random distribution in simulation box"""
        for dim in range(3):
            low, high = self.simulation.boundaries[dim]
            self.positions[:, dim] = cp.random.uniform(low, high, self.N)

    def deposit_to_grid(self):
        """
        Deposit particle masses to grid using Cloud-In-Cell (CIC)
        Returns: density field on simulation grid
        """
        shape = (self.simulation.N,) * self.simulation.dim
        rho_grid = cp.zeros(shape, dtype=cp.float64)

        # Grid spacing
        dx = self.simulation.dx

        # Grid boundaries
        (x_min, x_max) = self.simulation.boundaries[0]
        (y_min, y_max) = self.simulation.boundaries[1]
        (z_min, z_max) = self.simulation.boundaries[2]

        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

        px = x_min + cp.mod(self.positions[:, 0] - x_min, Lx)
        py = y_min + cp.mod(self.positions[:, 1] - y_min, Ly)
        pz = z_min + cp.mod(self.positions[:, 2] - z_min, Lz)

        # Convert positions to grid indices (fractional)
        ix = (px - x_min) / dx[0]
        iy = (py - y_min) / dx[1]
        iz = (pz - z_min) / dx[2]

        # Cloud-In-Cell: distribute to 8 nearest grid points
        # Floor to get base cell
        i0 = cp.floor(ix).astype(cp.int32)
        j0 = cp.floor(iy).astype(cp.int32)
        k0 = cp.floor(iz).astype(cp.int32)

        # Fractional offset within cell [0, 1]
        tx = ix - i0
        ty = iy - j0
        tz = iz - k0

        # Weights for 8 corners
        wx0, wx1 = 1 - tx, tx
        wy0, wy1 = 1 - ty, ty
        wz0, wz1 = 1 - tz, tz

        # Periodic boundary conditions
        N = self.simulation.N
        i0 %= N
        j0 %= N
        k0 %= N
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N

        # Mass per volume
        mass = self.m_particle / (dx[0] * dx[1] * dx[2])

        # Pre-compute weights for all 8 corners
        weights = [
            (wx0, wy0, wz0), (wx1, wy0, wz0),
            (wx0, wy1, wz0), (wx1, wy1, wz0),
            (wx0, wy0, wz1), (wx1, wy0, wz1),
            (wx0, wy1, wz1), (wx1, wy1, wz1)
        ]

        corners = [
            (i0, j0, k0), (i1, j0, k0),
            (i0, j1, k0), (i1, j1, k0),
            (i0, j0, k1), (i1, j0, k1),
            (i0, j1, k1), (i1, j1, k1)
        ]

        # Deposit to each corner using vectorized add.at
        for (wx, wy, wz), (ii, jj, kk) in zip(weights, corners):
            contribution = mass * wx * wy * wz
            flat_indices = ii * (N * N) + jj * N + kk
            cp.add.at(rho_grid.ravel(), flat_indices, contribution)

        return rho_grid

    def interpolate_force_from_grid(self, potential_grid):
        """
        Interpolate force F = -∇Φ from grid to particle positions (CIC)
        Returns: forces array (N_particles, 3)
        """
        forces = cp.zeros((self.N, 3), dtype=cp.float32)

        # Compute gradients on grid first
        grad_phi = []
        for dim, k in enumerate(self.simulation.k_space):
            phi_k = cp.fft.fftn(potential_grid)
            grad = cp.real(cp.fft.ifftn(1j * k * phi_k))
            grad_phi.append(grad)

        # Grid spacing and boundaries
        dx = self.simulation.dx
        x_min = self.simulation.boundaries[0][0]
        y_min = self.simulation.boundaries[1][0]
        z_min = self.simulation.boundaries[2][0]

        # Convert positions to grid indices
        ix = (self.positions[:, 0] - x_min) / dx[0]
        iy = (self.positions[:, 1] - y_min) / dx[1]
        iz = (self.positions[:, 2] - z_min) / dx[2]

        i0 = cp.floor(ix).astype(cp.int32)
        j0 = cp.floor(iy).astype(cp.int32)
        k0 = cp.floor(iz).astype(cp.int32)

        tx = ix - i0
        ty = iy - j0
        tz = iz - k0

        wx0, wx1 = 1 - tx, tx
        wy0, wy1 = 1 - ty, ty
        wz0, wz1 = 1 - tz, tz

        N = self.simulation.N
        i0 = i0 % N
        j0 = j0 % N
        k0 = k0 % N
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N

        # Interpolate each component of gradient
        for dim in range(3):
            g = grad_phi[dim]
            forces[:, dim] = -(
                    wx0 * wy0 * wz0 * g[i0, j0, k0] +
                    wx1 * wy0 * wz0 * g[i1, j0, k0] +
                    wx0 * wy1 * wz0 * g[i0, j1, k0] +
                    wx1 * wy1 * wz0 * g[i1, j1, k0] +
                    wx0 * wy0 * wz1 * g[i0, j0, k1] +
                    wx1 * wy0 * wz1 * g[i1, j0, k1] +
                    wx0 * wy1 * wz1 * g[i0, j1, k1] +
                    wx1 * wy1 * wz1 * g[i1, j1, k1]
            )

        return forces

    def integrate_leapfrog(self, potential_grid, dt):
        """
        Leapfrog integration: kick-drift-kick
        """
        # Half kick
        forces = self.interpolate_force_from_grid(potential_grid)
        self.velocities += (forces / self.m_particle) * (dt / 2)

        # Full drift
        self.positions += self.velocities * dt

        # Apply periodic boundary conditions
        for dim in range(3):
            low, high = self.simulation.boundaries[dim]
            width = high - low
            self.positions[:, dim] = ((self.positions[:, dim] - low) % width) + low

        # Half kick
        forces = self.interpolate_force_from_grid(potential_grid)
        self.velocities += (forces / self.m_particle) * (dt / 2)