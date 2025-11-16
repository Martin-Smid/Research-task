import cupy as cp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class NBodyBaryons:
    def __init__(self, simulation, N_particles, total_mass,
                 init_profile="hernquist",
                 scale_radius=0.5,
                 center=None,
                 velocity=None,
                 momenta=None,
                 radius=None,
                 truncation_radius=None,
                 vel_sigma=0.0,
                 pos_sigma=None,
                 **kwargs):
        """
        Initialize N-body baryon particles.

        Parameters
        ----------
        simulation : Simulation object
            The main simulation object containing grid info and boundaries
        N_particles : int
            Number of particles to simulate
        total_mass : float
            Total mass of all particles
        init_profile : str
            Profile type: "hernquist", "uniform", "cold_clump", "ring", "spherical_clump"
        scale_radius : float
            Scale radius for Hernquist profile (default: 0.5)
        center : tuple of 3 floats, optional
            Center position for clump/ring profiles. If None, uses box center.
        velocity : tuple of 3 floats, optional
            Velocity vector for particles. If None, uses zero or auto-computed.
        momenta : tuple of 3 floats, optional
            Momentum vector (alternative to velocity). Will be divided by particle mass.
        radius : float, optional
            Radius for clump/ring profiles. For Hernquist, acts as truncation_radius if specified.
        truncation_radius : float, optional
            Maximum radius for truncated Hernquist. Overrides radius if both specified.
        vel_sigma : float
            Velocity dispersion (1-sigma) for random velocities (default: 0.0)
        pos_sigma : float, optional
            Position dispersion for clump profiles. If None, auto-computed.
        **kwargs : dict
            Additional profile-specific parameters
        """
        self.simulation = simulation
        self.N = N_particles
        self.m_particle = total_mass / N_particles
        self.scale_radius = scale_radius
        # For Hernquist: radius param can specify truncation
        self.truncation_radius = truncation_radius if truncation_radius is not None else radius

        # Particle data (N_particles, 3)
        self.positions = cp.zeros((N_particles, 3), dtype=cp.float64)

        # Default: small thermal velocities
        if init_profile in ["hernquist", "uniform"] and velocity is None and momenta is None:
            sigma = 0.1
            self.velocities = cp.random.normal(0, sigma, size=(N_particles, 3))
        else:
            self.velocities = cp.zeros((N_particles, 3), dtype=cp.float64)

        # Set default center to box center if not specified
        if center is None:
            center = tuple(
                0.5 * (self.simulation.boundaries[i][0] + self.simulation.boundaries[i][1])
                for i in range(3)
            )

        # Handle velocity vs momenta
        if momenta is not None and velocity is not None:
            raise ValueError("Cannot specify both 'velocity' and 'momenta'. Choose one.")

        if momenta is not None:
            velocity = tuple(p / self.m_particle for p in momenta)

        # Initialize spatial distribution
        if init_profile == "hernquist":
            self.initialize_hernquist()

        elif init_profile == "uniform":
            self.initialize_uniform()

        elif init_profile == "cold_clump":
            frac_main = kwargs.get('frac_main', 0.9)
            as_momenta = kwargs.get('as_momenta', False)
            self.initialize_cold_clump(
                center=center,
                velocity=velocity if velocity is not None else (0.0, 0.0, 0.0),
                frac_main=frac_main,
                pos_sigma=pos_sigma,
                vel_sigma=vel_sigma,
                as_momenta=as_momenta
            )

        elif init_profile == "ring":
            plane = kwargs.get('plane', 'xy')
            if radius is None:
                radius = 0.05
            self.initialize_circular_ring(
                center=center,
                radius=radius,
                plane=plane,
                velocity=velocity,
                vel_sigma=vel_sigma
            )

        elif init_profile == "spherical_clump":
            if radius is None:
                radius = 0.1
            if velocity is None:
                velocity = (0.0, 0.0, 0.0)
            self.initialize_spherical_clump(
                center=center,
                radius=radius,
                velocity=velocity,
                vel_sigma=vel_sigma
            )
        else:
            raise ValueError(f"Unknown init_profile: {init_profile}")

    def initialize_hernquist(self):
        """
        Sample particle positions from Hernquist profile.
        Optionally truncates at self.truncation_radius.
        """
        if self.truncation_radius is None:
            # Standard Hernquist sampling
            u = cp.random.uniform(0, 1, self.N)
            r = self.scale_radius * cp.sqrt(u) / (1 - cp.sqrt(u))
        else:
            # Truncated Hernquist: rejection sampling
            r_max = self.truncation_radius
            a = self.scale_radius

            # Maximum of cumulative mass at truncation
            M_trunc = r_max ** 2 / (r_max + a) ** 2

            # Sample from truncated distribution
            u = cp.random.uniform(0, M_trunc, self.N)
            r = a * cp.sqrt(u) / (1 - cp.sqrt(u))

            # Double check truncation (should be automatic, but safety)
            r = cp.minimum(r, r_max)

        # Random directions (spherical coordinates)
        theta = cp.arccos(2 * cp.random.uniform(0, 1, self.N) - 1)
        phi = 2 * cp.pi * cp.random.uniform(0, 1, self.N)

        # Convert to Cartesian (centered at origin initially)
        self.positions[:, 0] = r * cp.sin(theta) * cp.cos(phi)
        self.positions[:, 1] = r * cp.sin(theta) * cp.sin(phi)
        self.positions[:, 2] = r * cp.cos(theta)

        # Velocities: start at rest
        self.velocities[:, :] = 0

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
        Return forces at particle positions by:
          1) computing F = -∇Φ on the grid (central differences),
          2) interpolating (Fx, Fy, Fz) from the grid to particle positions
             using SciPy's RegularGridInterpolator.

        Inputs
        ------
        potential_grid : cupy.ndarray (Nx, Ny, Nz)
            Gravitational potential Φ on the simulation grid.

        Returns
        -------
        forces : cupy.ndarray (N, 3)
            Interpolated forces at particle positions.
        """
        sim = self.simulation

        # --- grid axes as 1D numpy arrays (assumes your grids are regular) ---
        x = cp.asnumpy(sim.grids[0][:, 0, 0])
        y = cp.asnumpy(sim.grids[1][0, :, 0])
        z = cp.asnumpy(sim.grids[2][0, 0, :])

        # Spacings (assumed uniform)
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        dz = float(z[1] - z[0])

        Nx, Ny, Nz = potential_grid.shape
        kx = 2 * cp.pi * cp.fft.fftfreq(Nx, d=dx)
        ky = 2 * cp.pi * cp.fft.fftfreq(Ny, d=dy)
        kz = 2 * cp.pi * cp.fft.fftfreq(Nz, d=dz)

        Phi_k = cp.fft.fftn(potential_grid)
        dphidx = cp.real(cp.fft.ifftn(1j * kx[:, None, None] * Phi_k))
        dphidy = cp.real(cp.fft.ifftn(1j * ky[None, :, None] * Phi_k))
        dphidz = cp.real(cp.fft.ifftn(1j * kz[None, None, :] * Phi_k))

        Fx_grid = cp.asnumpy(-dphidx)
        Fy_grid = cp.asnumpy(-dphidy)
        Fz_grid = cp.asnumpy(-dphidz)

        Fx_interp = RegularGridInterpolator((x, y, z), Fx_grid, bounds_error=False, fill_value=None)
        Fy_interp = RegularGridInterpolator((x, y, z), Fy_grid, bounds_error=False, fill_value=None)
        Fz_interp = RegularGridInterpolator((x, y, z), Fz_grid, bounds_error=False, fill_value=None)

        # --- query points: particle positions in domain, wrapped periodically ---
        pos = cp.asnumpy(self.positions)  # (N, 3) -> numpy

        # Periodic wrap into [low, high) in each dim
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            pos[:, d] = ((pos[:, d] - low) % L) + low

        # Interpolate forces at particle positions
        Fx = Fx_interp(pos)
        Fy = Fy_interp(pos)
        Fz = Fz_interp(pos)

        forces = np.column_stack([Fx, Fy, Fz])
        return cp.asarray(forces)

    def integrate_leapfrog(self, potential_grid, dt):
        """
        Leapfrog integration: kick-drift-kick
        """
        # Half kick
        forces = self.interpolate_force_from_grid(potential_grid)
        self.velocities += forces * (dt / 2)

        # Full drift
        self.positions += self.velocities * dt

        # Apply periodic boundary conditions
        for dim in range(3):
            low, high = self.simulation.boundaries[dim]
            width = high - low
            self.positions[:, dim] = ((self.positions[:, dim] - low) % width) + low

        # Half kick
        forces = self.interpolate_force_from_grid(potential_grid)
        self.velocities += forces * (dt / 2)

    def initialize_cold_clump(
            self,
            center=(5.0, 0.0, 0.0),
            velocity=(0.0, 0.9485, 0.0),
            frac_main=0.9,
            pos_sigma=None,
            vel_sigma=0.0,
            as_momenta=False,
    ):
        """
        Place most particles in a tight Gaussian around `center` with bulk `velocity`.
        """
        sim = self.simulation
        N = self.N

        # Grid metrics
        x_axis = sim.grids[0][:, 0, 0]
        dx = float(x_axis[1] - x_axis[0])
        if pos_sigma is None:
            pos_sigma = 0.2 * dx

        cen = cp.asarray(center, dtype=cp.float32)
        bulk = cp.asarray(velocity, dtype=cp.float32)
        if as_momenta:
            bulk = bulk / cp.float32(self.m_particle)

        N_main = int(max(1, round(frac_main * N)))
        N_bg = N - N_main

        # Main clump positions
        self.positions[:N_main, :] = cen + pos_sigma * cp.random.standard_normal((N_main, 3), dtype=cp.float32)

        # Periodic wrap
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            self.positions[:N_main, d] = ((self.positions[:N_main, d] - low) % L) + low

        # Background
        if N_bg > 0:
            for d in range(3):
                low, high = sim.boundaries[d]
                self.positions[N_main:, d] = cp.random.uniform(low, high, size=(N_bg,), dtype=cp.float32)

        # Velocities
        if vel_sigma > 0.0:
            self.velocities[:] = bulk + vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float32)
        else:
            self.velocities[:] = bulk

    def _circular_speed_from_grid(self, potential_grid, center_xyz):
        """Return v_circ at center_xyz from grid potential."""
        sim = self.simulation

        x = cp.asnumpy(sim.grids[0][:, 0, 0])
        y = cp.asnumpy(sim.grids[1][0, :, 0])
        z = cp.asnumpy(sim.grids[2][0, 0, :])

        Phi = cp.asnumpy(potential_grid)

        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        dz = float(z[1] - z[0])
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

    def initialize_circular_ring(self, center=(0.5, 0.5, 0.5), radius=0.05, plane="xy",
                                 velocity=None, vel_sigma=0.0):
        """Deterministic ring of N particles at fixed radius."""
        N = self.N
        sim = self.simulation

        cen = cp.asarray(center, dtype=cp.float32)
        theta = cp.linspace(0, 2 * cp.pi, N, endpoint=False, dtype=cp.float32)

        if plane == "xy":
            ex = cp.stack((cp.cos(theta), cp.sin(theta), cp.zeros_like(theta)), axis=1)
            tang = cp.stack((-cp.sin(theta), cp.cos(theta), cp.zeros_like(theta)), axis=1)
        elif plane == "xz":
            ex = cp.stack((cp.cos(theta), cp.zeros_like(theta), cp.sin(theta)), axis=1)
            tang = cp.stack((-cp.sin(theta), cp.zeros_like(theta), cp.cos(theta)), axis=1)
        elif plane == "yz":
            ex = cp.stack((cp.zeros_like(theta), cp.cos(theta), cp.sin(theta)), axis=1)
            tang = cp.stack((cp.zeros_like(theta), -cp.sin(theta), cp.cos(theta)), axis=1)
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")

        self.positions = cen[None, :] + radius * ex

        if velocity is None:
            total_density = cp.zeros_like(sim.grids[0], dtype=cp.float32)
            V = sim.propagator.compute_gravity_potential(total_density)
            if sim.static_potential is not None:
                V = V + cp.real(sim.static_potential(sim))
            v_c = self._circular_speed_from_grid(V, tuple(cp.asnumpy(cen)))
            bulk_mag = cp.asarray(v_c, dtype=cp.float32)
        else:
            bulk_mag = cp.asarray(velocity, dtype=cp.float32)
            if bulk_mag.ndim == 0:
                bulk_mag = cp.full((N,), float(bulk_mag), dtype=cp.float32)

        if vel_sigma > 0.0:
            if bulk_mag.ndim == 0:
                self.velocities = bulk_mag * tang + vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float32)
            else:
                self.velocities = bulk_mag[:, None] * tang + vel_sigma * cp.random.standard_normal((N, 3),
                                                                                                   dtype=cp.float32)
        else:
            if bulk_mag.ndim == 0:
                self.velocities = bulk_mag * tang
            else:
                self.velocities = bulk_mag[:, None] * tang

    def initialize_spherical_clump(
            self,
            center=(0.0, 0.0, 0.0),
            radius=0.1,
            velocity=(0.0, 0.0, 0.0),
            vel_sigma=0.0,
    ):
        """3D spherically symmetric clump around center."""
        sim = self.simulation
        N = self.N

        cen = cp.asarray(center, dtype=cp.float32)

        # Uniform in sphere
        dirs = cp.random.normal(size=(N, 3)).astype(cp.float32)
        norms = cp.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= cp.maximum(norms, 1e-6)

        u = cp.random.uniform(0.0, 1.0, size=(N, 1)).astype(cp.float32)
        radii = radius * u ** (1.0 / 3.0)

        offsets = dirs * radii
        self.positions = cen[None, :] + offsets

        # Periodic wrap
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            self.positions[:, d] = ((self.positions[:, d] - low) % L) + low

        # Velocities
        base_v = cp.asarray(velocity, dtype=cp.float32)
        v = cp.tile(base_v[None, :], (N, 1))

        if vel_sigma > 0.0:
            noise = vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float32)
            noise[:, 2] = 0.0
            v += noise

        self.velocities = v