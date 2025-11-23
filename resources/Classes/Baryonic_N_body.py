import cupy as cp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cupyx.scipy.ndimage as ndimage


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
        Initialize N-body baryon particles. The core functionality uses CuPy
        for GPU acceleration, with CPU transfer only for SciPy's
        RegularGridInterpolator.

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

        # Particle data (N_particles, 3) - **Stored on GPU with CuPy**
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
            if radius is None:
                radius = 1  # This would be scale_radius
            if velocity is None:
                velocity = (0.0, 0.0, 0.0)
            if center is None:
                center = (0.0, 0.0, 0.0)
            self.initialize_hernquist(
                center=center,
                velocity=velocity,
                vel_sigma=vel_sigma
            )

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

        elif init_profile == "disk":
            if radius is None:
                radius = 0.1
            if velocity is None:
                velocity = (0.0, 0.0, 0.0)
            self.initialize_exponential_disk(
                center=center,
                radius=radius,
                velocity=velocity,
                vel_sigma=vel_sigma
            )

        else:
            raise ValueError(f"Unknown init_profile: {init_profile}")

    def initialize_hernquist(self, center=None, velocity=None, vel_sigma=0.0, angular_momentum=None,
                             rotation_axis=None):
        """
        Sample particle positions from Hernquist profile (GPU: CuPy).
        Optionally truncates at self.truncation_radius.

        Parameters:
        -----------
        center : tuple or None
            (x, y, z) coordinates for the center of the clump.
        velocity : tuple or None
            (vx, vy, vz) bulk velocity of the clump.
        vel_sigma : float
            Velocity dispersion to add random velocities.
        """
        # Set defaults
        if center is None:
            center = (0.0, 0.0, 0.0)
        if velocity is None:
            velocity = (0.0, 0.0, 0.0)

        if self.truncation_radius is None:
            # Standard Hernquist sampling (GPU)
            u = cp.random.uniform(0, 1, self.N)
            r = self.scale_radius * cp.sqrt(u) / (1 - cp.sqrt(u))
        else:
            r_max = self.truncation_radius
            a = self.scale_radius

            # Maximum of cumulative mass at truncation
            M_trunc = r_max ** 2 / (r_max + a) ** 2

            # Sample from truncated distribution (GPU)
            u = cp.random.uniform(0, M_trunc, self.N)
            r = a * cp.sqrt(u) / (1 - cp.sqrt(u))

            # Double check truncation (GPU)
            r = cp.minimum(r, r_max)

        # Random directions (spherical coordinates) (GPU)
        theta = cp.arccos(2 * cp.random.uniform(0, 1, self.N) - 1)
        phi = 2 * cp.pi * cp.random.uniform(0, 1, self.N)

        # Convert to Cartesian (centered at origin initially) (GPU)
        self.positions[:, 0] = r * cp.sin(theta) * cp.cos(phi)
        self.positions[:, 1] = r * cp.sin(theta) * cp.sin(phi)
        self.positions[:, 2] = r * cp.cos(theta)

        # Apply center offset (GPU)
        self.positions[:, 0] += center[0]
        self.positions[:, 1] += center[1]
        self.positions[:, 2] += center[2]

        self.velocities[:, 0] = velocity[0]
        self.velocities[:, 1] = velocity[1]
        self.velocities[:, 2] = velocity[2]

        # Add rotation if requested (GPU)
        if angular_momentum is not None:
            print("here")
            if rotation_axis is None:
                rotation_axis = (0.0, 0.0, 1.0)

            # Normalize rotation axis (GPU)
            axis = cp.array(rotation_axis)
            axis = axis / cp.sqrt(cp.sum(axis ** 2))

            # Get cylindrical radius from rotation axis (GPU)
            rel_pos = self.positions - cp.array(center)
            R_cyl = cp.sqrt(cp.sum(rel_pos[:, :2] ** 2, axis=1))  # xy-plane distance

            # Circular velocity: v = L / R (GPU)
            v_circ = angular_momentum / (R_cyl + 1e-10)  # avoid divide by zero

            # Tangential direction (GPU)
            v_tang_x = -rel_pos[:, 1] / (R_cyl + 1e-10) * v_circ
            v_tang_y = rel_pos[:, 0] / (R_cyl + 1e-10) * v_circ

            self.velocities[:, 0] += v_tang_x
            self.velocities[:, 1] += v_tang_y

        # Add velocity dispersion if requested (GPU)
        if vel_sigma > 0:
            self.velocities += cp.random.normal(0, vel_sigma, (self.N, 3))

    def initialize_uniform(self):
        """Uniform random distribution in simulation box (GPU: CuPy)"""
        for dim in range(3):
            low, high = self.simulation.boundaries[dim]
            self.positions[:, dim] = cp.random.uniform(low, high, self.N)

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
        # This avoids accessing the CPU-based 'grids' array
        dx, dy, dz = sim.dx

        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

        # 3. Periodic Wrap of Positions (GPU)
        # self.positions is (N_particles, 3) on GPU
        px = x_min + cp.mod(self.positions[:, 0] - x_min, Lx)
        py = y_min + cp.mod(self.positions[:, 1] - y_min, Ly)
        pz = z_min + cp.mod(self.positions[:, 2] - z_min, Lz)

        # 4. Convert to Fractional Grid Indices (GPU)
        # ix, iy, iz are effectively 'float' indices
        fx = (px - x_min) / dx
        fy = (py - y_min) / dy
        fz = (pz - z_min) / dz

        # 5. Cloud-In-Cell (CIC) Interpolation Weights (GPU)
        # Floor to get the bottom-left-front cell index
        i0 = cp.floor(fx).astype(cp.int32)
        j0 = cp.floor(fy).astype(cp.int32)
        k0 = cp.floor(fz).astype(cp.int32)

        # Calculate fractional offset (0.0 to 1.0)
        tx = fx - i0
        ty = fy - j0
        tz = fz - k0

        # Weights for the 8 corners of the cube
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
        # Mass per cell volume
        cell_volume = dx * dy * dz
        mass_val = self.m_particle / cell_volume

        # We deposit mass into the 8 corners using atomic adds
        # Define the 8 corner weights and indices
        corners_weights = [
            (wx0 * wy0 * wz0, i0, j0, k0), (wx1 * wy0 * wz0, i1, j0, k0),
            (wx0 * wy1 * wz0, i0, j1, k0), (wx1 * wy1 * wz0, i1, j1, k0),
            (wx0 * wy0 * wz1, i0, j0, k1), (wx1 * wy0 * wz1, i1, j0, k1),
            (wx0 * wy1 * wz1, i0, j1, k1), (wx1 * wy1 * wz1, i1, j1, k1)
        ]

        # Flatten resolution for add.at
        N_sq = N * N

        for w, ii, jj, kk in corners_weights:
            # Calculate flat index: i*N^2 + j*N + k
            flat_indices = ii * N_sq + jj * N + kk

            # Contribution to this corner
            contribution = mass_val * w

            # Atomic add to the grid
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


        coords = cp.empty((3, self.N), dtype=cp.float32)

        coords[0] = (self.positions[:, 0] - x_min) / dx
        coords[1] = (self.positions[:, 1] - y_min) / dy
        coords[2] = (self.positions[:, 2] - z_min) / dz

        Fx = ndimage.map_coordinates(Fx_grid, coords, order=1, mode='wrap')
        Fy = ndimage.map_coordinates(Fy_grid, coords, order=1, mode='wrap')
        Fz = ndimage.map_coordinates(Fz_grid, coords, order=1, mode='wrap')

        # Stack into (N, 3)
        forces = cp.stack((Fx, Fy, Fz), axis=1)

        return forces

    def integrate_leapfrog(self, potential_grid, dt):
        """
        Leapfrog integration: kick-drift-kick.
        Drift is on GPU (CuPy). Kick involves GPU->CPU->GPU transfer for force interpolation.
        """
        # Half kick
        forces = self.interpolate_force_from_grid(potential_grid)
        self.velocities += forces * (dt / 2)  # (GPU)

        # Full drift
        self.positions += self.velocities * dt  # (GPU)

        # Apply periodic boundary conditions
        for dim in range(3):
            low, high = self.simulation.boundaries[dim]
            width = high - low
            self.positions[:, dim] = ((self.positions[:, dim] - low) % width) + low  # (GPU)

        # Half kick
        forces = self.interpolate_force_from_grid(potential_grid)
        self.velocities += forces * (dt / 2)  # (GPU)

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
        Place most particles in a tight Gaussian around `center` (GPU: CuPy).
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

        # Main clump positions (GPU)
        self.positions[:N_main, :] = cen + pos_sigma * cp.random.standard_normal((N_main, 3), dtype=cp.float32)

        # Periodic wrap (GPU)
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            self.positions[:N_main, d] = ((self.positions[:N_main, d] - low) % L) + low

        # Background (GPU)
        if N_bg > 0:
            for d in range(3):
                low, high = sim.boundaries[d]
                self.positions[N_main:, d] = cp.random.uniform(low, high, size=(N_bg,), dtype=cp.float32)

        # Velocities (GPU)
        if vel_sigma > 0.0:
            self.velocities[:] = bulk + vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float32)
        else:
            self.velocities[:] = bulk

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

    def initialize_circular_ring(self, center=(0.5, 0.5, 0.5), radius=0.05, plane="xy",
                                 velocity=None, vel_sigma=0.0):
        """Deterministic ring of N particles at fixed radius (GPU: CuPy)."""
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

        self.positions = cen[None, :] + radius * ex  # (GPU)

        if velocity is None:
            # Requires force calculation which might involve CPU transfer
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

        # Velocity assignment (GPU)
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
        """3D spherically symmetric clump around center (GPU: CuPy)."""
        sim = self.simulation
        N = self.N

        cen = cp.asarray(center, dtype=cp.float32)

        # Uniform in sphere (GPU)
        dirs = cp.random.normal(size=(N, 3)).astype(cp.float32)
        norms = cp.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= cp.maximum(norms, 1e-6)

        u = cp.random.uniform(0.0, 1.0, size=(N, 1)).astype(cp.float32)
        radii = radius * u ** (1.0 / 3.0)

        offsets = dirs * radii
        self.positions = cen[None, :] + offsets  # (GPU)

        # Periodic wrap (GPU)
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            self.positions[:, d] = ((self.positions[:, d] - low) % L) + low

        # Velocities (GPU)
        base_v = cp.asarray(velocity, dtype=cp.float32)
        v = cp.tile(base_v[None, :], (N, 1))

        if vel_sigma > 0.0:
            noise = vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float32)
            noise[:, 2] = 0.0
            v += noise

        self.velocities = v

    def initialize_exponential_disk(self, center=None, velocity=None,
                                    height=0.1, radius=1.0,
                                    circular_velocity=200.0, vel_sigma=0.0):

        """
        Create an exponential disk with rotation (GPU: CuPy).
        """
        if center is None:
            center = (0.0, 0.0, 0.0)
        if velocity is None:
            velocity = (0.0, 0.0, 0.0)

        # Sample radial positions (exponential) (GPU)
        u = cp.random.uniform(0, 1, self.N)
        R = -radius * cp.log(1 - u)

        # Sample vertical positions (exponential) (GPU)
        z_sign = cp.random.choice([-1, 1], self.N)
        z = z_sign * height * cp.random.exponential(1.0, self.N)

        # Random angles (GPU)
        phi = 2 * cp.pi * cp.random.uniform(0, 1, self.N)

        # Cartesian positions (GPU)
        self.positions[:, 0] = R * cp.cos(phi) + center[0]
        self.positions[:, 1] = R * cp.sin(phi) + center[1]
        self.positions[:, 2] = z + center[2]

        # Circular velocities (simplified flat rotation curve) (GPU)
        v_circ = cp.full(self.N, circular_velocity * 1.0227, dtype=cp.float32)  # Convert km/s → kpc/Gyr

        # Tangential velocities (GPU)
        # Note: The original code sets v_z to 0.0 then adds bulk velocity[2] later.
        self.velocities[:, 0] = -v_circ * cp.sin(phi)
        self.velocities[:, 1] = v_circ * cp.cos(phi)
        self.velocities[:, 2] = 0.0

        # Add bulk COM velocity (GPU)
        self.velocities += cp.array(velocity)

        # Small **vertical** dispersion (GPU)
        sigma = vel_sigma * 1.0227  # km/s → kpc/Gyr
        self.velocities[:, 2] += cp.random.normal(0, sigma, self.N)