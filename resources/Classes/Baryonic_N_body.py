import cupy as cp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


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

        #TODO: think about what to do with particles outside of the boundary - maybe try to get rid of particles outside of boundary
        #TODO: once you delete a particle reduce the number of particles


        # Initialize spatial distribution
        if init_profile == "hernquist":
            self.initialize_hernquist()
        elif init_profile == "uniform":
            self.initialize_uniform()
        elif init_profile == "cold_clump":
            # kwargs (all optional): center, velocity, frac_main, pos_sigma, vel_sigma, as_momenta
            self.initialize_cold_clump()
        elif init_profile == "ring":
            # defaults: center box middle, small radius, xy plane, auto v_circ
            self.initialize_circular_ring(center=(5, 0, 0), radius=0.05, plane="xy",
                                          velocity=None, vel_sigma=0.0)
        elif init_profile == "spherical_clump":
            # planar, circularly symmetric clump at r≈5 with fixed tangential v
            self.initialize_spherical_clump(center=(0.0, 0.0, 0.0),
                                            radius=0.1,
                                            velocity=(0.0, 0.0, 0.0),
                                            vel_sigma=0.0)


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
        # If your grids are stored as 3D broadcast arrays, these slices match your usage elsewhere:
        #   x: sim.grids[0][:,0,0], y: sim.grids[1][0,:,0], z: sim.grids[2][0,0,:]
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


        # bounds_error=False + fill_value=None -> linear extrapolation at edges (but we’ll wrap anyway)
        Fx_interp = RegularGridInterpolator((x, y, z), Fx_grid, bounds_error=False, fill_value=None)
        Fy_interp = RegularGridInterpolator((x, y, z), Fy_grid, bounds_error=False, fill_value=None)
        Fz_interp = RegularGridInterpolator((x, y, z), Fz_grid, bounds_error=False, fill_value=None)

        # --- query points: particle positions in domain, wrapped periodically like before ---
        pos = cp.asnumpy(self.positions)  # (N, 3) -> numpy

        # Periodic wrap into [low, high) in each dim, using your simulation boundaries
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            pos[:, d] = ((pos[:, d] - low) % L) + low

        # Interpolate forces at particle positions
        # RegularGridInterpolator expects points as (N, 3) in the same axis order as (x,y,z)
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
        Great for circular-orbit sanity tests against a central point mass.

        Parameters
        ----------
        center : tuple of 3 floats
            Target position in box units (same units as sim.grids).
        velocity : tuple of 3 floats
            Either velocity or momentum vector (see `as_momenta`).
        frac_main : float in (0,1]
            Fraction of particles in the main clump (rest sprinkled uniformly).
        pos_sigma : float or None
            1σ positional spread for the clump (same units as grid). If None, uses 0.2*Δx.
        vel_sigma : float
            1σ velocity spread added isotropically (same units as `velocity`).
        as_momenta : bool
            If True, interprets `velocity` as momentum p and divides by m_particle.
        """
        sim = self.simulation
        N = self.N

        # Grid metrics (uniform spacing assumed)
        x_axis = sim.grids[0][:, 0, 0]
        dx = float(x_axis[1] - x_axis[0])
        if pos_sigma is None:
            pos_sigma = 0.2 * dx  # tight clump by default

        # Parse inputs
        cen = cp.asarray(center, dtype=cp.float32)
        bulk = cp.asarray(velocity, dtype=cp.float32)
        if as_momenta:
            bulk = bulk / cp.float32(self.m_particle)

        # How many in clump vs background
        N_main = int(max(1, round(frac_main * N)))
        N_bg = N - N_main

        # --- Main clump positions (Gaussian around center) ---
        self.positions[:N_main, :] = cen + pos_sigma * cp.random.standard_normal((N_main, 3), dtype=cp.float32)

        # Periodic wrap into box
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            self.positions[:N_main, d] = ((self.positions[:N_main, d] - low) % L) + low

        # --- Background (optional): sprinkle uniformly in box to avoid exact δ ---
        if N_bg > 0:
            for d in range(3):
                low, high = sim.boundaries[d]
                self.positions[N_main:, d] = cp.random.uniform(low, high, size=(N_bg,), dtype=cp.float32)

        # --- Velocities: bulk + small Gaussian noise ---
        # derive circular speed if velocity is None
        if velocity is None:
            # compose the same potential used during evolution
            total_density = cp.zeros_like(self.simulation.grids[0], dtype=cp.float32)
            V = self.simulation.propagator.compute_gravity_potential(total_density)
            if self.simulation.static_potential is not None:
                V = V + cp.real(self.simulation.static_potential(self.simulation))
            v_c = self._circular_speed_from_grid(V, tuple(cen.tolist()))
            # set bulk tangential velocity perpendicular to radius (xy-plane)
            bulk = cp.asarray((0.0, v_c, 0.0), dtype=cp.float32)

        # --- Velocities: bulk + small Gaussian noise ---
        if vel_sigma > 0.0:
            self.velocities[:] = bulk + vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float32)
        else:
            self.velocities[:] = bulk

    def _circular_speed_from_grid(self, potential_grid, center_xyz):
        """Return v_circ at center_xyz from grid potential: v_circ = sqrt(r * |dPhi/dr|)."""
        sim = self.simulation

        # 1D axes as numpy
        x = cp.asnumpy(sim.grids[0][:, 0, 0])
        y = cp.asnumpy(sim.grids[1][0, :, 0])
        z = cp.asnumpy(sim.grids[2][0, 0, :])

        Phi = cp.asnumpy(potential_grid)

        # FFT gradients on CPU for simplicity (consistent with your interpolate_force)
        dx = float(x[1] - x[0]);
        dy = float(y[1] - y[0]);
        dz = float(z[1] - z[0])
        kx = 2 * np.pi * np.fft.fftfreq(Phi.shape[0], d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(Phi.shape[1], d=dy)
        kz = 2 * np.pi * np.fft.fftfreq(Phi.shape[2], d=dz)
        Phi_k = np.fft.fftn(Phi)
        dphidx = np.fft.ifftn(1j * kx[:, None, None] * Phi_k).real
        dphidy = np.fft.ifftn(1j * ky[None, :, None] * Phi_k).real
        dphidz = np.fft.ifftn(1j * kz[None, None, :] * Phi_k).real

        # locate nearest cell to center
        cx, cy, cz = center_xyz
        ix = np.argmin(np.abs(x - cx))
        iy = np.argmin(np.abs(y - cy))
        iz = np.argmin(np.abs(z - cz))

        # radial unit vector from box center to (cx,cy,cz)
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
        """
        Deterministic ring of N particles at a fixed radius around 'center'.
        No stochastic placement. Angles are uniformly spaced.
        plane: 'xy' | 'xz' | 'yz'
        If velocity=None, set tangential speed to v_circ from the grid potential at 'center'.
        """
        N = self.N
        sim = self.simulation

        cen = cp.asarray(center, dtype=cp.float32)
        theta = cp.linspace(0, 2 * cp.pi, N, endpoint=False, dtype=cp.float32)

        # Unit vectors in the chosen plane
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

        # Positions: exact circle
        self.positions = cen[None, :] + radius * ex

        # Bulk tangential velocity (consistent with grid potential if not provided)
        if velocity is None:
            # compose potential exactly as in evolution
            total_density = cp.zeros_like(sim.grids[0], dtype=cp.float32)
            V = sim.propagator.compute_gravity_potential(total_density)
            if sim.static_potential is not None:
                V = V + cp.real(sim.static_potential(sim))
            # circular speed at 'center' (same for all ring points for small radius)
            v_c = self._circular_speed_from_grid(V, tuple(cp.asnumpy(cen)))
            bulk_mag = cp.asarray(v_c, dtype=cp.float32)
        else:
            bulk_mag = cp.asarray(velocity, dtype=cp.float32)
            if bulk_mag.ndim == 0:
                # scalar speed → tangential direction for each particle
                bulk_mag = cp.full((N,), float(bulk_mag), dtype=cp.float32)

        # Velocities: purely tangential + optional tiny Gaussian if requested
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

    def initialize_solid_clump(self, center=(0, 5.0, 0.0), radius=0.02,
                               velocity=(0.0, 0.9485, 0.0), vel_sigma=0.0):
        # put clump in the XY mid-plane: z = box center
        sim = self.simulation
        zc = 0.5 * (sim.boundaries[2][0] + sim.boundaries[2][1])

        N = self.N
        cx, cy, _ = center
        cen = cp.asarray((cx, cy, zc), dtype=cp.float32)  # force z to mid-plane

        # deterministic disk in XY (no z thickness)
        n_side = int(cp.ceil(N ** (1 / 2)))
        xs = cp.linspace(-1, 1, n_side, dtype=cp.float32)
        xx, yy = cp.meshgrid(xs, xs, indexing="ij")
        grid = cp.stack((xx.ravel(), yy.ravel(), cp.zeros_like(xx).ravel()), axis=1)[:N]
        norms = cp.maximum(cp.linalg.norm(grid[:, :2], axis=1, keepdims=True), 1e-6)
        pts = (grid / norms) * radius * 0.5
        self.positions = cen[None, :] + pts

        # velocities: exactly in-plane
        vel = cp.asarray(velocity, dtype=cp.float32)
        vel_3d = cp.zeros(3, dtype=cp.float32)
        vel_3d[0] = vel[0]
        vel_3d[1] = vel[1]
        # vel_3d[2] is already 0

        self.velocities = cp.tile(vel_3d[None, :], (N, 1))
        if vel_sigma > 0.0:
            noise = cp.random.standard_normal((N, 3), dtype=cp.float32)
            noise[:, 2] = 0.0  # no z-noise
            self.velocities += vel_sigma * noise

        self.velocities[:, 2] = 0.0

    def initialize_spherical_clump(
            self,
            center=(0.0, 0.0, 0.0),
            radius=0.1,
            velocity=(0.0, 0.0, 0.0),
            vel_sigma=0.0,
    ):
        """
        3D spherically symmetric clump around `center`.

        - Positions are distributed uniformly inside a sphere of radius `radius`.
        - All particles start with the same velocity `velocity`
          (typically tangential in the x–y plane, v_z = 0).
        - Optional small Gaussian scatter in velocities with width `vel_sigma`.
        """

        sim = self.simulation
        N = self.N

        cen = cp.asarray(center, dtype=cp.float32)

        # === positions: uniform in a sphere ===
        # draw random directions
        dirs = cp.random.normal(size=(N, 3)).astype(cp.float32)
        norms = cp.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= cp.maximum(norms, 1e-6)

        # draw radii with p(r) ∝ r^2 -> r = R * u^(1/3)
        u = cp.random.uniform(0.0, 1.0, size=(N, 1)).astype(cp.float32)
        radii = radius * u ** (1.0 / 3.0)

        offsets = dirs * radii
        self.positions = cen[None, :] + offsets

        # periodic wrap into box
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            self.positions[:, d] = ((self.positions[:, d] - low) % L) + low

        # === velocities: all the same (plus optional small noise) ===
        base_v = cp.asarray(velocity, dtype=cp.float32)
        v = cp.tile(base_v[None, :], (N, 1))

        if vel_sigma > 0.0:
            noise = vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float32)
            # if you want to keep the COM strictly in the x–y plane, kill z-noise:
            noise[:, 2] = 0.0
            v += noise

        self.velocities = v

