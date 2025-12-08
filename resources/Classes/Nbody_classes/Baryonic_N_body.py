import cupy as cp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cupyx.scipy.ndimage as ndimage
from resources.Classes.Nbody_classes.NBody import NBody
import os
import cupy as cp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cupyx.scipy.ndimage as ndimage
from resources.Classes.Nbody_classes.NBody import NBody


class Baryons(NBody):
    """
    Baryonic N-body system: inherits generic NBody mechanics and adds
    profile-specific initialization (Hernquist, clumps, rings, disks, ...).
    """

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
            Profile type: "hernquist", "uniform", "cold_clump", "ring", "spherical_clump", "disk"
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
        # Initialize generic N-body state
        super().__init__(simulation, N_particles, total_mass)

        self.scale_radius = scale_radius
        self.truncation_radius = truncation_radius if truncation_radius is not None else radius

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

        # Default velocity to zero if not specified
        if velocity is None:
            velocity = (0.0, 0.0, 0.0)

        # Default radius for profiles that need it
        if radius is None:
            radius = self._get_default_radius(init_profile)

        # Initialize velocities (small thermal for some profiles, zero otherwise)
        if init_profile in ["hernquist", "uniform"] and momenta is None:
            sigma = 0.1
            self.velocities = cp.random.normal(0, sigma, size=(N_particles, 3))
        else:
            self.velocities = cp.zeros((N_particles, 3), dtype=cp.float64)

        # Route to appropriate initialization method
        if init_profile == "hernquist":
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
                velocity=velocity,
                frac_main=frac_main,
                pos_sigma=pos_sigma,
                vel_sigma=vel_sigma,
                as_momenta=as_momenta
            )

        elif init_profile == "ring":
            plane = kwargs.get('plane', 'xy')
            self.initialize_circular_ring(
                center=center,
                radius=radius,
                plane=plane,
                velocity=velocity,
                vel_sigma=vel_sigma
            )

        elif init_profile == "spherical_clump":
            self.initialize_spherical_clump(
                center=center,
                radius=radius,
                velocity=velocity,
                vel_sigma=vel_sigma
            )

        elif init_profile == "disk":
            self.initialize_exponential_disk(
                center=center,
                radius=radius,
                velocity=velocity,
                vel_sigma=vel_sigma
            )
        elif init_profile == "from_file":
            file_path = kwargs.get('file_path')

            # Default center to the middle of the simulation box
            if center is None:
                center = tuple(
                    0.5 * (self.simulation.boundaries[i][0] + self.simulation.boundaries[i][1])
                    for i in range(3)
                )

            # Check if user wants to offset (Default to True usually for external files)
            apply_center_offset = kwargs.get('apply_center_offset', True)

            # Get scaling factors
            dist_factor = kwargs.get('dist_factor', 1.0)
            vel_factor = kwargs.get('vel_factor', 1.0)

            if file_path is None:
                raise ValueError("init_profile='from_file' requires a 'file_path' argument.")

            self.initialize_from_file(
                file_path,
                center,
                apply_center_offset,
                dist_factor=dist_factor,
                vel_factor=vel_factor
            )

        else:
            raise ValueError(f"Unknown init_profile: {init_profile}")

    def _get_default_radius(self, init_profile):
        """Get default radius for each profile type."""
        defaults = {
            "hernquist": 1.0,
            "ring": 0.05,
            "spherical_clump": 0.1,
            "disk": 0.1
        }
        return defaults.get(init_profile, 1.0)

    def initialize_hernquist(self, center, velocity, vel_sigma=0.0, angular_momentum=None,
                             rotation_axis=None):
        """
        Sample particle positions from Hernquist profile (GPU: CuPy).
        Optionally truncates at self.truncation_radius.

        Parameters:
        -----------
        center : tuple
            (x, y, z) coordinates for the center of the clump.
        velocity : tuple
            (vx, vy, vz) bulk velocity of the clump.
        vel_sigma : float
            Velocity dispersion to add random velocities.
        angular_momentum : float, optional
            If provided, adds rotation to the profile.
        rotation_axis : tuple, optional
            Axis of rotation (default: z-axis).
        """
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

        # Set bulk velocity (GPU)
        self.velocities[:, 0] = velocity[0]
        self.velocities[:, 1] = velocity[1]
        self.velocities[:, 2] = velocity[2]

        # Add rotation if requested (GPU)
        if angular_momentum is not None:
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

    def initialize_cold_clump(self, center, velocity, frac_main=0.9, pos_sigma=None,
                              vel_sigma=0.0, as_momenta=False):
        """
        Place most particles in a tight Gaussian around `center` (GPU: CuPy).

        Parameters:
        -----------
        center : tuple
            (x, y, z) coordinates for the center of the clump.
        velocity : tuple
            (vx, vy, vz) bulk velocity (or momentum if as_momenta=True).
        frac_main : float
            Fraction of particles in the main clump (rest are background).
        pos_sigma : float, optional
            Position dispersion. If None, auto-computed from grid spacing.
        vel_sigma : float
            Velocity dispersion.
        as_momenta : bool
            If True, interpret velocity as momentum.
        """
        sim = self.simulation
        N = self.N

        # Grid metrics
        x_axis = sim.grids[0][:, 0, 0]
        dx = float(x_axis[1] - x_axis[0])
        if pos_sigma is None:
            pos_sigma = 0.2 * dx

        cen = cp.asarray(center, dtype=cp.float64)
        bulk = cp.asarray(velocity, dtype=cp.float64)
        if as_momenta:
            bulk = bulk / cp.float64(self.m_particle)

        N_main = int(max(1, round(frac_main * N)))
        N_bg = N - N_main

        # Main clump positions (GPU)
        self.positions[:N_main, :] = cen + pos_sigma * cp.random.standard_normal((N_main, 3), dtype=cp.float64)

        # Periodic wrap (GPU)
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            self.positions[:N_main, d] = ((self.positions[:N_main, d] - low) % L) + low

        # Background (GPU)
        if N_bg > 0:
            for d in range(3):
                low, high = sim.boundaries[d]
                self.positions[N_main:, d] = cp.random.uniform(low, high, size=(N_bg,), dtype=cp.float64)

        # Velocities (GPU)
        if vel_sigma > 0.0:
            self.velocities[:] = bulk + vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float64)
        else:
            self.velocities[:] = bulk

    def initialize_circular_ring(self, center, radius, plane="xy", velocity=None, vel_sigma=0.0):
        """
        Deterministic ring of N particles at fixed radius (GPU: CuPy).

        Parameters:
        -----------
        center : tuple
            (x, y, z) coordinates for the center of the ring.
        radius : float
            Radius of the ring.
        plane : str
            Plane of the ring: 'xy', 'xz', or 'yz'.
        velocity : float, tuple, or None
            If None, compute circular velocity from potential.
            If float, use as magnitude of circular velocity.
            If tuple, interpret as bulk velocity vector.
        vel_sigma : float
            Velocity dispersion.
        """
        N = self.N
        sim = self.simulation

        cen = cp.asarray(center, dtype=cp.float64)
        theta = cp.linspace(0, 2 * cp.pi, N, endpoint=False, dtype=cp.float64)

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
            # Compute circular velocity from potential
            total_density = cp.zeros_like(sim.grids[0], dtype=cp.float64)
            V = sim.propagator.compute_gravity_potential(total_density)
            if sim.static_potential is not None:
                V = V + cp.real(sim.static_potential(sim))
            v_c = self._circular_speed_from_grid(V, tuple(cp.asnumpy(cen)))
            bulk_mag = cp.asarray(v_c, dtype=cp.float64)
        else:
            bulk_mag = cp.asarray(velocity, dtype=cp.float64)
            if bulk_mag.ndim == 0:
                bulk_mag = cp.full((N,), float(bulk_mag), dtype=cp.float64)

        # Velocity assignment (GPU)
        if vel_sigma > 0.0:
            if bulk_mag.ndim == 0:
                self.velocities = bulk_mag * tang + vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float64)
            else:
                self.velocities = bulk_mag[:, None] * tang + vel_sigma * cp.random.standard_normal((N, 3),
                                                                                                   dtype=cp.float64)
        else:
            if bulk_mag.ndim == 0:
                self.velocities = bulk_mag * tang
            else:
                self.velocities = bulk_mag[:, None] * tang

    def initialize_spherical_clump(self, center, radius, velocity, vel_sigma=0.0):
        """
        3D spherically symmetric clump around center (GPU: CuPy).

        Parameters:
        -----------
        center : tuple
            (x, y, z) coordinates for the center of the clump.
        radius : float
            Radius of the spherical clump.
        velocity : tuple
            (vx, vy, vz) bulk velocity of the clump.
        vel_sigma : float
            Velocity dispersion (applied only in x,y directions).
        """
        sim = self.simulation
        N = self.N

        cen = cp.asarray(center, dtype=cp.float64)

        # Uniform in sphere (GPU)
        dirs = cp.random.normal(size=(N, 3)).astype(cp.float64)
        norms = cp.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= cp.maximum(norms, 1e-6)

        u = cp.random.uniform(0.0, 1.0, size=(N, 1)).astype(cp.float64)
        radii = radius * u ** (1.0 / 3.0)

        offsets = dirs * radii
        self.positions = cen[None, :] + offsets  # (GPU)

        # Periodic wrap (GPU)
        for d in range(3):
            low, high = sim.boundaries[d]
            L = (high - low)
            self.positions[:, d] = ((self.positions[:, d] - low) % L) + low

        # Velocities (GPU)
        base_v = cp.asarray(velocity, dtype=cp.float64)
        v = cp.tile(base_v[None, :], (N, 1))

        if vel_sigma > 0.0:
            noise = vel_sigma * cp.random.standard_normal((N, 3), dtype=cp.float64)
            noise[:, 2] = 0.0
            v += noise

        self.velocities = v

    def initialize_exponential_disk(self, center, velocity, radius=1.0, height=0.1,
                                    circular_velocity=200.0, vel_sigma=0.0):
        """
        Create an exponential disk with rotation (GPU: CuPy).

        Parameters:
        -----------
        center : tuple
            (x, y, z) coordinates for the center of the disk.
        velocity : tuple
            (vx, vy, vz) bulk velocity of the disk.
        radius : float
            Scale radius of the exponential disk.
        height : float
            Scale height of the disk.
        circular_velocity : float
            Circular velocity in km/s (will be converted to kpc/Gyr).
        vel_sigma : float
            Vertical velocity dispersion in km/s.
        """
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
        v_circ = cp.full(self.N, circular_velocity * 1.0227, dtype=cp.float64)  # Convert km/s → kpc/Gyr

        # Tangential velocities (GPU)
        self.velocities[:, 0] = -v_circ * cp.sin(phi)
        self.velocities[:, 1] = v_circ * cp.cos(phi)
        self.velocities[:, 2] = 0.0

        # Add bulk COM velocity (GPU)
        self.velocities += cp.array(velocity)

        # Small **vertical** dispersion (GPU)
        sigma = vel_sigma * 1.0227  # km/s → kpc/Gyr
        self.velocities[:, 2] += cp.random.normal(0, sigma, self.N)

    def initialize_from_file(self, file_path, center, apply_center_offset=False, dist_factor=1.0, vel_factor=1.0):
        """
        Load particle data from a binary file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Binary file not found: {file_path}")

        print(f"Loading N-body data from: {file_path}")

        raw_data = np.fromfile(file_path, dtype=np.float64)

        # 1. Extract Particle Mass
        self.m_particle = float(raw_data[0])

        # 2. Extract Data Arrays
        data = raw_data[1:]
        N_in_file = data.size // 6

        print(f"  -> Applying Distance Factor: {dist_factor}")

        # Slice and Scale
        x_cpu = data[:N_in_file] * dist_factor
        y_cpu = data[N_in_file: 2 * N_in_file] * dist_factor
        z_cpu = data[2 * N_in_file: 3 * N_in_file] * dist_factor

        vx_cpu = data[3 * N_in_file: 4 * N_in_file] * dist_factor
        vy_cpu = data[4 * N_in_file: 5 * N_in_file] * dist_factor
        vz_cpu = data[5 * N_in_file:] * dist_factor

        # 3. Handle Mismatch
        if N_in_file < self.N:
            raise ValueError(f"Simulation requires {self.N} particles, but file only contains {N_in_file}.")

        indices = np.arange(self.N)

        # Transfer to GPU
        self.positions[:, 0] = cp.asarray(x_cpu[indices])
        self.positions[:, 1] = cp.asarray(y_cpu[indices])
        self.positions[:, 2] = cp.asarray(z_cpu[indices])

        self.velocities[:, 0] = cp.asarray(vx_cpu[indices])
        self.velocities[:, 1] = cp.asarray(vy_cpu[indices])
        self.velocities[:, 2] = cp.asarray(vz_cpu[indices])

        # 4. Apply Center Offset
        # This fixes the "Corners" bug by moving (0,0,0) to (Box/2, Box/2, Box/2)
        if apply_center_offset:
            print(f"  -> Shifting particles by center: {center}")
            self.positions[:, 0] += center[0]
            self.positions[:, 1] += center[1]
            self.positions[:, 2] += center[2]

        # 5. Periodic Wrap (Safety check)
        sim = self.simulation
        for dim in range(3):
            low, high = sim.boundaries[dim]
            L = high - low
            # The wrap ensures particles are strictly inside [low, high]
            self.positions[:, dim] = ((self.positions[:, dim] - low) % L) + low