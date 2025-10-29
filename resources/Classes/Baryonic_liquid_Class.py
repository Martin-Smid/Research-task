import cupy as cp
import numpy as np

import cupy as cp
import numpy as np


class BaryonicMatter_Class:
    def __init__(self, simulation, M_b=1e9, a=0.5, model="hernquist"):
        self.simulation = simulation
        self.grids = simulation.grids
        self.G = simulation.G
        self.M_b = M_b  # total baryonic mass
        self.a = a  # scale radius (same units as simulation.dUnits)
        if model:
            self.model = model.lower()
        else:
            self.model = None

        self.rho_b = self.create_density_profiles()

        # Initialize velocity field (baryonic matter starts at rest)
        shape = (self.simulation.N,) * self.simulation.dim
        self.v_b = [cp.zeros(shape, dtype=cp.float32) for _ in range(self.simulation.dim)]

    def drift_baryonic_matter(self, dt):
        """
        Apply continuity equation: ∂ρ/∂t + ∇·(ρv) = 0
        """
        ks = self.simulation.k_space
        rho = self.rho_b
        rho_k = cp.fft.fftn(rho)

        # Compute (v·∇)ρ term
        adv = 0
        for d, k in enumerate(ks):
            # Use -1j for proper gradient (∂/∂x -> ik in Fourier space)
            grad_rho_d = cp.real(cp.fft.ifftn(1j * k * rho_k))
            adv += self.v_b[d] * grad_rho_d

        # Compute (∇·v) term
        div_v = 0
        for d, k in enumerate(ks):
            v_k = cp.fft.fftn(self.v_b[d])
            div_v += cp.real(cp.fft.ifftn(1j * k * v_k))

        # Update density: ∂ρ/∂t = -(v·∇)ρ - ρ(∇·v)
        self.rho_b = rho - dt * (adv + rho * div_v)

        # Enforce positivity with a small floor
        self.rho_b = cp.maximum(self.rho_b, 1e-30)

    def _advect_field(self, field, velocity, dim, dt):
        """Advect a field along a velocity component."""
        # Simplified: uses spectral method
        dx = self.simulation.dx[dim]

        # Compute ∂field/∂x_dim using FFT
        k_component = self.simulation.k_space[dim]
        field_k = cp.fft.fftn(field)
        grad_field = cp.real(cp.fft.ifftn(-1j * k_component * field_k))

        # Update: ρ = ρ - dt * v * ∂ρ/∂x
        field -= dt * velocity * grad_field

    def update_velocity(self, gravity_potential,dt):
        """
        Update velocity field from acceleration: a = -∇(Φ + self-interaction)
        v_{n+1} = v_n + dt * a
        """
        # Acceleration from gravity
        for dim in range(self.simulation.dim):
            k_component = self.simulation.k_space[dim]
            gravity_k = cp.fft.fftn(gravity_potential)
            accel = cp.real(cp.fft.ifftn(-1j * k_component * gravity_k))
            self.v_b[dim] -= dt * accel


    def create_hernquist_profile(self):
        """Create Hernquist density profile."""
        x, y, z = [np.asarray(g) for g in self.grids]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r[r == 0] = 1e-12

        Mb = self.M_b
        a = self.a
        rho = (Mb / (2 * np.pi)) * (a / (r * (r + a) ** 3))

        self.rho_b = cp.asarray(rho.astype(cp.float32))
        return self.rho_b



    def create_density_profiles(self):
        if self.model == "hernquist":
            return self.create_hernquist_profile()
        elif self.model == None:
            shape = (self.simulation.N,) * self.simulation.dim
            return cp.zeros(shape, dtype=cp.float64)
        elif self.model == "two_clumps":
            return self.create_two_clump_profile()
        else:
            raise ValueError(f"Unknown density model: {self.model}")

    def create_hernquist_profile(self):
        """
        Create a Hernquist density profile:
            ρ(r) = (M_b / (2π)) * a / [r (r + a)^3]
        evaluated on the same grid as the simulation.
        taken from: https://articles.adsabs.harvard.edu/pdf/1990ApJ...356..359H
        eq: 2) and 5)
        """

        x, y, z = [np.asarray(g) for g in self.grids]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # avoids division by 0
        r[r == 0] = 1e-12

        Mb = self.M_b
        a = self.a

        rho = (Mb / (2 * np.pi)) * (a / (r * (r + a) ** 3))


        self.rho_b = cp.asarray(rho)
        return self.rho_b

    def create_two_clump_profile(self):
        """Create two Hernquist clumps to collide"""
        x, y, z = [np.asarray(g) for g in self.grids]

        # Get box center
        cx = 0.5 * (self.simulation.boundaries[0][0] + self.simulation.boundaries[0][1])
        cy = 0.5 * (self.simulation.boundaries[1][0] + self.simulation.boundaries[1][1])
        cz = 0.5 * (self.simulation.boundaries[2][0] + self.simulation.boundaries[2][1])

        separation = 0.3 * (self.simulation.boundaries[0][1] - self.simulation.boundaries[0][0])

        # First clump
        r1 = np.sqrt((x - cx - separation) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        r1[r1 == 0] = 1e-12

        # Second clump
        r2 = np.sqrt((x - cx + separation) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        r2[r2 == 0] = 1e-12

        Mb = self.M_b / 2  # Split mass between two clumps
        a = self.a

        rho1 = (Mb / (2 * np.pi)) * (a / (r1 * (r1 + a) ** 3))
        rho2 = (Mb / (2 * np.pi)) * (a / (r2 * (r2 + a) ** 3))

        self.rho_b = cp.asarray((rho1 + rho2).astype(cp.float32))
        return self.rho_b