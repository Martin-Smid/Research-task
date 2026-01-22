import cupy as cp
import numpy as np


class NBodyGas:
    """
    Euler equations solver for compressible gas on a uniform 3D grid.
    Uses second-order MUSCL-Hancock scheme with Rusanov flux.
    """
    
    def __init__(
            self,
            simulation,
            total_mass=None,
            rho=None,
            vx=None,
            vy=None,
            vz=None,
            cs=1.0,
            cfl=0.4,
            rho_floor=1e-12,
            max_substeps=100,
            name="gas",
            gamma=5.0 / 3.0,
            tcool=None,
            e_floor=1e-10,
    ):
        """
        Initialize gas component.
        
        Parameters
        ----------
        simulation : Simulation_Class
            Parent simulation object
        total_mass : float, optional
            Total gas mass (used if rho not provided)
        rho, vx, vy, vz : array-like, optional
            Initial density and velocity fields
        cs : float
            Sound speed (for isothermal EOS) or initial sound speed
        cfl : float
            CFL safety factor (typically 0.3-0.5)
        rho_floor : float
            Density floor to prevent numerical issues
        max_substeps : int
            Maximum subcycles per drift call
        gamma : float
            Adiabatic index (5/3 for monoatomic gas)
        tcool : float, optional
            Cooling timescale (None = no cooling)
        e_floor : float
            Specific internal energy floor
        """
        self.simulation = simulation
        self.name = name

        N = simulation.N
        shape = (N, N, N) if simulation.dim == 3 else (N, N)

        # Grid spacing
        if hasattr(simulation, 'dx') and isinstance(simulation.dx, (list, tuple)):
            self.dx = float(min(simulation.dx))
        else:
            dx_list = [(b[1] - b[0]) / N for b in simulation.boundaries]
            self.dx = float(min(dx_list))

        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError(f"Invalid dx={self.dx}")

        self.cell_volume = self.simulation.dV
        self.cfl = float(cfl)
        self.rho_floor = float(rho_floor)
        self.max_substeps = int(max_substeps)
        self.gamma = float(gamma)
        self.cs = float(cs)
        self.tcool = tcool
        self.e_floor = float(e_floor)
        
        # Energy accounting
        self.E_radiated = 0.0

        # Initialize density
        if rho is None:
            if total_mass is None:
                raise ValueError("Provide either rho or total_mass")
            box_volume = self.cell_volume * (simulation.N ** simulation.dim)
            rho0 = float(total_mass) / box_volume
            self.rho = cp.full(shape, rho0, dtype=cp.float64)
        else:
            self.rho = cp.asarray(rho, dtype=cp.float64)
            if self.rho.shape != shape:
                raise ValueError(f"rho shape {self.rho.shape} != {shape}")

        # Initialize velocities
        self.vx = cp.zeros(shape, dtype=cp.float64) if vx is None else cp.asarray(vx, dtype=cp.float64)
        self.vy = cp.zeros(shape, dtype=cp.float64) if vy is None else cp.asarray(vy, dtype=cp.float64)
        self.vz = cp.zeros(shape, dtype=cp.float64) if vz is None else cp.asarray(vz, dtype=cp.float64)

        # Initialize total energy: E = rho*e_int + 0.5*rho*v^2
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        e_int_init = (self.cs**2) / (self.gamma * (self.gamma - 1.0))
        self.E = self.rho * e_int_init + 0.5 * self.rho * v2
        
        # Store reference density for isothermal pressure (if needed)
        self.rho_ref = float(cp.mean(self.rho).get())

    def deposit_to_grid(self):
        """Return density field for gravity calculations."""
        return self.rho

    def kinetic_energy(self):
        """Total kinetic energy of gas."""
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        return float(0.5 * cp.sum(self.rho * v2) * self.cell_volume)

    def internal_energy(self):
        """Total internal energy of gas."""
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        K = 0.5 * self.rho * v2
        eint = cp.maximum(self.E - K, self.rho * self.e_floor)
        return float(cp.sum(eint) * self.cell_volume)

    def drift(self, dt, potential_grid, first_step=False, last_step=False):
        """
        Evolve gas for time dt (or dt/2 on first/last step).
        
        Parameters
        ----------
        dt : float
            Full timestep
        potential_grid : array or tuple
            Either (ax, ay, az) acceleration grids or gravitational potential
        first_step, last_step : bool
            Whether this is first/last step (uses dt/2)
        """
        dt_window = dt / 2.0 if (first_step or last_step) else dt
        if dt_window <= 0:
            return

        # Get acceleration grids
        ax, ay, az = self._get_acceleration_grids(potential_grid)

        # CFL-based subcycling
        dt_sub = self._compute_cfl_timestep()
        nsub = max(1, min(int(np.ceil(dt_window / dt_sub)), self.max_substeps))
        dt_actual = dt_window / nsub

        # Subcycle loop
        for _ in range(nsub):
            # Strang splitting: gravity kick, hydro, gravity kick
            self._gravity_kick(ax, ay, az, 0.5 * dt_actual)
            self._hydro_step(dt_actual)
            self._apply_cooling(dt_actual)
            self._gravity_kick(ax, ay, az, 0.5 * dt_actual)

    def _compute_cfl_timestep(self):
        """Compute CFL-limited timestep."""
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        vmax = float(cp.sqrt(cp.max(v2)))
        
        # Compute maximum sound speed from pressure
        P = self._compute_pressure(self.rho, self.E)
        cs_local = cp.sqrt(self.gamma * P / cp.maximum(self.rho, self.rho_floor))
        cs_max = float(cp.max(cs_local))
        
        signal_speed = max(vmax, 1e-12) + cs_max
        return self.cfl * self.dx / signal_speed

    def _gravity_kick(self, ax, ay, az, dt):
        """Update velocities due to gravity."""
        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt

    def _hydro_step(self, dt):
        """Single hydro step using MUSCL-Hancock scheme."""
        dx = self.dx
        
        # 1) Compute primitive variables and pressure
        P = self._compute_pressure(self.rho, self.E)
        
        # 2) Compute gradients
        rho_dx, rho_dy, rho_dz = self._get_gradient(self.rho, dx)
        vx_dx, vx_dy, vx_dz = self._get_gradient(self.vx, dx)
        vy_dx, vy_dy, vy_dz = self._get_gradient(self.vy, dx)
        vz_dx, vz_dy, vz_dz = self._get_gradient(self.vz, dx)
        E_dx, E_dy, E_dz = self._get_gradient(self.E, dx)
        P_dx, P_dy, P_dz = self._get_gradient(P, dx)
        
        # 3) Apply slope limiter
        rho_dx, rho_dy, rho_dz = self._slope_limiter(self.rho, dx, rho_dx, rho_dy, rho_dz)
        vx_dx, vx_dy, vx_dz = self._slope_limiter(self.vx, dx, vx_dx, vx_dy, vx_dz)
        vy_dx, vy_dy, vy_dz = self._slope_limiter(self.vy, dx, vy_dx, vy_dy, vy_dz)
        vz_dx, vz_dy, vz_dz = self._slope_limiter(self.vz, dx, vz_dx, vz_dy, vz_dz)
        E_dx, E_dy, E_dz = self._slope_limiter(self.E, dx, E_dx, E_dy, E_dz)
        P_dx, P_dy, P_dz = self._slope_limiter(P, dx, P_dx, P_dy, P_dz)
        
        # 4) Extrapolate to faces
        rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = self._extrap_to_face(
            self.rho, rho_dx, rho_dy, rho_dz, dx
        )
        vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = self._extrap_to_face(
            self.vx, vx_dx, vx_dy, vx_dz, dx
        )
        vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = self._extrap_to_face(
            self.vy, vy_dx, vy_dy, vy_dz, dx
        )
        vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = self._extrap_to_face(
            self.vz, vz_dx, vz_dy, vz_dz, dx
        )
        E_XL, E_XR, E_YL, E_YR, E_ZL, E_ZR = self._extrap_to_face(
            self.E, E_dx, E_dy, E_dz, dx
        )
        P_XL, P_XR, P_YL, P_YR, P_ZL, P_ZR = self._extrap_to_face(
            P, P_dx, P_dy, P_dz, dx
        )
        
        # Apply floor
        rho_XL = cp.maximum(rho_XL, self.rho_floor)
        rho_XR = cp.maximum(rho_XR, self.rho_floor)
        rho_YL = cp.maximum(rho_YL, self.rho_floor)
        rho_YR = cp.maximum(rho_YR, self.rho_floor)
        rho_ZL = cp.maximum(rho_ZL, self.rho_floor)
        rho_ZR = cp.maximum(rho_ZR, self.rho_floor)
        
        # 5) Compute fluxes at faces
        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X, flux_E_X = self._get_flux(
            rho_XL, vx_XL, vy_XL, vz_XL, E_XL, P_XL,
            rho_XR, vx_XR, vy_XR, vz_XR, E_XR, P_XR
        )
        flux_Mass_Y, flux_Momy_Y, flux_Momz_Y, flux_Momx_Y, flux_E_Y = self._get_flux(
            rho_YL, vy_YL, vz_YL, vx_YL, E_YL, P_YL,
            rho_YR, vy_YR, vz_YR, vx_YR, E_YR, P_YR
        )
        flux_Mass_Z, flux_Momz_Z, flux_Momx_Z, flux_Momy_Z, flux_E_Z = self._get_flux(
            rho_ZL, vz_ZL, vx_ZL, vy_ZL, E_ZL, P_ZL,
            rho_ZR, vz_ZR, vx_ZR, vy_ZR, E_ZR, P_ZR
        )
        
        # 6) Convert to conserved variables
        vol = dx**3
        Mass = self.rho * vol
        Momx = self.rho * self.vx * vol
        Momy = self.rho * self.vy * vol
        Momz = self.rho * self.vz * vol
        Etot = self.E * vol
        
        # 7) Update conserved variables
        Mass = self._apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
        Momx = self._apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
        Momy = self._apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z, dx, dt)
        Momz = self._apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z, dx, dt)
        Etot = self._apply_fluxes(Etot, flux_E_X, flux_E_Y, flux_E_Z, dx, dt)
        
        # 8) Convert back to primitive variables
        self.rho = Mass / vol
        self.rho = cp.maximum(self.rho, self.rho_floor)
        
        self.vx = Momx / (self.rho * vol)
        self.vy = Momy / (self.rho * vol)
        self.vz = Momz / (self.rho * vol)
        self.E = Etot / vol
        
        # Apply energy floor
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        eint = self.E - 0.5 * self.rho * v2
        eint = cp.maximum(eint, self.rho * self.e_floor)
        self.E = 0.5 * self.rho * v2 + eint

    def _compute_pressure(self, rho, E):
        """Compute pressure from total energy: P = (γ-1)(E - ½ρv²)."""
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        eint = E - 0.5 * rho * v2
        eint = cp.maximum(eint, rho * self.e_floor)
        return (self.gamma - 1.0) * eint

    def _get_flux(self, rho_L, vx_L, vy_L, vz_L, E_L, P_L,
                  rho_R, vx_R, vy_R, vz_R, E_R, P_R):
        """
        Compute Rusanov (local Lax-Friedrichs) fluxes.
        Returns: (flux_mass, flux_momx, flux_momy, flux_momz, flux_energy)
        """
        # Left fluxes
        FL_M = rho_L * vx_L
        FL_Px = rho_L * vx_L**2 + P_L
        FL_Py = rho_L * vx_L * vy_L
        FL_Pz = rho_L * vx_L * vz_L
        FL_E = (E_L + P_L) * vx_L
        
        # Right fluxes
        FR_M = rho_R * vx_R
        FR_Px = rho_R * vx_R**2 + P_R
        FR_Py = rho_R * vx_R * vy_R
        FR_Pz = rho_R * vx_R * vz_R
        FR_E = (E_R + P_R) * vx_R
        
        # Wave speeds
        cs_L = cp.sqrt(self.gamma * P_L / cp.maximum(rho_L, self.rho_floor))
        cs_R = cp.sqrt(self.gamma * P_R / cp.maximum(rho_R, self.rho_floor))
        C = cp.maximum(cp.abs(vx_L) + cs_L, cp.abs(vx_R) + cs_R)
        
        # Rusanov flux
        flux_M = 0.5 * (FL_M + FR_M - C * (rho_R - rho_L))
        flux_Px = 0.5 * (FL_Px + FR_Px - C * (rho_R * vx_R - rho_L * vx_L))
        flux_Py = 0.5 * (FL_Py + FR_Py - C * (rho_R * vy_R - rho_L * vy_L))
        flux_Pz = 0.5 * (FL_Pz + FR_Pz - C * (rho_R * vz_R - rho_L * vz_L))
        flux_E = 0.5 * (FL_E + FR_E - C * (E_R - E_L))
        
        return flux_M, flux_Px, flux_Py, flux_Pz, flux_E

    def _apply_cooling(self, dt):
        """Apply cooling with relaxation to floor."""
        if self.tcool is None or self.tcool <= 0:
            return
        
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        K = 0.5 * self.rho * v2
        eint = self.E - K
        eint_floor = self.rho * self.e_floor
        
        # Exponential relaxation
        fac = cp.exp(-dt / self.tcool)
        eint_new = eint_floor + (eint - eint_floor) * fac
        
        # Track radiated energy
        dE = cp.sum(cp.maximum(eint - eint_new, 0.0)) * self.cell_volume
        self.E_radiated += float(dE)
        
        self.E = K + eint_new

    def _get_acceleration_grids(self, potential_grid):
        """Extract or compute acceleration grids from potential."""
        if isinstance(potential_grid, (tuple, list)) and len(potential_grid) == 3:
            return tuple(cp.asarray(a, dtype=cp.float64) for a in potential_grid)
        
        # Compute from potential via FFT
        Phi = cp.asarray(potential_grid, dtype=cp.float64)
        Phi_k = cp.fft.fftn(Phi.astype(cp.complex128))
        kx, ky, kz = self.simulation.k_space
        
        ax = cp.fft.ifftn(-1j * kx * Phi_k).real.astype(cp.float64)
        ay = cp.fft.ifftn(-1j * ky * Phi_k).real.astype(cp.float64)
        az = cp.fft.ifftn(-1j * kz * Phi_k).real.astype(cp.float64)
        return ax, ay, az

    @staticmethod
    def _get_gradient(f, dx):
        """Central difference gradients."""
        f_dx = (cp.roll(f, -1, axis=0) - cp.roll(f, 1, axis=0)) / (2.0 * dx)
        f_dy = (cp.roll(f, -1, axis=1) - cp.roll(f, 1, axis=1)) / (2.0 * dx)
        f_dz = (cp.roll(f, -1, axis=2) - cp.roll(f, 1, axis=2)) / (2.0 * dx)
        return f_dx, f_dy, f_dz

    @staticmethod
    def _extrap_to_face(f, f_dx, f_dy, f_dz, dx):
        """Linear extrapolation to cell faces."""
        f_XL = f + 0.5 * f_dx * dx
        f_XR = cp.roll(f - 0.5 * f_dx * dx, -1, axis=0)
        
        f_YL = f + 0.5 * f_dy * dx
        f_YR = cp.roll(f - 0.5 * f_dy * dx, -1, axis=1)
        
        f_ZL = f + 0.5 * f_dz * dx
        f_ZR = cp.roll(f - 0.5 * f_dz * dx, -1, axis=2)
        
        return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR

    @staticmethod
    @staticmethod
    def _slope_limiter(f, dx, f_dx, f_dy, f_dz):
        """Minmod slope limiter to prevent oscillations."""

        def minmod_1d(f, df, axis):
            df_fwd = (cp.roll(f, -1, axis=axis) - f) / dx
            df_bwd = (f - cp.roll(f, 1, axis=axis)) / dx

            same_sign = (cp.sign(df) == cp.sign(df_fwd)) & (cp.sign(df) == cp.sign(df_bwd))
            min_abs_val = cp.minimum(cp.abs(df), cp.minimum(cp.abs(df_fwd), cp.abs(df_bwd)))

            limited = cp.where(
                same_sign,
                cp.sign(df) * min_abs_val,
                0.0
            )

            return limited

        return (minmod_1d(f, f_dx, 0), minmod_1d(f, f_dy, 1), minmod_1d(f, f_dz, 2))

    @staticmethod
    def _apply_fluxes(F, flux_X, flux_Y, flux_Z, dx, dt):
        """Apply conservative flux update."""
        fac = dt / dx
        F -= fac * flux_X
        F += fac * cp.roll(flux_X, 1, axis=0)
        F -= fac * flux_Y
        F += fac * cp.roll(flux_Y, 1, axis=1)
        F -= fac * flux_Z
        F += fac * cp.roll(flux_Z, 1, axis=2)
        return F

    def total_mass(self):
        return float(cp.sum(self.rho) * self.cell_volume)

    def total_momentum(self):
        Px = cp.sum(self.rho * self.vx) * self.cell_volume
        Py = cp.sum(self.rho * self.vy) * self.cell_volume
        Pz = cp.sum(self.rho * self.vz) * self.cell_volume
        return float(Px), float(Py), float(Pz)

    def total_energy_components(self):
        # jen pokud máš adiabatic/E (po 1A změnách)
        v2 = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz
        K = float(0.5 * cp.sum(self.rho * v2) * self.cell_volume)
        U = self.internal_energy() if hasattr(self, "internal_energy") else 0.0
        return K, U
