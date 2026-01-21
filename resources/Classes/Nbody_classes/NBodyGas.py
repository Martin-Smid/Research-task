import cupy as cp
import numpy as np


class NBodyGas:
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
            max_substeps=50,
            name="gas",
    ):
        self.simulation = simulation
        self.name = name

        N = simulation.N
        shape = (N, N, N) if simulation.dim == 3 else (N, N)

        # Ensure dx is properly extracted from simulation
        if hasattr(simulation, 'dx') and isinstance(simulation.dx, (list, tuple)):
            self.dx = float(min(simulation.dx))
        else:
            # Fallback: calculate from boundaries
            dx_list = [(b[1] - b[0]) / N for b in simulation.boundaries]
            self.dx = float(min(dx_list))

        # Validate dx
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError(f"Invalid dx value: {self.dx}. Check simulation grid setup.")

        self.cell_volume = self.simulation.dV

        self.cs = float(cs)
        self.cfl = float(cfl)
        self.rho_floor = float(rho_floor)
        self.max_substeps = int(max_substeps)

        # Allocate/initialize state on GPU
        if rho is None:
            if total_mass is None:
                raise ValueError("Provide either rho grid or total_mass to initialize gas.")
            box_volume = self.cell_volume * (simulation.N ** simulation.dim)
            rho0 = float(total_mass) / box_volume
            self.rho = cp.full(shape, rho0, dtype=cp.float64)
        else:
            self.rho = cp.asarray(rho, dtype=cp.float64)
            if self.rho.shape != shape:
                raise ValueError(f"rho has shape {self.rho.shape}, expected {shape}.")

        # velocities default to 0 - ensure they're properly initialized
        self.vx = cp.zeros(shape, dtype=cp.float64) if vx is None else cp.asarray(vx, dtype=cp.float64)
        self.vy = cp.zeros(shape, dtype=cp.float64) if vy is None else cp.asarray(vy, dtype=cp.float64)
        self.vz = cp.zeros(shape, dtype=cp.float64) if vz is None else cp.asarray(vz, dtype=cp.float64)

        if self.vx.shape != shape or self.vy.shape != shape or self.vz.shape != shape:
            raise ValueError("Velocity components must match rho grid shape.")

        # Safety tracking
        self.nan_count = 0
        self.max_nan_resets = 10

    def deposit_to_grid(self):
        # Used by Evolution._compute_total_density()
        return self.rho

    def kinetic_energy(self):
        v2 = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz
        K = 0.5 * cp.sum(self.rho * v2) * self.cell_volume
        return float(K)

    def drift(self, dt, potential_grid, first_step=False, last_step=False):
        """
        Evolve gas by dt (or dt/2 if first/last step) under:
          - isothermal hydro flux update
          - external gravity from potential_grid
        """
        # match particle convention: half-step window on first/last
        dt_window = dt / 2.0 if (first_step or last_step) else dt
        if dt_window <= 0:
            return

        # Check for NaN before starting
        if not self._check_state_validity():
            print(f"ERROR: Invalid state detected before drift. Aborting evolution.")
            return

        # Prepare gravity acceleration grids ONCE for this drift window
        ax, ay, az = self._get_acceleration_grids(potential_grid)

        # CFL subcycling with safety checks
        v2 = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz
        vmax_sq = float(cp.max(v2))

        # Check for NaN or invalid values
        if not np.isfinite(vmax_sq) or vmax_sq < 0:
            print(f"Warning: Invalid velocity detected (vmax^2={vmax_sq}).")
            self._attempt_recovery()
            return

        vmax = float(cp.sqrt(vmax_sq))
        vmax = max(vmax, 1e-12)

        # CFL timestep
        dt_cfl = self.cfl * self.dx / (vmax + self.cs)

        # Additional safety check for dt_cfl
        if not np.isfinite(dt_cfl) or dt_cfl <= 0:
            print(f"Warning: Invalid dt_cfl={dt_cfl}. Using fallback timestep.")
            dt_cfl = self.cfl * self.dx / self.cs

        nsub = int(np.ceil(dt_window / dt_cfl))
        nsub = max(1, min(nsub, self.max_substeps))
        sub_dt = dt_window / nsub

        if nsub == self.max_substeps and dt_window / nsub > dt_cfl:
            print(f"CFL violated: sub_dt={dt_window / nsub:.3e} > dt_cfl={dt_cfl:.3e}")

        for isub in range(nsub):
            # Strang split: half gravity, hydro, half gravity
            self._gravity_kick(ax, ay, az, 0.5 * sub_dt)

            self.rho, self.vx, self.vy, self.vz = self._hydro_fluxes(
                self.rho, self.vx, self.vy, self.vz, sub_dt, self.dx, self.cs
            )

            self._gravity_kick(ax, ay, az, 0.5 * sub_dt)

            # Check validity after each substep
            if not self._check_state_validity():
                print(f"ERROR: Invalid state at substep {isub + 1}/{nsub}")
                self._attempt_recovery()
                break

    def _check_state_validity(self):
        """Check if current state contains NaN or Inf values."""
        checks = [
            cp.isfinite(self.rho).all(),
            cp.isfinite(self.vx).all(),
            cp.isfinite(self.vy).all(),
            cp.isfinite(self.vz).all(),
            (self.rho >= 0).all()
        ]
        return all(checks)

    def _attempt_recovery(self):
        """Try to recover from invalid state."""
        self.nan_count += 1

        if self.nan_count > self.max_nan_resets:
            raise RuntimeError(
                f"Gas simulation failed after {self.nan_count} recovery attempts. "
                "Check initial conditions, CFL number, or reduce timestep."
            )

        print(f"  Attempting recovery (attempt {self.nan_count}/{self.max_nan_resets})")

        # Replace invalid values with safe defaults
        self.rho = cp.where(cp.isfinite(self.rho), self.rho, self.rho_floor)
        self.rho_ref = float(cp.mean(self.rho).get())
        self.vx = cp.where(cp.isfinite(self.vx), self.vx, 0.0)
        self.vy = cp.where(cp.isfinite(self.vy), self.vy, 0.0)
        self.vz = cp.where(cp.isfinite(self.vz), self.vz, 0.0)

        # Apply floor
        self.rho = cp.maximum(self.rho, self.rho_floor)


    def _gravity_kick(self, ax, ay, az, dt):
        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt

    def _get_acceleration_grids(self, potential_grid):
        if isinstance(potential_grid, (tuple, list)) and len(potential_grid) == 3:
            ax, ay, az = potential_grid
            return (cp.asarray(ax, dtype=cp.float64),
                    cp.asarray(ay, dtype=cp.float64),
                    cp.asarray(az, dtype=cp.float64))

        # Otherwise compute from Phi using simulation.k_space
        Phi = cp.asarray(potential_grid, dtype=cp.float64)
        Phi_k = cp.fft.fftn(Phi.astype(cp.complex128))

        kx, ky, kz = self.simulation.k_space
        ax = cp.fft.ifftn((-1j) * kx * Phi_k).real.astype(cp.float64)
        ay = cp.fft.ifftn((-1j) * ky * Phi_k).real.astype(cp.float64)
        az = cp.fft.ifftn((-1j) * kz * Phi_k).real.astype(cp.float64)
        return ax, ay, az

    @staticmethod
    def _get_conserved(rho, vx, vy, vz, vol):
        Mass = rho * vol
        Momx = rho * vx * vol
        Momy = rho * vy * vol
        Momz = rho * vz * vol
        return Mass, Momx, Momy, Momz

    @staticmethod
    def _get_primitive(Mass, Momx, Momy, Momz, vol, rho_floor=1e-12):
        rho = Mass / vol
        rho = cp.maximum(rho, rho_floor)

        # Safe division: if rho is at floor, velocity should be zero
        denom = cp.maximum(rho * vol, rho_floor * vol)
        vx = Momx / denom
        vy = Momy / denom
        vz = Momz / denom

        return rho, vx, vy, vz

    @staticmethod
    def _get_gradient(f, dx):
        f_dx = (cp.roll(f, -1, axis=0) - cp.roll(f, 1, axis=0)) / (2.0 * dx)
        f_dy = (cp.roll(f, -1, axis=1) - cp.roll(f, 1, axis=1)) / (2.0 * dx)
        f_dz = (cp.roll(f, -1, axis=2) - cp.roll(f, 1, axis=2)) / (2.0 * dx)
        return f_dx, f_dy, f_dz

    @staticmethod
    def _extrap_to_face(f, f_dx, f_dy, f_dz, dx):
        # Linear extrapolation to cell faces
        f_XL = f + 0.5 * f_dx * dx
        f_XR = f - 0.5 * f_dx * dx
        f_XR = cp.roll(f_XR, -1, axis=0)

        f_YL = f + 0.5 * f_dy * dx
        f_YR = f - 0.5 * f_dy * dx
        f_YR = cp.roll(f_YR, -1, axis=1)

        f_ZL = f + 0.5 * f_dz * dx
        f_ZR = f - 0.5 * f_dz * dx
        f_ZR = cp.roll(f_ZR, -1, axis=2)

        return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR

    @staticmethod
    def _get_flux(rho_L, vx_L, vy_L, vz_L, rho_R, vx_R, vy_R, vz_R, cs):
        # Average states with floor
        rho_star = 0.5 * (rho_L + rho_R)
        rho_star = cp.maximum(rho_star, 1e-12)

        momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
        momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
        momz_star = 0.5 * (rho_L * vz_L + rho_R * vz_R)

        P_star = rho_star * cs * cs

        # Safe flux calculation with floor on denominator
        rho_safe = cp.maximum(rho_star, 1e-12)

        flux_Mass = momx_star
        flux_Momx = momx_star ** 2 / rho_safe + P_star
        flux_Momy = momx_star * momy_star / rho_safe
        flux_Momz = momx_star * momz_star / rho_safe

        # HLL wave speeds
        C_L = cs + cp.abs(vx_L)
        C_R = cs + cp.abs(vx_R)
        C = cp.maximum(C_L, C_R)

        # Diffusive corrections
        flux_Mass -= C * 0.5 * (rho_R - rho_L)
        flux_Momx -= C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
        flux_Momy -= C * 0.5 * (rho_R * vy_R - rho_L * vy_L)
        flux_Momz -= C * 0.5 * (rho_R * vz_R - rho_L * vz_L)

        return flux_Mass, flux_Momx, flux_Momy, flux_Momz

    @staticmethod
    def _apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
        # For 3D: flux update is dt * (flux_in - flux_out) / dx
        # Factor should be dt/dx for proper conservative update
        fac = dt / dx

        F += -fac * flux_F_X
        F += fac * cp.roll(flux_F_X, 1, axis=0)

        F += -fac * flux_F_Y
        F += fac * cp.roll(flux_F_Y, 1, axis=1)

        F += -fac * flux_F_Z
        F += fac * cp.roll(flux_F_Z, 1, axis=2)

        return F

    @staticmethod
    def _slope_limiter(f, dx, f_dx, f_dy, f_dz):
        """
        Apply minmod slope limiter to prevent oscillations.
        Fixed version: only apply limiter once per direction.
        """
        eps = 1.0e-12

        def minmod_limit_1d(f, df, axis):
            """Apply minmod limiter in one direction."""
            # Forward and backward differences
            df_forward = (cp.roll(f, -1, axis=axis) - f) / dx
            df_backward = (f - cp.roll(f, 1, axis=axis)) / dx

            # Minmod limiter: take smallest magnitude if same sign, else zero
            sign_check = cp.sign(df) * cp.sign(df_forward) * cp.sign(df_backward)

            limited = cp.where(
                sign_check > 0,
                cp.sign(df) * cp.minimum(
                    cp.abs(df),
                    cp.minimum(cp.abs(df_forward), cp.abs(df_backward))
                ),
                0.0
            )

            return limited

        # Apply limiter independently in each direction
        f_dx_limited = minmod_limit_1d(f, f_dx, axis=0)
        f_dy_limited = minmod_limit_1d(f, f_dy, axis=1)
        f_dz_limited = minmod_limit_1d(f, f_dz, axis=2)

        return f_dx_limited, f_dy_limited, f_dz_limited

    def _hydro_fluxes(self, rho, vx, vy, vz, dt, dx, cs):
        # Compute gradients
        rho_dx, rho_dy, rho_dz = self._get_gradient(rho, dx)
        vx_dx, vx_dy, vx_dz = self._get_gradient(vx, dx)
        vy_dx, vy_dy, vy_dz = self._get_gradient(vy, dx)
        vz_dx, vz_dy, vz_dz = self._get_gradient(vz, dx)

        # Apply slope limiters
        rho_dx, rho_dy, rho_dz = self._slope_limiter(rho, dx, rho_dx, rho_dy, rho_dz)
        vx_dx, vx_dy, vx_dz = self._slope_limiter(vx, dx, vx_dx, vx_dy, vx_dz)
        vy_dx, vy_dy, vy_dz = self._slope_limiter(vy, dx, vy_dx, vy_dy, vy_dz)
        vz_dx, vz_dy, vz_dz = self._slope_limiter(vz, dx, vz_dx, vz_dy, vz_dz)

        # Extrapolate to faces
        rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = self._extrap_to_face(
            rho, rho_dx, rho_dy, rho_dz, dx
        )
        vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = self._extrap_to_face(
            vx, vx_dx, vx_dy, vx_dz, dx
        )
        vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = self._extrap_to_face(
            vy, vy_dx, vy_dy, vy_dz, dx
        )
        vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = self._extrap_to_face(
            vz, vz_dx, vz_dy, vz_dz, dx
        )

        # Apply floor to extrapolated densities
        rho_XL = cp.maximum(rho_XL, self.rho_floor)
        rho_XR = cp.maximum(rho_XR, self.rho_floor)
        rho_YL = cp.maximum(rho_YL, self.rho_floor)
        rho_YR = cp.maximum(rho_YR, self.rho_floor)
        rho_ZL = cp.maximum(rho_ZL, self.rho_floor)
        rho_ZR = cp.maximum(rho_ZR, self.rho_floor)

        # Compute fluxes
        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X = self._get_flux(
            rho_XL, vx_XL, vy_XL, vz_XL, rho_XR, vx_XR, vy_XR, vz_XR, cs
        )
        flux_Mass_Y, flux_Momy_Y, flux_Momz_Y, flux_Momx_Y = self._get_flux(
            rho_YL, vy_YL, vz_YL, vx_YL, rho_YR, vy_YR, vz_YR, vx_YR, cs
        )
        flux_Mass_Z, flux_Momz_Z, flux_Momx_Z, flux_Momy_Z = self._get_flux(
            rho_ZL, vz_ZL, vx_ZL, vy_ZL, rho_ZR, vz_ZR, vx_ZR, vy_ZR, cs
        )

        # Get conserved variables
        Mass, Momx, Momy, Momz = self._get_conserved(rho, vx, vy, vz, dx ** 3)

        # Apply flux updates
        Mass = self._apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
        Momx = self._apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
        Momy = self._apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z, dx, dt)
        Momz = self._apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z, dx, dt)

        # Convert back to primitive variables
        rho, vx, vy, vz = self._get_primitive(
            Mass, Momx, Momy, Momz, dx ** 3, rho_floor=self.rho_floor
        )

        # Final floor enforcement
        rho = cp.maximum(rho, self.rho_floor)

        return rho, vx, vy, vz