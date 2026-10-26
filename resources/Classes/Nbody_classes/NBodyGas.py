import cupy as cp
import numpy as np


def free_fall_estimate(mass, radius, G=None, steps_per_t_ff=100, print_summary=True):
    """What to expect from a cold uniform gas sphere of the given mass [Msun] and radius [kpc].

    Returns a dict with the free-fall (collapse) time t_ff = sqrt(3 pi / (32 G rho0)), rho0 = 3 M / (4 pi R^3),
    the recommended time step h = t_ff / steps_per_t_ff, and a total_time covering collapse and rebound
    (1.3 t_ff). Times are in Gyr for G in kpc^3 / (Msun Gyr^2) (the default). The collapse reaches its
    densest point at ~1.05 t_ff in simulations. A step much larger than t_ff / 100 makes the gas need
    many CFL substeps, and beyond max_substeps the run stops with a FloatingPointError."""
    if G is None:
        from astropy import constants
        G = constants.G.to("kpc3/(Msun Gyr2)").value
    rho0 = 3.0 * mass / (4.0 * np.pi * radius ** 3)
    t_ff = float(np.sqrt(3.0 * np.pi / (32.0 * G * rho0)))
    estimate = {"rho0": rho0, "t_ff": t_ff, "h": t_ff / steps_per_t_ff, "total_time": 1.3 * t_ff,
                "steps": int(round(1.3 * steps_per_t_ff))}
    if print_summary:
        print(f"M = {mass:.3g} Msun, R = {radius:g} kpc: rho0 = {rho0:.3g} Msun/kpc^3, "
              f"t_ff = {t_ff:.5f} Gyr ({t_ff * 1e3:.2f} Myr)\n"
              f"  recommended h = {estimate['h']:.3e} Gyr, total_time = {estimate['total_time']:.5f} Gyr "
              f"({estimate['steps']} steps), densest point at ~{1.05 * t_ff:.5f} Gyr")
    return estimate


class NBodyGas:
    """
    Compressible Euler gas on a uniform periodic grid (CuPy).

    Goal
    ----
    Make initialization and hydro evolution independent of dimension ``dim`` in the
    same spirit as the wave-function classes (fields are shaped as ``(N,)*dim``).

    Implementation
    --------------
    - Conserved variables: rho, mom[i]=rho*v[i], E (total energy density)
    - Pressure (adiabatic): P = (gamma-1) * (E - 0.5*rho*|v|^2)
    - N-D update: dimensionally split MUSCL (piecewise-linear) reconstruction +
      Rusanov (local Lax-Friedrichs) flux, periodic boundaries via cp.roll.

    Compatibility
    -------------
    The rest of the project expects attributes ``rho, vx, vy, vz, E`` and gravity
    forces as a 3-tuple (Fx,Fy,Fz). We keep vx/vy/vz for backward compatibility:
    - For dim < 3, missing components are kept as zero arrays.
    - For dim > 3, extra velocity components exist in ``self.vel`` but only the
      first three are exposed as vx/vy/vz (and only the first three can be kicked
      by Evolution's gravity forces unless you provide a longer force tuple).
    """

    # Gravitational potential on the grid; Evolution sets it before every drift. When set,
    # the gravity work on E uses the interface mass flux (see _sweep_axis); None = old behaviour.
    phi_total = None
    _face_work = None
    # Optional callable d_rho -> (forces, phi) giving the gravity after the gas density changed by
    # d_rho; Evolution sets it. When present, drift uses a second-order kick-drift-kick (see drift).
    potential_update = None
    E_guard_injected = 0.0  # cumulative energy added by the internal-energy floor guard

    def __init__(
        self,
        simulation,
        total_mass=None,
        rho=None,
        vx=None,
        vy=None,
        vz=None,
        v=None,  # optional: list/tuple of length dim with velocity component arrays
        cs=1.0,
        cfl=0.4,
        rho_floor=1e-12,
        max_substeps=100,
        name="gas",
        gamma=5.0 / 3.0,
        tcool=None,
        e_floor=1e-10,
    ):
        self.simulation = simulation
        self.name = name
        self.dim = int(simulation.dim)

        N = int(simulation.N)
        shape = (N,) * self.dim

        # Grid spacing (per-axis) and conservative min(dx) for CFL
        if hasattr(simulation, "dx") and isinstance(simulation.dx, (list, tuple)):
            self.dx_list = [float(d) for d in simulation.dx]
        else:
            self.dx_list = [float((b[1] - b[0]) / N) for b in simulation.boundaries]

        if (len(self.dx_list) != self.dim) or any((not np.isfinite(d) or d <= 0) for d in self.dx_list):
            raise ValueError(f"Invalid dx_list={self.dx_list}")

        self.dx = float(min(self.dx_list))  # conservative dx for CFL

        self.cell_volume = float(self.simulation.dV)
        self.cfl = float(cfl)
        self.rho_floor = float(rho_floor)
        self.max_substeps = int(max_substeps)
        self.gamma = float(gamma)
        self.cs = float(cs)
        self.tcool = tcool
        self.e_floor = float(e_floor)

        # Energy accounting
        self.E_radiated = 0.0

        # -------------------------
        # Initialize density field
        # -------------------------
        if rho is None:
            if total_mass is None:
                raise ValueError("Provide either rho or total_mass")
            box_volume = self.cell_volume * (N ** self.dim)
            rho0 = float(total_mass) / box_volume
            self.rho = cp.full(shape, rho0, dtype=cp.float64)
        else:
            self.rho = cp.asarray(rho, dtype=cp.float64)
            if self.rho.shape != shape:
                raise ValueError(f"rho shape {self.rho.shape} != {shape}")

        # -------------------------
        # Initialize velocities
        # -------------------------
        # Default named components (backward compatibility)
        self.vx = cp.zeros(shape, dtype=cp.float64) if vx is None else cp.asarray(vx, dtype=cp.float64)
        self.vy = cp.zeros(shape, dtype=cp.float64) if vy is None else cp.asarray(vy, dtype=cp.float64)
        self.vz = cp.zeros(shape, dtype=cp.float64) if vz is None else cp.asarray(vz, dtype=cp.float64)

        for arr_name in ("vx", "vy", "vz"):
            arr = getattr(self, arr_name)
            if arr.shape != shape:
                raise ValueError(f"{arr_name} shape {arr.shape} != {shape}")

        # Dimension-aware velocity container
        base_vel = [self.vx, self.vy, self.vz]
        if self.dim > 3:
            base_vel.extend([cp.zeros(shape, dtype=cp.float64) for _ in range(self.dim - 3)])
        self.vel = base_vel[: self.dim]

        # Optional user-supplied full velocity list
        if v is not None:
            if not isinstance(v, (list, tuple)) or len(v) != self.dim:
                raise ValueError(f"`v` must be a list/tuple of length dim={self.dim}")
            self.vel = [cp.asarray(v_i, dtype=cp.float64) for v_i in v]
            for i, v_i in enumerate(self.vel):
                if v_i.shape != shape:
                    raise ValueError(f"v[{i}] shape {v_i.shape} != {shape}")
            # sync named
            self.vx = self.vel[0]
            self.vy = self.vel[1] if self.dim >= 2 else cp.zeros(shape, dtype=cp.float64)
            self.vz = self.vel[2] if self.dim >= 3 else cp.zeros(shape, dtype=cp.float64)

        # -------------------------
        # Initialize total energy density
        # -------------------------
        v2 = cp.zeros_like(self.rho)
        for vv in self.vel:
            v2 = v2 + vv * vv

        # If you want isothermal later, keep cs as a reference; for now use adiabatic E init
        e_int_init = (self.cs ** 2) / (self.gamma * (self.gamma - 1.0))
        self.E = self.rho * e_int_init + 0.5 * self.rho * v2

        # Reference density (optional)
        self.rho_ref = float(cp.mean(self.rho).get())

    # -------------------------------------------------------------------------
    # Public API used by Evolution / Simulation
    # -------------------------------------------------------------------------
    def deposit_to_grid(self):
        return self.rho

    def kinetic_energy(self):
        v2 = cp.zeros_like(self.rho)
        for vv in self.vel:
            v2 = v2 + vv * vv
        return float(0.5 * cp.sum(self.rho * v2) * self.cell_volume)

    def internal_energy(self):
        v2 = cp.zeros_like(self.rho)
        for vv in self.vel:
            v2 = v2 + vv * vv
        K = 0.5 * self.rho * v2
        eint = cp.maximum(self.E - K, self.rho * self.e_floor)
        return float(cp.sum(eint) * self.cell_volume)

    def drift(self, dt, potential_grid, first_step=False, last_step=False):
        """
        Evolve gas for time dt with subcycling.

        Parameters
        ----------
        dt : float
        potential_grid : tuple/list or array
            Usually (Fx,Fy,Fz) from Evolution. If an array is provided, we interpret
            it as a potential Phi and compute forces by centered differences.
        """
        # TODO: check this - drift already advances a complete Strang step.
        dt_window = dt
        if not np.isfinite(dt_window) or dt_window < 0:
            raise ValueError("Gas drift requires a finite, non-negative time step.")
        if dt_window == 0:
            return

        # Get acceleration grids (Fx,Fy,Fz) as provided by Evolution (3-tuple)
        ax, ay, az = self._get_acceleration_grids(potential_grid)

        if self.potential_update is not None and self.phi_total is not None:
            # Kick(dt/2, g_n) - hydro(dt) - Kick(dt/2, g_{n+1}): g_{n+1} comes from the density after
            # the hydro, so the potential is no longer frozen over the window (second order in time).
            rho_old, phi_old = self.rho, self.phi_total
            kick_work = self._gravity_kick(ax, ay, az, 0.5 * dt_window)
            face_work = 0.0
            for dt_actual in self._substeps(dt_window):
                self._hydro_step(dt_actual)
                face_work = face_work + self._face_work
                self._apply_cooling(dt_actual)
            d_rho = self.rho - rho_old
            (ax, ay, az), phi_new = self.potential_update(d_rho)
            kick_work = kick_work + self._gravity_kick(ax, ay, az, 0.5 * dt_window)
            # Interface-flux work used phi_n; the trapezoid correction -1/2 d_rho d_phi completes
            # dW = integral of (phi_n + phi_{n+1})/2 * d_rho.
            self._add_gravity_energy_correction(face_work - 0.5 * d_rho * (phi_new - phi_old) - kick_work)
            return

        for dt_actual in self._substeps(dt_window):
            # Strang: gravity kick / hydro / (optional) cooling / gravity kick
            kick_work = self._gravity_kick(ax, ay, az, 0.5 * dt_actual)
            self._hydro_step(dt_actual)
            self._apply_cooling(dt_actual)
            kick_work = kick_work + self._gravity_kick(ax, ay, az, 0.5 * dt_actual)
            if self._face_work is not None:
                # Replace the cell-centred work rho*v.a by the interface-flux work, so that
                # E_gas + W changes only by the (small) time variation of Phi.
                self._add_gravity_energy_correction(self._face_work - kick_work)

    def _substeps(self, dt_window):
        """Yield CFL-limited substeps that add up to dt_window. The CFL step is re-evaluated from the
        current state before every substep (a flow that accelerates inside the window, e.g. after a
        collapse, would otherwise outrun a step computed once at the start); the remaining time is
        split evenly into as many steps as the current estimate needs."""
        remaining = dt_window
        done = 0
        while remaining > 0:
            dt_cfl = self._compute_cfl_timestep()
            if not np.isfinite(dt_cfl) or dt_cfl <= 0:
                raise FloatingPointError("[Gas] non-finite CFL time step: the gas state contains NaN/inf.")
            need = max(1, int(np.ceil(remaining / dt_cfl)))
            if done + need > self.max_substeps:
                raise FloatingPointError(
                    f"[Gas] CFL requires {done + need} substeps but max_substeps={self.max_substeps}. "
                    f"Reduce global h or increase max_substeps."
                )
            dt_actual = remaining / need
            remaining -= dt_actual
            done += 1
            yield dt_actual

    def _add_gravity_energy_correction(self, correction):
        """E += correction, but never let the internal energy E - K fall below its floor.
        Where the correction would, only the part that keeps E - K at the floor is applied; the
        energy this adds relative to exact conservation is summed in E_guard_injected."""
        v2 = cp.zeros_like(self.rho)
        for vv in self.vel:
            v2 = v2 + vv * vv
        eint = self.E - 0.5 * self.rho * v2
        applied = cp.maximum(correction, self.rho * self.e_floor - eint)
        self.E += applied
        self.E_guard_injected += float(cp.sum(applied - correction)) * self.cell_volume

    # -------------------------------------------------------------------------
    # Core numerics
    # -------------------------------------------------------------------------
    def _compute_cfl_timestep(self):
        v2 = cp.zeros_like(self.rho)
        for vv in self.vel:
            v2 = v2 + vv * vv
        vmax = float(cp.sqrt(cp.max(v2)))

        # maximum sound speed from cell-centered pressure
        P = self._compute_pressure(self.rho, self.E)
        cs_local = cp.sqrt(self.gamma * P / cp.maximum(self.rho, self.rho_floor))
        cs_max = float(cp.max(cs_local))

        signal_speed = max(vmax, 1e-12) + cs_max
        return self.cfl * self.dx / signal_speed

    def _gravity_kick(self, ax, ay, az, dt):
        """
        Update velocities due to gravity.

        Evolution currently provides forces as a 3-tuple (Fx,Fy,Fz).
        We apply them to the first three velocity components only.
        """
        a_list = [ax, ay, az]
        work = 0.0
        for i in range(min(self.dim, len(a_list))):
            delta_v = a_list[i] * dt
            # TODO: check this - gravity must change total energy with kinetic energy.
            axis_work = self.rho * (
                self.vel[i] * delta_v + 0.5 * delta_v * delta_v
            )
            self.E += axis_work
            work = work + axis_work
            self.vel[i] += delta_v

        # sync named components
        self.vx = self.vel[0]
        self.vy = self.vel[1] if self.dim >= 2 else cp.zeros_like(self.vx)
        self.vz = self.vel[2] if self.dim >= 3 else cp.zeros_like(self.vx)
        return work

    def _hydro_step(self, dt):
        """N-D hydro update via directional splitting."""
        self._hydro_step_nd(dt)

    # ---- N-D MUSCL + Rusanov ------------------------------------------------
    @staticmethod
    def _minmod(a, b):
        return 0.5 * (cp.sign(a) + cp.sign(b)) * cp.minimum(cp.abs(a), cp.abs(b))

    def _pressure_from_conserved(self, rho, moms, E):
        rho_safe = cp.maximum(rho, self.rho_floor)
        v2 = cp.zeros_like(rho_safe)
        for m in moms:
            v2 = v2 + (m / rho_safe) ** 2
        eint = E - 0.5 * rho_safe * v2
        eint = cp.maximum(eint, rho_safe * self.e_floor)
        return (self.gamma - 1.0) * eint

    def _reset_emptied_cells(self, rho, moms, E):
        """Cells the update emptied below rho_floor: clamp the density and drop their momentum and
        energy. Keeping them gave v = m / rho_floor (1e12 and more), which later overflowed to NaN."""
        emptied = rho < self.rho_floor
        rho = cp.maximum(rho, self.rho_floor)
        moms = [cp.where(emptied, 0.0, m) for m in moms]
        E = cp.where(emptied, rho * self.e_floor, E)
        return rho, moms, E

    def _hydro_step_nd(self, dt):
        # Build conserved variables
        rho = self.rho
        moms = [rho * v for v in self.vel]
        E = self.E

        # Dimensionally split update (Lie splitting). For many comparisons (esp. 1D) this is enough.
        self._face_work = None  # filled by _sweep_axis when self.phi_total is set
        for axis in range(self.dim):
            # TODO: check this - each directional operator advances the full dt.
            rho, moms, E = self._sweep_axis(rho, moms, E, axis, dt, self.dx_list[axis])
            rho, moms, E = self._reset_emptied_cells(rho, moms, E)

        # Back to primitive
        rho = cp.maximum(rho, self.rho_floor)
        self.rho = rho

        self.vel = [m / rho for m in moms]
        self.E = E

        # Sync named components (pad with zeros when dim < 3)
        self.vx = self.vel[0]
        self.vy = self.vel[1] if self.dim >= 2 else cp.zeros_like(self.vx)
        self.vz = self.vel[2] if self.dim >= 3 else cp.zeros_like(self.vx)

        # Energy floor
        v2 = cp.zeros_like(self.rho)
        for vv in self.vel:
            v2 = v2 + vv * vv
        eint = self.E - 0.5 * self.rho * v2
        eint = cp.maximum(eint, self.rho * self.e_floor)
        self.E = 0.5 * self.rho * v2 + eint

    def _sweep_axis(self, rho, moms, E, axis, dt, dx):
        """
        One directional finite-volume sweep along 'axis' with periodic BC.
        Arrays are updated in a conservative form.
        """
        # Move sweep axis to front for vectorized 1D operations
        def mv(a):
            return cp.moveaxis(a, axis, 0)

        def imv(a):
            return cp.moveaxis(a, 0, axis)

        rho0 = mv(rho)
        moms0 = [mv(m) for m in moms]
        E0 = mv(E)

        # Slopes (cell-centered) for each conserved component
        def slopes(u):
            du_b = u - cp.roll(u, 1, axis=0)
            du_f = cp.roll(u, -1, axis=0) - u
            return self._minmod(du_b, du_f)

        s_rho = slopes(rho0)
        s_m = [slopes(m) for m in moms0]
        s_E = slopes(E0)

        # Reconstruct left/right states at interfaces i+1/2
        rho_L = rho0 + 0.5 * s_rho
        rho_R = cp.roll(rho0, -1, axis=0) - 0.5 * cp.roll(s_rho, -1, axis=0)

        m_L = [m + 0.5 * sm for m, sm in zip(moms0, s_m)]
        m_R = [cp.roll(m, -1, axis=0) - 0.5 * cp.roll(sm, -1, axis=0) for m, sm in zip(moms0, s_m)]

        E_L = E0 + 0.5 * s_E
        E_R = cp.roll(E0, -1, axis=0) - 0.5 * cp.roll(s_E, -1, axis=0)

        # Near-vacuum guard: limiting rho and m independently can leave a face with finite momentum but
        # (floored) tiny density, i.e. a face velocity of 1e10 and more (a vacuum cell between two
        # diverging flows). Where a face velocity exceeds 10x (|v| + c) of the two adjacent cells, fall
        # back to their first-order (cell-centred) states for that face.
        rho_cell = cp.maximum(rho0, self.rho_floor)
        c_cell = cp.sqrt(self.gamma * self._pressure_from_conserved(rho0, moms0, E0) / rho_cell)
        s_cell = cp.abs(moms0[axis] / rho_cell) + c_cell
        s_face = 10.0 * cp.maximum(s_cell, cp.roll(s_cell, -1, axis=0))
        bad = (cp.abs(m_L[axis] / cp.maximum(rho_L, self.rho_floor)) > s_face) | \
              (cp.abs(m_R[axis] / cp.maximum(rho_R, self.rho_floor)) > s_face)
        rho_L = cp.where(bad, rho0, rho_L)
        rho_R = cp.where(bad, cp.roll(rho0, -1, axis=0), rho_R)
        m_L = [cp.where(bad, m, mL) for m, mL in zip(moms0, m_L)]
        m_R = [cp.where(bad, cp.roll(m, -1, axis=0), mR) for m, mR in zip(moms0, m_R)]
        E_L = cp.where(bad, E0, E_L)
        E_R = cp.where(bad, cp.roll(E0, -1, axis=0), E_R)

        # Floors
        rho_L = cp.maximum(rho_L, self.rho_floor)
        rho_R = cp.maximum(rho_R, self.rho_floor)

        P_L = self._pressure_from_conserved(rho_L, m_L, E_L)
        P_R = self._pressure_from_conserved(rho_R, m_R, E_R)

        # Primitive along sweep direction
        vL_k = m_L[axis] / rho_L
        vR_k = m_R[axis] / rho_R

        cL = cp.sqrt(self.gamma * P_L / rho_L)
        cR = cp.sqrt(self.gamma * P_R / rho_R)
        smax = cp.maximum(cp.abs(vL_k) + cL, cp.abs(vR_k) + cR)

        # Fluxes (Euler)
        # mass flux
        FL_rho = m_L[axis]
        FR_rho = m_R[axis]

        # momentum fluxes
        FL_m = []
        FR_m = []
        for i in range(self.dim):
            vL_i = m_L[i] / rho_L
            vR_i = m_R[i] / rho_R
            FL = m_L[i] * vL_k
            FR = m_R[i] * vR_k
            if i == axis:
                FL = FL + P_L
                FR = FR + P_R
            FL_m.append(FL)
            FR_m.append(FR)

        # energy flux
        FL_E = (E_L + P_L) * vL_k
        FR_E = (E_R + P_R) * vR_k

        # Rusanov flux at interfaces
        flux_rho = 0.5 * (FL_rho + FR_rho) - 0.5 * smax * (rho_R - rho_L)
        flux_m = [0.5 * (fL + fR) - 0.5 * smax * (mR - mL) for fL, fR, mL, mR in zip(FL_m, FR_m, m_L, m_R)]
        flux_E = 0.5 * (FL_E + FR_E) - 0.5 * smax * (E_R - E_L)

        # Conservative update: U_i <- U_i - (dt/dx) (F_{i+1/2} - F_{i-1/2})
        fac = float(dt) / float(dx)

        rho1 = rho0 - fac * (flux_rho - cp.roll(flux_rho, 1, axis=0))
        m1 = [m - fac * (f - cp.roll(f, 1, axis=0)) for m, f in zip(moms0, flux_m)]
        E1 = E0 - fac * (flux_E - cp.roll(flux_E, 1, axis=0))

        if self.phi_total is not None:
            # Gravity work done by the mass flux across each interface, F_rho*(Phi_{i+1}-Phi_i),
            # shared equally by the two neighbouring cells. Summed over cells it equals -dW exactly.
            phi0 = mv(self.phi_total)
            interface_work = flux_rho * (cp.roll(phi0, -1, axis=0) - phi0)
            work = imv(-0.5 * fac * (interface_work + cp.roll(interface_work, 1, axis=0)))
            self._face_work = work if self._face_work is None else self._face_work + work

        return imv(rho1), [imv(mi) for mi in m1], imv(E1)

    # -------------------------------------------------------------------------
    # Thermodynamics / cooling
    # -------------------------------------------------------------------------
    def _compute_pressure(self, rho, E):
        """Cell-centered pressure from current velocity fields."""
        v2 = cp.zeros_like(rho)
        for vv in self.vel:
            v2 = v2 + vv * vv
        eint = E - 0.5 * rho * v2
        eint = cp.maximum(eint, rho * self.e_floor)
        return (self.gamma - 1.0) * eint

    def _apply_cooling(self, dt):
        """Optional exponential cooling of internal energy."""
        if self.tcool is None:
            return

        # compute internal energy density, cool it, keep kinetic unchanged
        v2 = cp.zeros_like(self.rho)
        for vv in self.vel:
            v2 = v2 + vv * vv
        K = 0.5 * self.rho * v2
        eint = cp.maximum(self.E - K, self.rho * self.e_floor)

        # exponential decay
        eint_new = eint * cp.exp(-dt / float(self.tcool))
        self.E_radiated += float(cp.sum(eint - eint_new) * self.cell_volume)

        self.E = K + eint_new

    # -------------------------------------------------------------------------
    # Gravity force grids helper
    # -------------------------------------------------------------------------
    def _get_acceleration_grids(self, potential_grid):
        """
        Accept either:
        - tuple/list: interpreted as (Fx,Fy,Fz) already
        - array: interpreted as potential Phi; compute centered-diff forces
        """
        if isinstance(potential_grid, (tuple, list)):
            # Evolution provides (Fx,Fy,Fz)
            Fx = cp.asarray(potential_grid[0], dtype=cp.float64)
            Fy = cp.asarray(potential_grid[1], dtype=cp.float64) if len(potential_grid) > 1 else cp.zeros_like(Fx)
            Fz = cp.asarray(potential_grid[2], dtype=cp.float64) if len(potential_grid) > 2 else cp.zeros_like(Fx)
            return Fx, Fy, Fz

        Phi = cp.asarray(potential_grid, dtype=cp.float64)
        # centered differences along first 3 axes (if present), periodic BC
        Fx = -(cp.roll(Phi, -1, axis=0) - cp.roll(Phi, 1, axis=0)) / (2.0 * self.dx_list[0])
        if self.dim >= 2:
            Fy = -(cp.roll(Phi, -1, axis=1) - cp.roll(Phi, 1, axis=1)) / (2.0 * self.dx_list[1])
        else:
            Fy = cp.zeros_like(Fx)
        if self.dim >= 3:
            Fz = -(cp.roll(Phi, -1, axis=2) - cp.roll(Phi, 1, axis=2)) / (2.0 * self.dx_list[2])
        else:
            Fz = cp.zeros_like(Fx)
        return Fx.astype(cp.float64), Fy.astype(cp.float64), Fz.astype(cp.float64)

    # -------------------------------------------------------------------------
    # Diagnostics (Evolution expects 3-momentum tuple)
    # -------------------------------------------------------------------------
    def total_mass(self):
        return float(cp.sum(self.rho) * self.cell_volume)

    def total_momentum(self):
        # Return 3-tuple for backward compatibility with Evolution_Class
        moms = [cp.sum(self.rho * self.vel[i]) * self.cell_volume for i in range(self.dim)]
        Px = float(moms[0]) if self.dim >= 1 else 0.0
        Py = float(moms[1]) if self.dim >= 2 else 0.0
        Pz = float(moms[2]) if self.dim >= 3 else 0.0
        return Px, Py, Pz

    def total_energy_components(self):
        v2 = cp.zeros_like(self.rho)
        for vv in self.vel:
            v2 = v2 + vv * vv
        K = float(0.5 * cp.sum(self.rho * v2) * self.cell_volume)
        U = self.internal_energy()
        return K, U

    def compute_rotation_curve(self, nbins=40, center=None, zmax=None, rmax=None,
                               mass_weighted=True, axis=None, mass_fraction=0.5):
        """Return a curve in the component's own frame; weights are masses by default."""
        from resources.Classes.Rotation_Curve_Class import RotationCurves
        rotation = RotationCurves(self.simulation)
        data = rotation.gas_data(self, center, axis)
        return rotation.calculate(data, nbins, rmax, zmax, mass_fraction, mass_weighted=mass_weighted)[:4]
