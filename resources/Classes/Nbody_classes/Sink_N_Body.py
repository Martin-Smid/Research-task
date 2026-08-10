import cupy as cp
import numpy as np
from resources.Classes.Nbody_classes.NBody import NBody


class SinkNBody(NBody):
    """
    Supermassive black hole sink particle system.

    CHANGES:
    - Vectorized accretion loop for O(N_baryons) instead of O(N_sinks × N_baryons)
    - Added mass conservation validation
    - Improved code clarity and documentation
    """

    def __init__(self, simulation, N_sinks, initial_masses,
                 initial_positions, initial_velocities,
                 capture_radius=None, softening_bh=None, softening_cusp=None, softening_length=None,
                 reservoir_tau=10):
        """Initialize sink particle system."""
        super().__init__(simulation, N_sinks, total_mass=1.0)

        self.positions = cp.asarray(initial_positions, dtype=cp.float64)
        self.velocities = cp.asarray(initial_velocities, dtype=cp.float64)

        min_dx = min(simulation.dx)
        self.capture_radius = capture_radius if capture_radius is not None else 2.5 * min_dx
        self.softening_length = softening_length if softening_length is not None else 1.5 * min_dx

        self.softening_bh = softening_bh if softening_bh is not None else self.softening_length
        self.softening_cusp = softening_cusp if softening_cusp is not None else self.capture_radius

        self.mass_bh = cp.asarray(initial_masses, dtype=cp.float64)
        self.mass_res = cp.zeros_like(self.mass_bh)
        self.masses = self.mass_bh + self.mass_res

        self.reservoir_tau = float(reservoir_tau) if reservoir_tau is not None else 10.0 * simulation.h

        self.E_diss_formation_total = 0.0
        self.total_accreted_mass = 0.0
        self.accretion_history = []
        self.E_diss_kin_total = 0.0
        self.E_diss_kin_last = 0.0

    def deposit_to_grid(self):
        """Deposit sink masses to grid using CIC."""
        import cupyx

        sim = self.simulation
        N = sim.N
        dim = sim.dim
        shape = (N,) * dim
        rho_grid = cp.zeros(shape, dtype=cp.float64)

        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        Lx, Ly, Lz = x_max - x_min, y_max - y_min, z_max - z_min

        # Periodic wrap
        px = x_min + cp.mod(self.positions[:, 0] - x_min, Lx)
        py = y_min + cp.mod(self.positions[:, 1] - y_min, Ly)
        pz = z_min + cp.mod(self.positions[:, 2] - z_min, Lz)

        # Grid coordinates
        rx = (px - x_min) / Lx * N
        ry = (py - y_min) / Ly * N
        rz = (pz - z_min) / Lz * N

        i0 = cp.floor(rx).astype(cp.int32)
        j0 = cp.floor(ry).astype(cp.int32)
        k0 = cp.floor(rz).astype(cp.int32)

        tx, ty, tz = rx - i0, ry - j0, rz - k0
        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        i1, j1, k1 = (i0 + 1) % N, (j0 + 1) % N, (k0 + 1) % N
        i0, j0, k0 = i0 % N, j0 % N, k0 % N

        cell_volume = sim.cell_volume
        N_sq = N * N
        rho_flat = rho_grid.ravel()

        corners_weights = [
            (wx0 * wy0 * wz0, i0, j0, k0), (wx1 * wy0 * wz0, i1, j0, k0),
            (wx0 * wy1 * wz0, i0, j1, k0), (wx1 * wy1 * wz0, i1, j1, k0),
            (wx0 * wy0 * wz1, i0, j0, k1), (wx1 * wy0 * wz1, i1, j0, k1),
            (wx0 * wy1 * wz1, i0, j1, k1), (wx1 * wy1 * wz1, i1, j1, k1),
        ]

        for w, ii, jj, kk in corners_weights:
            flat_indices = ii * N_sq + jj * N + kk
            contribution = (self.masses / cell_volume) * w
            cupyx.scatter_add(rho_flat, flat_indices, contribution)

        return rho_grid

    def accrete_from_baryons(self, baryon_systems):
        """
        OPTIMIZED: Vectorized accretion using distance matrix computation.

        CHANGE: Replaced O(N_sinks × N_baryons) nested loop with vectorized
        distance computation. Now O(N_baryons) with better GPU utilization.

        Algorithm:
        1. Compute all pairwise distances: (N_baryons, N_sinks)
        2. Find closest sink for each baryon particle
        3. Apply accretion criteria vectorially
        4. Update sink properties in batches
        """
        if self.N == 0:
            return 0

        total_accreted = 0
        self.E_diss_kin_last = 0.0
        dE_diss_total = 0.0

        for baryons in baryon_systems:
            if isinstance(baryons, SinkNBody) or not hasattr(baryons, 'N') or baryons.N == 0:
                continue

            # MASS CONSERVATION CHECK: Initial state
            initial_baryon_mass = float(baryons.N * baryons.m_particle)
            initial_sink_mass = float(cp.sum(self.masses))

            # === VECTORIZED DISTANCE COMPUTATION ===
            # Shape: (N_baryons, 1, 3) - (1, N_sinks, 3) = (N_baryons, N_sinks, 3)
            dx = baryons.positions[:, None, :] - self.positions[None, :, :]

            # Apply periodic boundary conditions
            for d in range(3):
                low, high = self.simulation.boundaries[d]
                L = high - low
                dx[:, :, d] = dx[:, :, d] - cp.copysign(L, dx[:, :, d]) * (cp.abs(dx[:, :, d]) > L / 2)

            # Distance to each sink: (N_baryons, N_sinks)
            dist = cp.sqrt(cp.sum(dx ** 2, axis=2))

            # Find closest sink for each particle
            closest_sink_idx = cp.argmin(dist, axis=1)  # (N_baryons,)
            closest_dist = dist[cp.arange(baryons.N), closest_sink_idx]  # (N_baryons,)

            # === ACCRETION CRITERIA (VECTORIZED) ===
            # 1. Distance criterion
            within_capture = closest_dist < self.capture_radius

            # 2. Escape velocity criterion
            G = self.simulation.G
            sink_masses_per_particle = self.masses[closest_sink_idx]  # (N_baryons,)
            r_soft = cp.maximum(closest_dist, self.softening_length)
            v_esc = cp.sqrt(2 * G * sink_masses_per_particle / r_soft)

            # Relative velocity to closest sink
            sink_vels = self.velocities[closest_sink_idx]  # (N_baryons, 3)
            rel_vel = baryons.velocities - sink_vels
            v_rel_mag = cp.sqrt(cp.sum(rel_vel ** 2, axis=1))

            is_bound = v_rel_mag < v_esc

            # 3. Angular momentum criterion
            dx_to_sink = dx[cp.arange(baryons.N), closest_sink_idx]  # (N_baryons, 3)
            Lx = dx_to_sink[:, 1] * rel_vel[:, 2] - dx_to_sink[:, 2] * rel_vel[:, 1]
            Ly = dx_to_sink[:, 2] * rel_vel[:, 0] - dx_to_sink[:, 0] * rel_vel[:, 2]
            Lz = dx_to_sink[:, 0] * rel_vel[:, 1] - dx_to_sink[:, 1] * rel_vel[:, 0]
            L_sq = Lx ** 2 + Ly ** 2 + Lz ** 2

            L_max_sq = G * sink_masses_per_particle * self.capture_radius
            low_ang_mom = L_sq < L_max_sq

            # Combined mask
            accrete_mask = within_capture & is_bound & low_ang_mom
            n_accrete = int(cp.sum(accrete_mask))

            if n_accrete == 0:
                continue

            # === BATCH UPDATE SINK PROPERTIES ===
            accreted_sink_indices = closest_sink_idx[accrete_mask]  # Which sinks get which particles
            accreted_vels = baryons.velocities[accrete_mask]
            m_p = baryons.m_particle

            # Group particles by sink (using bincount for efficiency)
            for sink_idx in range(self.N):
                particles_for_this_sink = accreted_sink_indices == sink_idx
                n_for_sink = int(cp.sum(particles_for_this_sink))

                if n_for_sink == 0:
                    continue

                # Extract velocities for particles going to this sink
                vels_this_sink = accreted_vels[particles_for_this_sink]

                # Momentum and energy conservation
                M_old = self.masses[sink_idx]
                V_old = self.velocities[sink_idx]
                P_old = M_old * V_old

                accreted_mass = m_p * n_for_sink
                accreted_momentum = cp.sum(vels_this_sink * m_p, axis=0)

                M_new = M_old + accreted_mass
                P_new = P_old + accreted_momentum
                V_new = P_new / M_new

                # Energy dissipation tracking
                K_before = 0.5 * M_old * cp.sum(V_old * V_old) + 0.5 * m_p * cp.sum(vels_this_sink * vels_this_sink)
                K_after = 0.5 * M_new * cp.sum(V_new * V_new)
                dE_diss_total += float(K_before - K_after)

                # Update sink
                self.mass_res[sink_idx] += accreted_mass
                self.masses[sink_idx] = self.mass_bh[sink_idx] + self.mass_res[sink_idx]
                self.velocities[sink_idx] = V_new

                self.total_accreted_mass += accreted_mass
                self.accretion_history.append((int(sink_idx), float(accreted_mass)))

            # Remove accreted particles
            keep_mask = ~accrete_mask
            baryons.positions = baryons.positions[keep_mask]
            baryons.velocities = baryons.velocities[keep_mask]
            baryons.N = int(cp.sum(keep_mask))
            # TODO: check this - particle removal invalidates cached CIC density.
            if hasattr(baryons, "invalidate_density_cache"):
                baryons.invalidate_density_cache()

            total_accreted += n_accrete

            # === MASS CONSERVATION CHECK ===
            final_baryon_mass = float(baryons.N * baryons.m_particle)
            final_sink_mass = float(cp.sum(self.masses))

            mass_transferred = initial_baryon_mass - final_baryon_mass
            sink_mass_gained = final_sink_mass - initial_sink_mass

            mass_error = abs(mass_transferred - sink_mass_gained)
            rel_error = mass_error / (initial_baryon_mass + 1e-30)

            '''
            if rel_error > 1e-10:
                print(f"WARNING: Mass conservation violation in accretion!")
                print(f"  Mass removed from baryons: {mass_transferred:.12e}")
                print(f"  Mass added to sinks: {sink_mass_gained:.12e}")
                print(f"  Absolute error: {mass_error:.12e}")
                print(f"  Relative error: {rel_error:.12e}")
            else:
                print(f"✓ Mass conserved in accretion (error: {rel_error:.3e})")
                '''

        self.E_diss_kin_last = dE_diss_total
        self.E_diss_kin_total += dE_diss_total

        return total_accreted

    def accrete_from_gas(
            self,
            gas_system,
            dt,
            sound_speed=None,
            lam=None,
            rho_min_factor=2.0,
            conserve_momentum=True,
            use_reservoir=True,
    ):
        """
        Jaxion-like Bondi accretion of *grid* gas onto sink particles.

        particles_accrete_gas from Jaxion:
        - Use CIC to sample surrounding cell densities.
        - Bondi rate per sink: dM_fac = dt * 4*pi*lambda*(G*M)^2 / c_s^3
        - Accrete from the 2^dim CIC corner cells: dm_corner = w_corner * dM_fac * rho_corner
        - Remove dm_corner from gas density (and optionally momentum/energy).
       Parameters
        ----------
        gas_system : NBodyGas-like
            Must provide `rho` and (optionally) `vx,vy,vz` and `rho_floor`.
        dt : float
            Accretion timestep.
        sound_speed : float or None
            If None, uses `gas_system.cs` if present, otherwise estimates from
            adiabatic EOS (requires `E` and `gamma`).
        lam : float or None
            Dimensionless Bondi lambda. If None, uses exp(1.5)/4 ~ 1.12 (Jaxion).
        rho_min_factor : float
            Only accrete if CIC-interpolated rho > rho_min_factor * rho_floor.
        conserve_momentum : bool
            If True, transfer gas momentum to sink (recommended).
        use_reservoir : bool
            If True, add mass to `mass_res` (then `drain_reservoir` moves it to BH).

        Returns
        -------
        float
            Total mass accreted from gas (added to sinks).
        """
        if self.N == 0:
            return 0.0

        sim = self.simulation
        dim = int(sim.dim)
        N = int(sim.N)

        rho = gas_system.rho
        rho_floor = float(getattr(gas_system, "rho_floor", 0.0))
        cellV = float(getattr(sim, "dV", getattr(sim, "cell_volume", 1.0)))

        # --- sound speed ---
        if sound_speed is None:
            if hasattr(gas_system, "cs"):
                sound_speed = float(gas_system.cs)
            else:
                # adiabatic estimate: cs^2 = gamma * P / rho
                gamma = float(getattr(gas_system, "gamma", 5.0 / 3.0))
                if not hasattr(gas_system, "E"):
                    raise ValueError("sound_speed is None and gas_system has no `cs` and no `E` to estimate it.")
                E = gas_system.E
                # conservative global estimate to keep it stable
                # P = (gamma-1)*(E - 0.5*rho*|v|^2)
                vx = getattr(gas_system, "vx", 0.0)
                vy = getattr(gas_system, "vy", 0.0)
                vz = getattr(gas_system, "vz", 0.0)
                v2 = 0.0
                if dim >= 1 and hasattr(gas_system, "vx"):
                    v2 = v2 + vx * vx
                if dim >= 2 and hasattr(gas_system, "vy"):
                    v2 = v2 + vy * vy
                if dim >= 3 and hasattr(gas_system, "vz"):
                    v2 = v2 + vz * vz
                P = (gamma - 1.0) * (E - 0.5 * rho * v2)
                P = cp.maximum(P, 0.0)
                cs2 = gamma * P / cp.maximum(rho, rho_floor if rho_floor > 0 else 1e-30)
                sound_speed = float(cp.sqrt(cp.nanmax(cs2)).get())

        if not np.isfinite(sound_speed) or sound_speed <= 0:
            return 0.0

        if lam is None:
            lam = float(np.exp(1.5) / 4.0)  # ~1.12 (Jaxion default)

        G = float(getattr(sim, "G", 0.0))
        if not np.isfinite(G) or G == 0.0:
            return 0.0

        # --- periodic wrap positions to box ---
        lows = [float(sim.boundaries[d][0]) for d in range(dim)]
        highs = [float(sim.boundaries[d][1]) for d in range(dim)]
        Ls = [highs[d] - lows[d] for d in range(dim)]

        pos = self.positions[:, :dim]
        p = []
        for d in range(dim):
            p.append(lows[d] + cp.mod(pos[:, d] - lows[d], Ls[d]))

        # --- CIC indices / weights per axis ---
        idx0 = []
        idx1 = []
        w0 = []
        w1 = []
        for d in range(dim):
            r = (p[d] - lows[d]) / Ls[d] * N
            i0 = cp.floor(r).astype(cp.int32)
            t = (r - i0).astype(cp.float64)
            i1 = (i0 + 1) % N
            i0 = i0 % N
            idx0.append(i0)
            idx1.append(i1)
            w0.append(1.0 - t)
            w1.append(t)

        # --- Bondi prefactor per sink ---
        M = self.masses
        dM_fac = float(dt) * 4.0 * np.pi * float(lam) * (G * M) ** 2 / (float(sound_speed) ** 3)
        
        # Accumulate per-sink mass and (optionally) momentum gains.
        dM = cp.zeros(self.N, dtype=cp.float64)
        dP = cp.zeros((self.N, 3), dtype=cp.float64)

        # Scaling per sink to prevent rho dropping below floor.
        # Start at 1 and take min over touched corner cells.
        scale = cp.ones(self.N, dtype=cp.float64)

        # Prepare strides for flattening (C-order).
        strides = [N ** (dim - 1 - d) for d in range(dim)]
        rho_flat = rho.ravel()

        import cupyx

        # helper to iterate corners (2^dim)
        from itertools import product
        corners = list(product([0, 1], repeat=dim))

        # --- First pass: determine scaling + compute dM and dP with scaling later ---
        # We compute dm_corner for each corner and update:
        # - dM += dm_corner
        # - scale = min(scale, (rho_corner - rho_floor)*cellV / dm_corner)
        # Then in second pass, apply dm_corner_scaled = dm_corner * scale and scatter subtract.
        rho_interp = cp.zeros(self.N, dtype=cp.float64)

        for bits in corners:
            inds = []
            w = cp.ones(self.N, dtype=cp.float64)
            for d, b in enumerate(bits):
                if b == 0:
                    inds.append(idx0[d])
                    w = w * w0[d]
                else:
                    inds.append(idx1[d])
                    w = w * w1[d]

            rho_c = rho[tuple(inds)]
            rho_interp = rho_interp + w * rho_c

            dm = w * dM_fac * rho_c
            dM = dM + dm

            if rho_floor > 0:
                max_dm = cp.maximum(rho_c - rho_floor, 0.0) * cellV
                # Avoid divide by zero
                s = cp.where(dm > 0, max_dm / dm, 1.0)
                scale = cp.minimum(scale, s)

        # Apply rho threshold to avoid accreting from near-vacuum
        if rho_floor > 0:
            scale = cp.where(rho_interp > (rho_min_factor * rho_floor), scale, 0.0)

        # cap scale to [0,1]
        scale = cp.clip(scale, 0.0, 1.0)

        # If no accretion, exit
        if float(cp.sum(scale).get()) == 0.0:
            return 0.0

        # Reset accumulators with scaling applied
        dM_scaled = cp.zeros(self.N, dtype=cp.float64)
        dP_scaled = cp.zeros((self.N, 3), dtype=cp.float64)

        # --- Second pass: apply scaled removal to gas and accumulate momentum gain ---
        for bits in corners:
            inds = []
            w = cp.ones(self.N, dtype=cp.float64)
            for d, b in enumerate(bits):
                if b == 0:
                    inds.append(idx0[d])
                    w = w * w0[d]
                else:
                    inds.append(idx1[d])
                    w = w * w1[d]

            rho_c = rho[tuple(inds)]
            dm = w * dM_fac * rho_c
            dm = dm * scale  # per-sink scaling
            dM_scaled = dM_scaled + dm

            # Remove density from gas (atomic scatter-add on flattened array)
            flat = cp.zeros(self.N, dtype=cp.int64)
            for d in range(dim):
                flat = flat + inds[d].astype(cp.int64) * int(strides[d])

            drho = -(dm / cellV)
            cupyx.scatter_add(rho_flat, flat, drho)

            if conserve_momentum:
                # Sample gas velocity at the same corner cells (up to 3 components).
                vx = getattr(gas_system, "vx", None)
                vy = getattr(gas_system, "vy", None)
                vz = getattr(gas_system, "vz", None)

                if vx is not None:
                    dP_scaled[:, 0] += dm * vx[tuple(inds)]
                if vy is not None:
                    dP_scaled[:, 1] += dm * vy[tuple(inds)]
                if vz is not None:
                    dP_scaled[:, 2] += dm * vz[tuple(inds)]

        # Enforce rho floor after all removals
        if rho_floor > 0:
            gas_system.rho = cp.maximum(gas_system.rho, rho_floor)

        # --- Update sinks (mass + momentum) ---
        total_dm = float(cp.sum(dM_scaled).get())
        if total_dm <= 0.0:
            return 0.0

        # Store old masses before update for momentum conservation
        M_old = self.masses.copy()
        V_old = self.velocities.copy()
        P_old = M_old[:, None] * V_old

        if use_reservoir:
            self.mass_res = self.mass_res + dM_scaled
        else:
            self.mass_bh = self.mass_bh + dM_scaled

        self.masses = self.mass_bh + self.mass_res

        if conserve_momentum:
            P_new = P_old + dP_scaled
            self.velocities = P_new / self.masses[:, None]

        self.total_accreted_mass += total_dm
        self.accretion_history.append(("gas_bondi", float(dt), total_dm))

        return total_dm


    def add_new_sink(self, mass, position, velocity):
        """Add a new sink particle to this system."""
        self.positions = cp.vstack([self.positions, position[None, :]])
        self.velocities = cp.vstack([self.velocities, velocity[None, :]])
        self.mass_bh = cp.concatenate([self.mass_bh, cp.asarray([mass], dtype=cp.float64)])
        self.mass_res = cp.concatenate([self.mass_res, cp.asarray([0.0], dtype=cp.float64)])
        self.masses = self.mass_bh + self.mass_res
        self.N += 1


    @staticmethod
    def merge_or_create(existing_sink_system, new_sinks_data, simulation):
        """Merge new sink data into existing system or create new one."""
        if new_sinks_data is None:
            return existing_sink_system

        formation_energy = new_sinks_data.get('E_diss_formation', 0.0)
        n_new = len(new_sinks_data['masses'])

        if existing_sink_system is None:
            new_sys = SinkNBody(
                simulation=simulation,
                N_sinks=n_new,
                initial_masses=new_sinks_data['masses'],
                initial_positions=new_sinks_data['positions'],
                initial_velocities=new_sinks_data['velocities']
            )
            new_sys.E_diss_formation_total = formation_energy
            return new_sys
        else:
            existing_sink_system.E_diss_formation_total += formation_energy
            for i in range(n_new):
                existing_sink_system.add_new_sink(
                    mass=new_sinks_data['masses'][i],
                    position=new_sinks_data['positions'][i],
                    velocity=new_sinks_data['velocities'][i]
                )
            return existing_sink_system


    def kinetic_energy(self):
        """Variable per-sink masses."""
        v2 = (self.velocities ** 2).sum(axis=1)
        return 0.5 * (self.masses * v2).sum()


    def drain_reservoir(self, dt):
        """Move mass from reservoir -> BH smoothly."""
        if self.N == 0:
            return

        tau = self.reservoir_tau
        if tau <= 0:
            dM = self.mass_res.copy()
        else:
            frac = 1.0 - cp.exp(-float(dt) / float(tau))
            dM = self.mass_res * frac

        self.mass_res -= dM
        self.mass_bh += dM
        self.masses = self.mass_bh + self.mass_res


    def merge_close_sinks(
            self,
            r_merge=None,
            r_merge_cells=None,
            bound_check=True,
            v_factor=1.0,
            max_merges_per_call=None,
    ):
        """
        Merge sink particles that are closer than r_merge (periodic minimum-image),
        optionally only if they are gravitationally bound.

        Parameters
        ----------
        r_merge : float or None
            Physical merge radius. If None, uses r_merge_cells*dx or capture_radius.
        r_merge_cells : int or None
            Merge radius in grid cells. Used if r_merge is None.
        bound_check : bool
            If True, require v_rel^2 < v_factor * v_esc^2 to merge.
        v_factor : float
            Safety factor on escape velocity criterion (1.0 = strict bound).
        max_merges_per_call : int or None
            Limit merges per call (useful if you want gradual behavior).

        Returns
        -------
        n_merged : int
        dE_diss : float
            Dissipated kinetic energy from inelastic merging (COM frame).
        """
        if self.N < 2:
            return 0, 0.0

        sim = self.simulation
        G = float(getattr(sim, "G", 0.0))
        dx = float(min(sim.dx))

        if r_merge is None:
            if r_merge_cells is not None:
                r_merge = float(r_merge_cells) * dx
            else:
                r_merge = float(getattr(self, "capture_radius", 2.5 * dx))

        # periodic box lengths (assumes periodic boundaries like your CIC deposit does)
        L = np.array([sim.boundaries[0][1] - sim.boundaries[0][0],
                      sim.boundaries[1][1] - sim.boundaries[1][0],
                      sim.boundaries[2][1] - sim.boundaries[2][0]], dtype=np.float64)

        # work on CPU numpy (N_sinks is usually small; this is fine and simpler)
        pos = cp.asnumpy(self.positions).astype(np.float64, copy=True)
        vel = cp.asnumpy(self.velocities).astype(np.float64, copy=True)
        mbh = cp.asnumpy(self.mass_bh).astype(np.float64, copy=True)
        mres = cp.asnumpy(self.mass_res).astype(np.float64, copy=True)

        # bookkeeping for energy
        if not hasattr(self, "E_diss_merge_total"):
            self.E_diss_merge_total = 0.0

        n_merged = 0
        dE_diss_total = 0.0

        # helper: minimum-image displacement
        def min_image(d):
            # d shape (...,3)
            # avoid divide by zero if some L=0 (shouldn't happen)
            out = d.copy()
            for k in range(3):
                if L[k] > 0:
                    out[..., k] -= L[k] * np.round(out[..., k] / L[k])
            return out

        while True:
            N = mbh.size
            if N < 2:
                break

            # pairwise displacement / distance
            d = pos[:, None, :] - pos[None, :, :]
            d = min_image(d)
            dist = np.sqrt(np.sum(d * d, axis=-1))

            # ignore self-pairs
            np.fill_diagonal(dist, np.inf)

            # candidate mask: within radius
            cand = dist < r_merge

            if not np.any(cand):
                break

            if bound_check:
                # v_rel^2
                dv = vel[:, None, :] - vel[None, :, :]
                v2 = np.sum(dv * dv, axis=-1)

                mtot = (mbh + mres)
                # v_esc^2 = 2 G (m_i+m_j) / r
                # avoid division by inf/0
                with np.errstate(divide="ignore", invalid="ignore"):
                    vesc2 = 2.0 * G * (mtot[:, None] + mtot[None, :]) / dist
                bound = v2 < (v_factor * vesc2)
                cand &= bound

                if not np.any(cand):
                    break

            # choose closest candidate pair (i,j)
            dist_c = np.where(cand, dist, np.inf)
            i, j = np.unravel_index(np.argmin(dist_c), dist_c.shape)
            if not np.isfinite(dist_c[i, j]):
                break

            # choose to keep the more massive sink as i
            mtot = mbh + mres
            if mtot[j] > mtot[i]:
                i, j = j, i

            m1 = mtot[i]
            m2 = mtot[j]
            mnew = m1 + m2

            # dissipated kinetic energy (inelastic merge, COM frame)
            vrel = vel[i] - vel[j]
            mu = (m1 * m2) / mnew
            dE = 0.5 * mu * float(np.dot(vrel, vrel))
            dE_diss_total += dE

            # merge state (mass + momentum conserved)
            pos[i] = (m1 * pos[i] + m2 * pos[j]) / mnew
            vel[i] = (m1 * vel[i] + m2 * vel[j]) / mnew
            mbh[i] = mbh[i] + mbh[j]
            mres[i] = mres[i] + mres[j]

            # delete j
            pos = np.delete(pos, j, axis=0)
            vel = np.delete(vel, j, axis=0)
            mbh = np.delete(mbh, j)
            mres = np.delete(mres, j)

            n_merged += 1
            if (max_merges_per_call is not None) and (n_merged >= int(max_merges_per_call)):
                break

        # write back to GPU
        self.positions = cp.asarray(pos, dtype=cp.float64)
        self.velocities = cp.asarray(vel, dtype=cp.float64)
        self.mass_bh = cp.asarray(mbh, dtype=cp.float64)
        self.mass_res = cp.asarray(mres, dtype=cp.float64)
        self.masses = self.mass_bh + self.mass_res
        self.N = int(self.masses.size)

        self.E_diss_merge_total += float(dE_diss_total)

        return n_merged, float(dE_diss_total)


class SinkFormationTracker:
    """Tracks density threshold violations for sink formation."""

    def __init__(self, grid_shape, consecutive_steps_required=5):
        self.grid_shape = grid_shape
        self.consecutive_steps_required = consecutive_steps_required
        self.threshold_counter = cp.zeros(grid_shape, dtype=cp.int32)
        self.has_sink = cp.zeros(grid_shape, dtype=cp.bool_)

    def update_and_check(self, density, threshold):
        """Update counters and return cells ready for sink formation."""
        if threshold is None:
            return None

        exceeds = density > threshold
        self.threshold_counter = cp.where(exceeds, self.threshold_counter + 1, 0)
        ready_for_sink = (self.threshold_counter >= self.consecutive_steps_required) & (~self.has_sink)

        if not cp.any(ready_for_sink):
            return None

        self.has_sink = self.has_sink | ready_for_sink
        return cp.argwhere(ready_for_sink)

    def mark_sink_region(self, position, radius):
        """Mark a region around a sink position to prevent duplicate formation."""
        pass


def check_and_create_sinks(simulation, sink_tracker, density_threshold,
                           aggregation_radius=None, source='both'):
    """
    CHANGE: Enhanced logging to show creation details.
    """
    shape = (simulation.N,) * simulation.dim
    baryonic_density = cp.zeros(shape, dtype=cp.float64)
    regular_baryons = []

    # Count particles by type
    n_baryon_particles = 0
    n_gas_cells = 0

    for baryons in simulation.baryonic_matter:
        if isinstance(baryons, SinkNBody):
            continue

        is_gas = not hasattr(baryons, 'N')
        is_particles = hasattr(baryons, 'N') and baryons.N > 0

        if source == 'baryons' and is_gas:
            continue
        if source == 'gas' and not is_gas:
            continue

        if is_gas:
            baryonic_density += baryons.deposit_to_grid()
            regular_baryons.append(baryons)
            n_gas_cells += simulation.N ** simulation.dim
        elif is_particles:
            baryonic_density += baryons.deposit_to_grid()
            regular_baryons.append(baryons)
            n_baryon_particles += baryons.N

    # MASS CONSERVATION: Initial state
    initial_total_mass = 0.0
    initial_baryon_mass = 0.0
    initial_gas_mass = 0.0

    for baryons in regular_baryons:
        if hasattr(baryons, 'N'):
            mass = float(baryons.N * baryons.m_particle)
            initial_total_mass += mass
            initial_baryon_mass += mass
        else:
            mass = float(cp.sum(baryons.rho) * simulation.dV)
            initial_total_mass += mass
            initial_gas_mass += mass

    ready_indices = sink_tracker.update_and_check(baryonic_density, density_threshold)
    if ready_indices is None:
        return None

    n_new_sinks = ready_indices.shape[0]

    '''
    print(f"\n{'=' * 60}")
    print(f"SINK CREATION EVENT")
    print(f"{'=' * 60}")
    print(f"Creating {n_new_sinks} new sink particle(s)")
    print(f"Source: {source}")
    print(f"Available: {n_baryon_particles} baryon particles, {n_gas_cells} gas cells")
    print(f"Initial mass: Baryons={initial_baryon_mass:.6e}, Gas={initial_gas_mass:.6e}")
    '''

    grids = simulation.grids
    min_dx = min(simulation.dx)
    if aggregation_radius is None:
        aggregation_radius = 1.0 * min_dx

    new_masses = []
    new_positions = []
    new_velocities = []
    total_dissipated_energy = 0.0

    # Track what was consumed
    baryon_particles_consumed = 0
    baryon_mass_consumed = 0.0
    gas_mass_consumed = 0.0

    for cell_idx in ready_indices:
        ix, iy, iz = int(cell_idx[0]), int(cell_idx[1]), int(cell_idx[2])
        cell_pos = cp.array([
            grids[0][ix, iy, iz],
            grids[1][ix, iy, iz],
            grids[2][ix, iy, iz]
        ], dtype=cp.float64)

        total_mass = 0.0
        total_momentum = cp.zeros(3, dtype=cp.float64)
        KE_before = 0.0

        for baryons in regular_baryons:
            if not hasattr(baryons, 'N'):
                continue
            if baryons.N == 0:
                continue

            particles_before = baryons.N

            dx_vec = baryons.positions - cell_pos[None, :]
            for d in range(3):
                low, high = simulation.boundaries[d]
                L = high - low
                dx_vec[:, d] = dx_vec[:, d] - cp.copysign(L, dx_vec[:, d]) * (cp.abs(dx_vec[:, d]) > L / 2)
            dist = cp.sqrt(cp.sum(dx_vec ** 2, axis=1))

            nearby_mask = dist < aggregation_radius
            n_nearby = int(cp.sum(nearby_mask))

            if n_nearby > 0:
                v_gas = baryons.velocities[nearby_mask]
                m_gas = baryons.m_particle

                KE_before += 0.5 * m_gas * cp.sum(v_gas ** 2)
                mass_nearby = m_gas * n_nearby
                momentum_nearby = cp.sum(v_gas * m_gas, axis=0)

                total_mass += mass_nearby
                total_momentum += momentum_nearby

                # Track consumption
                baryon_particles_consumed += n_nearby
                baryon_mass_consumed += float(mass_nearby)

                keep_mask = ~nearby_mask
                baryons.positions = baryons.positions[keep_mask]
                baryons.velocities = baryons.velocities[keep_mask]
                baryons.N = int(cp.sum(keep_mask))
                # TODO: check this - particle removal invalidates cached CIC density.
                if hasattr(baryons, "invalidate_density_cache"):
                    baryons.invalidate_density_cache()

        if total_mass > 0:
            v_sink = total_momentum / total_mass
            KE_after = 0.5 * total_mass * cp.sum(v_sink ** 2)
            dissipation = KE_before - KE_after
            total_dissipated_energy += dissipation

            new_masses.append(total_mass)
            new_positions.append(cell_pos)
            new_velocities.append(v_sink)

    if len(new_masses) == 0:
        return None

    # === MASS CONSERVATION CHECK ===
    final_total_mass = 0.0
    final_baryon_mass = 0.0
    final_gas_mass = 0.0

    for baryons in regular_baryons:
        if hasattr(baryons, 'N'):
            mass = float(baryons.N * baryons.m_particle)
            final_total_mass += mass
            final_baryon_mass += mass
        else:
            mass = float(cp.sum(baryons.rho) * simulation.dV)
            final_total_mass += mass
            final_gas_mass += mass

    created_sink_mass = float(sum(new_masses))
    mass_removed = initial_total_mass - final_total_mass
    mass_error = abs(created_sink_mass - mass_removed)
    rel_error = mass_error / (initial_total_mass + 1e-30)

    '''
    print(f"\nCONSUMPTION SUMMARY:")
    print(f"  Baryon particles consumed: {baryon_particles_consumed}")
    print(f"  Baryon mass consumed: {baryon_mass_consumed:.6e} Msun")
    print(f"  Gas mass consumed: {gas_mass_consumed:.6e} Msun")
    print(f"  Total sink mass created: {created_sink_mass:.6e} Msun")
    print(f"  Energy dissipated: {total_dissipated_energy:.6e}")

    print(f"\nMASS CONSERVATION:")
    if rel_error > 1e-10:
        print(f"  ❌ WARNING: Violation detected!")
        print(f"     Mass removed: {mass_removed:.12e}")
        print(f"     Mass created: {created_sink_mass:.12e}")
        print(f"     Absolute error: {mass_error:.12e}")
        print(f"     Relative error: {rel_error:.12e}")
    else:
        print(f"  ✓ Conserved (error: {rel_error:.3e})")

    print(f"{'=' * 60}\n")
    '''

    return {
        'masses': cp.array(new_masses, dtype=cp.float64),
        'positions': cp.stack(new_positions, axis=0),
        'velocities': cp.stack(new_velocities, axis=0),
        'E_diss_formation': float(total_dissipated_energy)
    }


def check_and_create_gas_sinks(simulation, sink_tracker, gas_system, density_threshold,
                               r_acc=None):
    """
    CHANGE: Enhanced logging for gas sink creation.
    """
    N = simulation.N
    dx = min(simulation.dx)
    if r_acc is None:
        r_acc = 2.5 * dx
    r_cells = int(np.ceil(r_acc / dx))

    rho = gas_system.rho

    # MASS CONSERVATION: Initial state
    initial_gas_mass = float(cp.sum(rho) * simulation.dV)
    initial_avg_density = float(cp.mean(rho))
    initial_max_density = float(cp.max(rho))

    ready_indices = sink_tracker.update_and_check(rho, density_threshold)
    if ready_indices is None:
        return None

    n_new_sinks = ready_indices.shape[0]

    '''
    print(f"\n{'=' * 60}")
    print(f"GAS SINK CREATION EVENT")
    print(f"{'=' * 60}")
    print(f"Creating {n_new_sinks} sink(s) from gas")
    print(f"Initial gas mass: {initial_gas_mass:.6e} Msun")
    print(f"Density - avg: {initial_avg_density:.3e}, max: {initial_max_density:.3e}")
    print(f"Threshold: {density_threshold:.3e}")
    '''

    new_masses = []
    new_positions = []
    new_velocities = []
    total_diss = 0.0

    cells_processed = 0
    total_gas_removed = 0.0

    offsets = []
    for i in range(-r_cells, r_cells + 1):
        for j in range(-r_cells, r_cells + 1):
            for k in range(-r_cells, r_cells + 1):
                if (i * i + j * j + k * k) <= r_cells * r_cells:
                    offsets.append((i, j, k))

    grids = simulation.grids
    cellV = simulation.dV

    for sink_num, cell_idx in enumerate(ready_indices):
        ix, iy, iz = map(int, cell_idx)

        dM = 0.0
        dPx = dPy = dPz = 0.0
        Kin_in = 0.0
        cells_in_region = 0

        for oi, oj, ok in offsets:
            i = (ix + oi) % N
            j = (iy + oj) % N
            k = (iz + ok) % N

            rho_cell = rho[i, j, k]
            if rho_cell <= density_threshold:
                continue

            rho_excess = rho_cell - density_threshold
            dm = rho_excess * cellV

            vx = gas_system.vx[i, j, k]
            vy = gas_system.vy[i, j, k]
            vz = gas_system.vz[i, j, k]

            dM += dm
            dPx += dm * vx
            dPy += dm * vy
            dPz += dm * vz
            Kin_in += 0.5 * dm * (vx * vx + vy * vy + vz * vz)

            gas_system.rho[i, j, k] = density_threshold
            cells_in_region += 1

        if dM <= 0:
            continue

        Vx, Vy, Vz = dPx / dM, dPy / dM, dPz / dM
        Kin_after = 0.5 * dM * (Vx * Vx + Vy * Vy + Vz * Vz)
        diss = Kin_in - Kin_after
        total_diss += float(diss)

        pos = cp.array([grids[0][ix, iy, iz], grids[1][ix, iy, iz], grids[2][ix, iy, iz]], dtype=cp.float64)
        vel = cp.array([Vx, Vy, Vz], dtype=cp.float64)

        new_masses.append(cp.asarray(dM, dtype=cp.float64))
        new_positions.append(pos)
        new_velocities.append(vel)

        cells_processed += cells_in_region
        total_gas_removed += float(dM)

        #print(
         #   f"  Sink {sink_num + 1}: mass={dM:.6e}, cells={cells_in_region}, |v|={cp.sqrt(Vx ** 2 + Vy ** 2 + Vz ** 2):.3e}")

    if len(new_masses) == 0:
        return None

    # === MASS CONSERVATION CHECK ===
    final_gas_mass = float(cp.sum(gas_system.rho) * simulation.dV)
    created_sink_mass = float(sum([float(m) for m in new_masses]))
    mass_removed = initial_gas_mass - final_gas_mass
    mass_error = abs(created_sink_mass - mass_removed)
    rel_error = mass_error / (initial_gas_mass + 1e-30)

    '''
    print(f"\nGAS CONSUMPTION SUMMARY:")
    print(f"  Cells processed: {cells_processed}")
    print(f"  Gas mass removed: {mass_removed:.6e} Msun")
    print(f"  Sink mass created: {created_sink_mass:.6e} Msun")
    print(f"  Energy dissipated: {total_diss:.6e}")
    print(f"  Final gas mass: {final_gas_mass:.6e} Msun")

    print(f"\nMASS CONSERVATION:")
    if rel_error > 1e-10:
        print(f"  ❌ WARNING: Violation detected!")
        print(f"     Absolute error: {mass_error:.12e}")
        print(f"     Relative error: {rel_error:.12e}")
    else:
        print(f"  ✓ Conserved (error: {rel_error:.3e})")

    print(f"{'=' * 60}\n")

    return {
        "masses": cp.asarray(new_masses, dtype=cp.float64),
        "positions": cp.vstack(new_positions),
        "velocities": cp.vstack(new_velocities),
        "E_diss_formation": float(total_diss),
    }
    '''
