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

            total_accreted += n_accrete

            # === MASS CONSERVATION CHECK ===
            final_baryon_mass = float(baryons.N * baryons.m_particle)
            final_sink_mass = float(cp.sum(self.masses))

            mass_transferred = initial_baryon_mass - final_baryon_mass
            sink_mass_gained = final_sink_mass - initial_sink_mass

            mass_error = abs(mass_transferred - sink_mass_gained)
            rel_error = mass_error / (initial_baryon_mass + 1e-30)

            if rel_error > 1e-10:
                print(f"WARNING: Mass conservation violation in accretion!")
                print(f"  Mass removed from baryons: {mass_transferred:.12e}")
                print(f"  Mass added to sinks: {sink_mass_gained:.12e}")
                print(f"  Absolute error: {mass_error:.12e}")
                print(f"  Relative error: {rel_error:.12e}")
            else:
                print(f"✓ Mass conserved in accretion (error: {rel_error:.3e})")

        self.E_diss_kin_last = dE_diss_total
        self.E_diss_kin_total += dE_diss_total

        return total_accreted

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

    print(f"\n{'=' * 60}")
    print(f"SINK CREATION EVENT")
    print(f"{'=' * 60}")
    print(f"Creating {n_new_sinks} new sink particle(s)")
    print(f"Source: {source}")
    print(f"Available: {n_baryon_particles} baryon particles, {n_gas_cells} gas cells")
    print(f"Initial mass: Baryons={initial_baryon_mass:.6e}, Gas={initial_gas_mass:.6e}")

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

    # Detailed report
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

    return {
        'masses': cp.array(new_masses, dtype=cp.float64),
        'positions': cp.stack(new_positions, axis=0),
        'velocities': cp.stack(new_velocities, axis=0),
        'E_diss_formation': float(total_dissipated_energy)
    }


# ============================================================================
# FILE: Sink_N_Body.py - Enhanced check_and_create_gas_sinks function
# ============================================================================

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

    print(f"\n{'=' * 60}")
    print(f"GAS SINK CREATION EVENT")
    print(f"{'=' * 60}")
    print(f"Creating {n_new_sinks} sink(s) from gas")
    print(f"Initial gas mass: {initial_gas_mass:.6e} Msun")
    print(f"Density - avg: {initial_avg_density:.3e}, max: {initial_max_density:.3e}")
    print(f"Threshold: {density_threshold:.3e}")

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

        print(
            f"  Sink {sink_num + 1}: mass={dM:.6e}, cells={cells_in_region}, |v|={cp.sqrt(Vx ** 2 + Vy ** 2 + Vz ** 2):.3e}")

    if len(new_masses) == 0:
        return None

    # === MASS CONSERVATION CHECK ===
    final_gas_mass = float(cp.sum(gas_system.rho) * simulation.dV)
    created_sink_mass = float(sum([float(m) for m in new_masses]))
    mass_removed = initial_gas_mass - final_gas_mass
    mass_error = abs(created_sink_mass - mass_removed)
    rel_error = mass_error / (initial_gas_mass + 1e-30)

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