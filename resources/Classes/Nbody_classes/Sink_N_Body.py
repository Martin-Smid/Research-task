import cupy as cp
import numpy as np
from resources.Classes.Nbody_classes.NBody import NBody


class SinkNBody(NBody):
    """
    Supermassive black hole sink particle system.

    Each sink is a softened, accreting point mass that:
    - Deposits mass to grid for gravity calculation
    - Accretes nearby baryonic particles within capture radius
    - Conserves momentum during accretion
    - Uses Plummer softening for gravity
    """

    def __init__(self, simulation, N_sinks, initial_masses,
                 initial_positions, initial_velocities,
                 capture_radius=None, softening_length=None,
                 reservoir_tau=10):
        """
        Initialize sink particle system.

        Parameters
        ----------
        simulation : Simulation object
            Main simulation containing grid info
        N_sinks : int
            Number of sink particles
        initial_masses : cp.ndarray (N_sinks,)
            Initial mass of each sink
        initial_positions : cp.ndarray (N_sinks, 3)
            Initial positions
        initial_velocities : cp.ndarray (N_sinks, 3)
            Initial velocities
        capture_radius : float, optional
            Radius within which particles are accreted.
            Default: 2.5 * grid spacing
        softening_length : float, optional
            Plummer softening length for gravity.
            Default: 1.5 * grid spacing
        """
        # Initialize base NBody with dummy total mass
        super().__init__(simulation, N_sinks, total_mass=1.0)

        # Override particle mass with individual tracking
        self.positions = cp.asarray(initial_positions, dtype=cp.float64)
        self.velocities = cp.asarray(initial_velocities, dtype=cp.float64)

        # Set capture and softening radii
        min_dx = min(simulation.dx)
        self.capture_radius = capture_radius if capture_radius is not None else 2.5 * min_dx
        self.softening_length = softening_length if softening_length is not None else 1.5 * min_dx

        self.mass_bh = cp.asarray(initial_masses, dtype=cp.float64)          # gravitating BH mass component
        self.mass_res = cp.zeros_like(self.mass_bh)                          # unresolved reservoir mass
        self.masses = self.mass_bh + self.mass_res    

        self.reservoir_tau = float(reservoir_tau) if reservoir_tau is not None else 10.0 * simulation.h

        # Accretion tracking
        self.total_accreted_mass = 0.0
        self.accretion_history = []

        self.E_diss_kin_total = 0.0
        self.E_diss_kin_last = 0.0

    def deposit_to_grid(self):
        """
        Deposit sink masses to grid using CIC.
        Each sink deposits its individual mass.
        """
        import cupyx

        sim = self.simulation
        N = sim.N
        dim = sim.dim

        shape = (N,) * dim
        rho_grid = cp.zeros(shape, dtype=cp.float64)

        # Boundaries
        (x_min, x_max) = sim.boundaries[0]
        (y_min, y_max) = sim.boundaries[1]
        (z_min, z_max) = sim.boundaries[2]

        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

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

        tx = rx - i0
        ty = ry - j0
        tz = rz - k0

        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N

        i0 = i0 % N
        j0 = j0 % N
        k0 = k0 % N

        cell_volume = sim.cell_volume

        # 8 corners with individual sink masses
        corners_weights = [
            (wx0 * wy0 * wz0, i0, j0, k0),
            (wx1 * wy0 * wz0, i1, j0, k0),
            (wx0 * wy1 * wz0, i0, j1, k0),
            (wx1 * wy1 * wz0, i1, j1, k0),
            (wx0 * wy0 * wz1, i0, j0, k1),
            (wx1 * wy0 * wz1, i1, j0, k1),
            (wx0 * wy1 * wz1, i0, j1, k1),
            (wx1 * wy1 * wz1, i1, j1, k1),
        ]

        N_sq = N * N
        rho_flat = rho_grid.ravel()

        for w, ii, jj, kk in corners_weights:
            flat_indices = ii * N_sq + jj * N + kk
            # Use individual sink masses
            contribution = (self.masses / cell_volume) * w
            cupyx.scatter_add(rho_flat, flat_indices, contribution)

        return rho_grid

    def accrete_from_baryons(self, baryon_systems):
        """
        Capture particles from baryonic systems based on escape velocity and angular momentum criteria.

        Captured mass is added to a subgrid reservoir (mass_res) and does NOT instantly increase BH mass (mass_bh).
        Total gravitating mass (masses = mass_bh + mass_res) is conserved, and sink momentum is updated.

        Parameters
        ----------
        baryon_systems : list of Baryons
            Baryonic N-body systems to check for accretion

        Returns
        -------
        int : Total number of particles captured
        """
        total_accreted = 0
        dE_diss_call = cp.asarray(0.0, dtype=cp.float64)
        self.E_diss_kin_last = 0.0

        for baryons in baryon_systems:
            # Skip if it's a sink system or has no particles
            if isinstance(baryons, SinkNBody) or baryons.N == 0:
                continue

            # Check each sink
            for sink_idx in range(self.N):
                sink_pos = self.positions[sink_idx]
                sink_vel = self.velocities[sink_idx]
                sink_mass_total = self.masses[sink_idx]   # total gravitating mass = BH + reservoir

                # 1) Compute distances (periodic)
                dx = baryons.positions - sink_pos[None, :]
                for d in range(3):
                    low, high = self.simulation.boundaries[d]
                    L = high - low
                    dx[:, d] = dx[:, d] - cp.copysign(L, dx[:, d]) * (cp.abs(dx[:, d]) > L / 2)

                dist = cp.sqrt(cp.sum(dx**2, axis=1))

                # Find particles within capture radius
                capture_mask = dist < self.capture_radius
                if not cp.any(capture_mask):
                    continue

                # Escape velocity criterion: v_esc = sqrt(2 * G * M / r_soft)
                G = self.simulation.G
                rel_vel = baryons.velocities - sink_vel[None, :]
                v_rel_mag = cp.sqrt(cp.sum(rel_vel**2, axis=1))

                r_soft = cp.maximum(dist, self.softening_length)
                v_esc = cp.sqrt(2 * G * sink_mass_total / r_soft)

                # Angular momentum criterion
                Lx = dx[:, 1] * rel_vel[:, 2] - dx[:, 2] * rel_vel[:, 1]
                Ly = dx[:, 2] * rel_vel[:, 0] - dx[:, 0] * rel_vel[:, 2]
                Lz = dx[:, 0] * rel_vel[:, 1] - dx[:, 1] * rel_vel[:, 0]
                L_sq = Lx**2 + Ly**2 + Lz**2

                # Max allowed L for circular orbit at R=capture_radius:
                # L_max^2 = G * M * R
                L_max_sq = G * sink_mass_total * self.capture_radius
                ang_mom_condition = L_sq < L_max_sq

                # Must be close enough AND bound AND low enough angular momentum
                accrete_mask = capture_mask & (v_rel_mag < v_esc) & ang_mom_condition

                n_accrete = int(cp.sum(accrete_mask))
                if n_accrete == 0:
                    continue

                # ---- Capture bookkeeping (mass -> reservoir, momentum conserved) ----
                accreted_mass = baryons.m_particle * n_accrete
                accreted_momentum = cp.sum(baryons.velocities[accrete_mask] * baryons.m_particle, axis=0)

                # Momentum conservation for the sink's bulk motion
                M_old = sink_mass_total
                P_old = M_old * sink_vel
                M_new = M_old + accreted_mass
                P_new = P_old + accreted_momentum


                V_old = sink_vel.copy()
                v_acc = baryons.velocities[accrete_mask]
                m_p = float(baryons.m_particle)

                K_before = 0.5 * M_old * cp.sum(V_old * V_old) + 0.5 * m_p * cp.sum(v_acc * v_acc)
                V_new = P_new / M_new
                K_after  = 0.5 * M_new * cp.sum(V_new * V_new)

                dE_diss_call += (K_before - K_after)

                # Captured mass goes to reservoir (BH mass stays unchanged here)
                self.mass_res[sink_idx] += accreted_mass

                # Update total gravitating mass (keep old interface)
                self.masses[sink_idx] = self.mass_bh[sink_idx] + self.mass_res[sink_idx]

                # Update sink velocity
                self.velocities[sink_idx] = P_new / M_new

                # Remove captured particles (keep non-captured)
                keep_mask = ~accrete_mask
                baryons.positions = baryons.positions[keep_mask]
                baryons.velocities = baryons.velocities[keep_mask]
                baryons.N = int(cp.sum(keep_mask))

                total_accreted += n_accrete
                self.total_accreted_mass += accreted_mass
                self.accretion_history.append((int(sink_idx), float(accreted_mass)))

                self.E_diss_kin_last = float(dE_diss_call.get())
                self.E_diss_kin_total += self.E_diss_kin_last

        return total_accreted


    def add_new_sink(self, mass, position, velocity):
        """
        Add a new sink particle to this system.

        Parameters
        ----------
        mass : float
            Mass of new sink
        position : cp.ndarray (3,)
            Position of new sink
        velocity : cp.ndarray (3,)
            Velocity of new sink
        """
        # Append to existing arrays
        self.masses = cp.append(self.masses, mass)
        self.positions = cp.vstack([self.positions, position[None, :]])
        self.velocities = cp.vstack([self.velocities, velocity[None, :]])
        self.N += 1
        self.mass_bh = cp.concatenate([self.mass_bh, cp.asarray([mass], dtype=cp.float64)])
        self.mass_res = cp.concatenate([self.mass_res, cp.asarray([0.0], dtype=cp.float64)])
        self.masses = self.mass_bh + self.mass_res


    @staticmethod
    def merge_or_create(existing_sink_system, new_sinks_data, simulation):
        """
        Merge new sink data into existing system or create new one.

        Parameters
        ----------
        existing_sink_system : SinkNBody or None
            Existing sink system (if any)
        new_sinks_data : dict
            Dictionary with 'masses', 'positions', 'velocities' arrays
        simulation : Simulation object
            Main simulation

        Returns
        -------
        SinkNBody
            Updated or new sink system
        """
        print("got to merge or create")
        if new_sinks_data is None:
            return existing_sink_system

        n_new = len(new_sinks_data['masses'])
        if n_new == 0:
            return existing_sink_system

        if existing_sink_system is None:
            # Create new system
            return SinkNBody(
                simulation=simulation,
                N_sinks=n_new,
                initial_masses=new_sinks_data['masses'],
                initial_positions=new_sinks_data['positions'],
                initial_velocities=new_sinks_data['velocities']
            )
        else:
            # Add to existing system
            for i in range(n_new):
                existing_sink_system.add_new_sink(
                    mass=new_sinks_data['masses'][i],
                    position=new_sinks_data['positions'][i],
                    velocity=new_sinks_data['velocities'][i]
                )
            return existing_sink_system

    def kinetic_energy(self):
        """
        Variable per-sink masses.
        """
        v2 = (self.velocities ** 2).sum(axis=1)  # |v|^2 per sink
        return 0.5 * (self.masses * v2).sum()

    def drain_reservoir(self, dt):
        """
        Move mass from reservoir -> BH smoothly.
        This changes BH mass but does NOT change total gravitating mass (BH+res stays constant).
        """
        if self.N == 0:
            return

        tau = self.reservoir_tau
        if tau <= 0:
            # instant drain
            dM = self.mass_res.copy()
        else:
            # exponential drain fraction per step: dM = M_res * (1 - exp(-dt/tau))
            frac = 1.0 - cp.exp(-float(dt) / float(tau))
            dM = self.mass_res * frac

        # transfer mass internally
        self.mass_res -= dM
        self.mass_bh += dM
        print(f"Drained reservoir: {dM.sum()} from reservoir")
        # keep old interface consistent
        self.masses = self.mass_bh + self.mass_res

class SinkFormationTracker:
    """
    Tracks density threshold violations for sink formation.
    Requires consecutive threshold exceedances before creating a sink.
    """

    def __init__(self, grid_shape, consecutive_steps_required=5):
        """
        Parameters
        ----------
        grid_shape : tuple
            Shape of density grid (N, N, N)
        consecutive_steps_required : int
            Number of consecutive steps density must exceed threshold
        """
        self.grid_shape = grid_shape
        self.consecutive_steps_required = consecutive_steps_required

        # Counter for each grid cell
        self.threshold_counter = cp.zeros(grid_shape, dtype=cp.int32)

        # Track which cells already have sinks
        self.has_sink = cp.zeros(grid_shape, dtype=cp.bool_)

    def update_and_check(self, baryonic_density, threshold):
        """
        Update counters and return cells ready for sink formation.

        Parameters
        ----------
        baryonic_density : cp.ndarray
            Current baryonic density field
        threshold : float
            Density threshold for sink formation

        Returns
        -------
        cp.ndarray or None
            Indices of cells ready for sink formation, or None
        """
        # Check which cells exceed threshold
        exceeds = baryonic_density > threshold

        # Increment counter where threshold exceeded
        self.threshold_counter = cp.where(exceeds,
                                          self.threshold_counter + 1,
                                          0)

        # Find cells that have exceeded threshold for required consecutive steps
        # AND don't already have a sink
        ready_for_sink = (self.threshold_counter >= self.consecutive_steps_required) & (~self.has_sink)

        if not cp.any(ready_for_sink):
            return None

        # Mark these cells as having sinks
        self.has_sink = self.has_sink | ready_for_sink

        # Return indices
        indices = cp.argwhere(ready_for_sink)
        return indices
    
    


    def mark_sink_region(self, position, radius):
        """
        Mark a region around a sink position to prevent duplicate formation.

        Parameters
        ----------
        position : cp.ndarray (3,)
            Position of new sink
        radius : float
            Radius around sink to mark
        """
        # For now, just mark the cell itself
        pass



def check_and_create_sinks(simulation, sink_tracker, density_threshold,
                           aggregation_radius=None):
    """
    Check for sink formation conditions and create new sinks if criteria met.
    Called from within the evolution loop.

    Parameters
    ----------
    simulation : Simulation object
        Main simulation
    sink_tracker : SinkFormationTracker
        Tracker for consecutive threshold violations
    density_threshold : float
        Critical density for sink formation
    aggregation_radius : float, optional
        Radius for aggregating particles into new sink

    Returns
    -------
    dict or None
        Dictionary with new sink data, or None if no sinks formed
    """
    # Compute baryonic density only (exclude ULDM and existing sinks)
    shape = (simulation.N,) * simulation.dim
    baryonic_density = cp.zeros(shape, dtype=cp.float64)


    regular_baryons = []
    for baryons in simulation.baryonic_matter:
        if not isinstance(baryons, SinkNBody) and baryons.N > 0:
            baryonic_density += baryons.deposit_to_grid()
            regular_baryons.append(baryons)

    # Check for cells ready for sink formation
    ready_indices = sink_tracker.update_and_check(baryonic_density, density_threshold)

    if ready_indices is None:
        return None

    n_new_sinks = ready_indices.shape[0]
    print(f"Creating {n_new_sinks} new sink particle(s)")

    # Setup
    grids = simulation.grids
    min_dx = min(simulation.dx)
    if aggregation_radius is None:
        aggregation_radius = 1.0 * min_dx

    new_masses = []
    new_positions = []
    new_velocities = []

    for cell_idx in ready_indices:
        ix, iy, iz = int(cell_idx[0]), int(cell_idx[1]), int(cell_idx[2])

        # Cell center position
        cell_pos = cp.array([
            grids[0][ix, iy, iz],
            grids[1][ix, iy, iz],
            grids[2][ix, iy, iz]
        ], dtype=cp.float64)

        # Aggregate mass and momentum from nearby particles
        total_mass = 0.0
        total_momentum = cp.zeros(3, dtype=cp.float64)

        for baryons in regular_baryons:
            if baryons.N == 0:
                continue

            # Distance to cell center (periodic)
            dx_vec = baryons.positions - cell_pos[None, :]
            for d in range(3):
                low, high = simulation.boundaries[d]
                L = high - low
                dx_vec[:, d] = dx_vec[:, d] - cp.copysign(L, dx_vec[:, d]) * (cp.abs(dx_vec[:, d]) > L / 2)

            dist = cp.sqrt(cp.sum(dx_vec ** 2, axis=1))

            # Particles within aggregation radius
            nearby_mask = dist < aggregation_radius
            n_nearby = int(cp.sum(nearby_mask))

            if n_nearby > 0:
                mass_nearby = baryons.m_particle * n_nearby
                momentum_nearby = cp.sum(baryons.velocities[nearby_mask] * baryons.m_particle, axis=0)

                total_mass += mass_nearby
                total_momentum += momentum_nearby

                # Remove aggregated particles
                keep_mask = ~nearby_mask
                baryons.positions = baryons.positions[keep_mask]
                baryons.velocities = baryons.velocities[keep_mask]
                baryons.N = int(cp.sum(keep_mask))

        # Only create sink if we aggregated some mass
        if total_mass > 0:
            new_masses.append(total_mass)
            new_positions.append(cell_pos)
            new_velocities.append(total_momentum / total_mass)

    if len(new_masses) == 0:
        return None

    return {
        'masses': cp.array(new_masses, dtype=cp.float64),
        'positions': cp.stack(new_positions, axis=0),
        'velocities': cp.stack(new_velocities, axis=0)
    }


