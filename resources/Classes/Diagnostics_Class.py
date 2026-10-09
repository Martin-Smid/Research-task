"""Diagnostics calculated from the current simulation state; writing stays in Scribe."""

import cupy as cp
import numpy as np
from resources.Classes.Rotation_Curve_Class import RotationCurves

class Diagnostics:
    def __init__(self, simulation, propagator, sink_potential):
        """Keep live state references and a provider for the current sink potential."""
        self.simulation = simulation
        self.propagator = propagator
        self.sink_potential = sink_potential
        self.last_kinetic_energy = 0
        self.rotation_curves = RotationCurves(simulation)

    def compute_energy(self, wave_functions, total_density, current_time):
        """Return energy components; Evolution passes them to Scribe."""
        K_flow, U_quantum, K_baryons = self._compute_kinetic_energy(wave_functions)
        K_total = K_flow + U_quantum + K_baryons

        W_self, W_static, W_total = self._compute_potential_energy(
            wave_functions, total_density, current_time
        )

        U_iso_total = 0.0
        E_diss_total = 0.0

        if hasattr(self.simulation, 'baryonic_matter') and self.simulation.baryonic_matter:
            for sys in self.simulation.baryonic_matter:
                if self._is_sink_system(sys):
                    # assume these are cumulative totals
                    E_diss_total += float(getattr(sys, 'E_diss_kin_total', 0.0))
                    E_diss_total += float(getattr(sys, 'E_diss_formation_total', 0.0))

                # Preserve the legacy gas accumulation during this extraction.
                for sys in self.simulation.baryonic_matter:
                    if sys.__class__.__name__.lower().endswith("gas"):
                        if hasattr(sys, "internal_energy"):
                            U_iso_total += float(sys.internal_energy())

        return {
            "K_total": K_total, "W": W_total, "U_iso": U_iso_total,
            "K_flow": K_flow, "U_quantum": U_quantum, "K_baryons": K_baryons,
            "W_self": W_self, "W_static": W_static, "E_diss": E_diss_total,
        }

    def _compute_potential_energy(self, wave_functions, total_density, current_time):
        """
        Compute potential energy with the existing normalization and cross-terms.
        rho_w denotes all supplied non-sink density; rho_s is deposited sink density.

        W = 1/2 ∫ rho_w * phi_w dV
          + 1/2 ∫ rho_s * phi_s dV
          + 1/2 ∫ (rho_w * phi_s + rho_s * phi_w) dV
          +     ∫ (rho_w + rho_s) * phi_static dV  (if any)
        """
        dV = float(self.simulation.dV)

        rho_w = total_density  # Waves and non-sink baryons, as supplied by Evolution.
        phi_w = self.propagator.compute_gravity_potential(rho_w)
        phi_s = self.sink_potential()  # analytic sinks (all sinks)

        # sinks deposited on grid (for energy integrals)
        rho_s = cp.zeros_like(rho_w)
        if getattr(self.simulation, "baryonic_matter", None):
            for sys in self.simulation.baryonic_matter:
                if self._is_sink_system(sys):
                    rho_s += sys.deposit_to_grid()

        # self terms
        W_w_self = 0.5 * cp.sum(rho_w * phi_w) * dV
        W_s_self = 0.5 * cp.sum(rho_s * phi_s) * dV

        W_cross = 0.5 * (cp.sum(rho_w * phi_s) + cp.sum(rho_s * phi_w)) * dV

        W_static = 0.0
        if self.simulation.static_potential is not None:
            phi_static = self.simulation.static_potential(self.simulation)
            W_static = cp.sum((rho_w + rho_s) * cp.real(phi_static)) * dV

        W_total = cp.real(W_w_self + W_s_self + W_cross + W_static)

        self.W_self = cp.real(W_w_self + W_s_self + W_cross)  # pokud chceš mít "self" jako všechno bez static
        self.W_static = cp.real(W_static)

        return self.W_self, self.W_static, W_total

    def _compute_kinetic_energy(self, wave_functions):
        """Compute kinetic energy: flow + quantum (waves) + baryons."""
        dx = np.prod(self.simulation.dx)
        k_space = self.simulation.k_space

        K_flow_total = 0.0
        U_quantum_total = 0.0

        # --- wavefunctions ---
        for wf in wave_functions:
            rho = wf.calculate_density()  # |ψ|²
            sqrt_rho = cp.sqrt(rho + 1e-30)  # Add small value to avoid division by zero

            # Compute gradients in Fourier space
            psi_k = cp.fft.fftn(wf.psi)
            sqrt_rho_k = cp.fft.fftn(sqrt_rho)

            grad_psi_squared = cp.zeros_like(rho, dtype=cp.float64)
            grad_sqrt_rho_squared = cp.zeros_like(rho, dtype=cp.float64)

            # Loop over spatial dimensions to compute gradients
            for dim in range(self.simulation.dim):
                # Gradient of psi: ∇ψ = iFFT(ik * FFT(ψ))
                grad_psi_dim = cp.fft.ifftn(1j * k_space[dim] * psi_k)
                grad_psi_squared += cp.abs(grad_psi_dim) ** 2

                # Gradient of sqrt(rho): ∇√ρ = iFFT(ik * FFT(√ρ))
                grad_sqrt_rho_dim = cp.fft.ifftn(1j * k_space[dim] * sqrt_rho_k)
                grad_sqrt_rho_squared += cp.abs(grad_sqrt_rho_dim) ** 2

            # U_quantum = (ℏ²/2m) ∫ |∇√ρ|² dV
            U_quantum = 0.5 * self.simulation.h_bar_tilde ** 2 * cp.sum(grad_sqrt_rho_squared) * dx

            # K_total_this_wf = (ℏ²/2m) ∫ |∇ψ|² dV
            K_total_this_wf = 0.5 * self.simulation.h_bar_tilde ** 2 * cp.sum(grad_psi_squared) * dx
            K_flow = K_total_this_wf - U_quantum

            K_flow_total += K_flow
            U_quantum_total += U_quantum

        # --- baryons: ½ m v² summed over ALL particles in ALL systems ---
        K_baryons = 0.0
        if getattr(self.simulation, 'baryonic_matter', None):
            for baryons in self.simulation.baryonic_matter:
                if hasattr(baryons, "kinetic_energy"):
                    K_baryons += baryons.kinetic_energy()

        # Store for possible simple logging
        self.K_flow = K_flow_total
        self.U_quantum = U_quantum_total
        self.K_baryons = K_baryons
        self.last_kinetic_energy = K_flow_total + U_quantum_total + K_baryons

        return K_flow_total, U_quantum_total, K_baryons

    def compute_radial_profile(self, total_density, ix, iy, iz, Nbins=250):
        """Compute and save the spherically averaged radial density profile."""
        BoxSize = [b[1] - b[0] for b in self.simulation.boundaries]
        rho = total_density

        grid_x = cp.asarray(self.simulation.grids[0])
        grid_y = cp.asarray(self.simulation.grids[1])
        grid_z = cp.asarray(self.simulation.grids[2])

        center_x = grid_x[ix, iy, iz]
        center_y = grid_y[ix, iy, iz]
        center_z = grid_z[ix, iy, iz]

        # Compute shifted periodic coordinates
        Delta_x = grid_x - center_x
        Delta_y = grid_y - center_y
        Delta_z = grid_z - center_z

        Delta_x -= cp.copysign(BoxSize[0], Delta_x) * (cp.abs(Delta_x) > BoxSize[0] / 2)
        Delta_y -= cp.copysign(BoxSize[1], Delta_y) * (cp.abs(Delta_y) > BoxSize[1] / 2)
        Delta_z -= cp.copysign(BoxSize[2], Delta_z) * (cp.abs(Delta_z) > BoxSize[2] / 2)

        r = cp.sqrt(Delta_x ** 2 + Delta_y ** 2 + Delta_z ** 2)

        # Transfer to CPU
        r_cpu = cp.asnumpy(r).ravel()
        rho_cpu = cp.asnumpy(rho).ravel()

        # Create radial bins
        max_radius = 0.95 * 0.5 * min(BoxSize)
        bins = np.concatenate(([0.0], np.geomspace(0.003, max_radius, Nbins)))

        # Compute mean density per bin in one pass: bin i holds bins[i] < r <= bins[i + 1]
        index = np.searchsorted(bins, r_cpu) - 1
        inside = (index >= 0) & (index < Nbins)
        counts = np.bincount(index[inside], minlength=Nbins)
        sums = np.bincount(index[inside], weights=rho_cpu[inside], minlength=Nbins)
        filled = counts > 0
        bin_centers = (0.5 * (bins[:-1] + bins[1:]))[filled]
        rho_avg = sums[filled] / counts[filled]

        return bin_centers, rho_avg

    def compute_rotation_curves(self, nbins=40, rmax=None, zmax_gas=1.0, zmax_stars=2.0,
                                mass_fraction=0.5):
        """Find each component's own frame and yield its GPU-computed rotation curve."""
        components = self.simulation.baryonic_matter
        names = [getattr(component, "name", f"baryons_{i}") for i, component in enumerate(components)]
        for i, component in enumerate(components):
            if self._is_sink_system(component):
                continue
            name = names[i] if names.count(names[i]) == 1 else f"{names[i]}_{i}"
            try:
                if self._is_gas_system(component):
                    data = self.rotation_curves.gas_data(component)
                    zmax = zmax_gas
                elif hasattr(component, "positions") and hasattr(component, "m_particle"):
                    data = self.rotation_curves.baryon_data(component)
                    zmax = zmax_stars
                else:
                    continue
                R, vphi, sigma, weights, frame = self.rotation_curves.calculate(
                    data, nbins, rmax, zmax, mass_fraction
                )
                yield dict(component_name=name, R_centers=R, vphi_mean=vphi,
                           vphi_std=sigma, weights=weights, frame=frame)
            except Exception as error:
                print(f"[Diagnostics] Warning: rotation curve for {name}: {error}")

    @staticmethod
    def _is_sink_system(component):
        return component.__class__.__name__.lower().startswith("sink")

    @staticmethod
    def _is_gas_system(component):
        return component.__class__.__name__.lower() == "nbodygas"
