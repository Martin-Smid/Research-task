"""GPU rotation curves in each component's own periodic, mass-weighted frame."""

import cupy as cp
import numpy as np


class RotationCurves:
    def __init__(self, simulation):
        self.simulation = simulation
        self.lower = cp.asarray([b[0] for b in simulation.boundaries])
        self.box = cp.asarray([b[1] - b[0] for b in simulation.boundaries])

    def baryon_data(self, component, center=None, axis=None):
        """Use particle positions, velocities and masses to find their own frame."""
        mass = cp.broadcast_to(cp.asarray(component.m_particle), (component.N,))
        return self._prepare(component.positions, component.velocities, mass, center, axis)

    def gas_data(self, component, center=None, axis=None):
        """Use grid-cell centers, gas velocities and rho*dV as cell masses."""
        if self.simulation.dim != 3:
            raise ValueError("Rotation curves require a 3D simulation")
        positions = cp.stack([cp.asarray(g).ravel() for g in self.simulation.grids], axis=1)
        velocities = cp.stack([v.ravel() for v in component.vel], axis=1)
        mass = component.rho.ravel() * self.simulation.dV
        return self._prepare(positions, velocities, mass, center, axis)

    def _prepare(self, positions, velocities, mass, center, axis):
        if self.simulation.dim != 3:
            raise ValueError("Rotation curves require a 3D simulation")
        valid = cp.all(cp.isfinite(positions), axis=1) & cp.all(cp.isfinite(velocities), axis=1)
        valid &= cp.isfinite(mass) & (mass > 0)
        positions, velocities, mass = positions[valid], velocities[valid], mass[valid]
        frame = {"status": "empty", "center": np.full(3, np.nan),
                 "bulk_velocity": np.full(3, np.nan), "axis": np.full(3, np.nan),
                 "axis_coherence": 0.0}
        if mass.size == 0:
            return None, frame
        total_mass = cp.sum(mass)
        if center is None:
            # Circular mean locates a periodic image; unwrap there for the center of mass.
            angles = 2 * cp.pi * (positions - self.lower) / self.box
            moment = cp.sum(mass[:, None] * cp.exp(1j * angles), axis=0) / total_mass
            seed = self.lower + cp.mod(cp.angle(moment), 2 * cp.pi) * self.box / (2 * cp.pi)
            offset = cp.mod(positions - seed + self.box / 2, self.box) - self.box / 2
            center = seed + cp.sum(mass[:, None] * offset, axis=0) / total_mass
            center_ok = bool(cp.all(cp.abs(moment) > 1e-6))
        else:
            center = cp.asarray(center, dtype=cp.float64)
            if center.shape != (3,) or not bool(cp.all(cp.isfinite(center))):
                raise ValueError("center must contain three finite coordinates")
            center_ok = True
        center = self.lower + cp.mod(center - self.lower, self.box)
        positions = cp.mod(positions - center + self.box / 2, self.box) - self.box / 2
        bulk_velocity = cp.sum(mass[:, None] * velocities, axis=0) / total_mass
        velocities = velocities - bulk_velocity
        angular_momentum = mass[:, None] * cp.cross(positions, velocities)
        net = cp.sum(angular_momentum, axis=0)
        norm = cp.linalg.norm(net)
        scale = cp.sum(cp.linalg.norm(angular_momentum, axis=1))
        coherence = float(norm / cp.maximum(scale, 1e-300))
        if axis is None:
            axis_ok = bool(norm > 0) and coherence > 1e-6
            axis = net / norm if axis_ok else cp.full(3, cp.nan)
        else:
            axis = cp.asarray(axis, dtype=cp.float64)
            if axis.shape != (3,) or not bool(cp.all(cp.isfinite(axis))) or float(cp.linalg.norm(axis)) == 0:
                raise ValueError("axis must be a finite non-zero three-vector")
            axis = axis / cp.linalg.norm(axis)
            axis_ok = True
        frame.update(center=cp.asnumpy(center), bulk_velocity=cp.asnumpy(bulk_velocity),
                     axis=cp.asnumpy(axis), axis_coherence=coherence)
        frame["status"] = "ok" if center_ok and axis_ok else (
            "center_ambiguous" if not center_ok else "axis_undefined")
        return (positions, velocities, mass, axis) if frame["status"] == "ok" else None, frame

    def calculate(self, data, nbins=40, rmax=None, zmax=None, mass_fraction=0.5, mass_weighted=True):
        """Bin azimuthal velocities; rmax defaults to projected half-mass radius.

        The mass radius uses the whole component before the height cut. All large
        arrays, sorting and reductions stay on the GPU; only the binned output returns.
        """
        if not isinstance(nbins, int) or nbins < 1:
            raise ValueError("nbins must be a positive integer")
        if mass_fraction is not None and not 0 < mass_fraction <= 1:
            raise ValueError("mass_fraction must be in (0, 1] or None")
        if rmax is not None and (not np.isfinite(rmax) or rmax <= 0):
            raise ValueError("rmax must be positive and finite")
        if zmax is not None and (not np.isfinite(zmax) or zmax < 0):
            raise ValueError("zmax must be non-negative and finite")
        samples, frame = data
        frame = dict(frame, rmax=np.nan, zmax=zmax, mass_fraction=mass_fraction, enclosed_mass_fraction=0.0)
        if samples is None:
            return *(np.array([]) for _ in range(4)), frame
        positions, velocities, mass, axis = samples
        height = positions @ axis
        radial = positions - height[:, None] * axis
        radius = cp.linalg.norm(radial, axis=1)
        cutoff = float(cp.min(self.box)) / 2
        if mass_fraction is not None:
            order = cp.argsort(radius)
            cumulative = cp.cumsum(mass[order])
            index = min(int(cp.searchsorted(cumulative, mass_fraction * cumulative[-1])), mass.size - 1)
            cutoff = min(cutoff, float(radius[order[index]]))
        if rmax is not None:
            cutoff = min(cutoff, rmax)
        frame.update(rmax=cutoff, enclosed_mass_fraction=float(cp.sum(mass[radius <= cutoff]) / cp.sum(mass)))
        if cutoff <= 0:
            frame["status"] = "zero_radius"
            return *(np.array([]) for _ in range(4)), frame
        selected = (radius > 0) & (radius <= cutoff)
        if zmax is not None:
            selected &= cp.abs(height) <= zmax
        r = radius[selected]
        vphi = cp.sum(cp.cross(axis, radial[selected]) * velocities[selected], axis=1) / r
        weights = mass[selected] if mass_weighted else cp.ones_like(r)
        # Include the rightmost edge rather than losing points exactly at the cutoff.
        bins = cp.minimum((r / cutoff * nbins).astype(cp.int64), nbins - 1)
        weight = cp.bincount(bins, weights=weights, minlength=nbins)
        mean = cp.bincount(bins, weights=weights * vphi, minlength=nbins) / cp.maximum(weight, 1e-300)
        variance = cp.bincount(bins, weights=weights * (vphi - mean[bins]) ** 2, minlength=nbins)
        sigma = cp.sqrt(variance / cp.maximum(weight, 1e-300))
        mean[weight == 0] = cp.nan
        sigma[weight == 0] = cp.nan
        centers = (cp.arange(nbins) + 0.5) * cutoff / nbins
        return *(cp.asnumpy(a) for a in (centers, mean, sigma, weight)), frame
