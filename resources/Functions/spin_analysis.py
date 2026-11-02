"""Analysis helpers for (multi-component) ULDM density and wave-function fields; plain numpy, no GPU, no simulation object.

All functions take arrays (and a few numbers) and return numbers or arrays, so they can be used on any saved snapshot.
Fields are numpy arrays of shape (N, N, N); a spin-s vector is a list of 2s+1 such arrays ordered by m = -s ... s."""

import numpy as np


def spin_matrices(spin):
    """J_x, J_y, J_z in the m basis (m = -s ... s) of Wave_vector_class (Condon-Shortley phases, tested in test_spin_basis.py)."""
    m = np.arange(-spin, spin + 1)
    j_plus = np.zeros((2 * spin + 1,) * 2)
    for i in range(2 * spin):
        j_plus[i + 1, i] = np.sqrt(spin * (spin + 1) - m[i] * (m[i] + 1))
    return (j_plus + j_plus.T) / 2, (j_plus - j_plus.T) / (2j), np.diag(m).astype(float)


def spin_density(fields, spin):
    """(S_x, S_y, S_z) densities, sum_{m m'} psi_m^* (J_k)_{m m'} psi_m'."""
    stacked = np.stack(fields)
    return [np.real(np.einsum("m...,mn,n...->...", np.conj(stacked), J, stacked)) for J in spin_matrices(spin)]


def power_spectrum(delta):
    """Angular-averaged |delta_k|^2 / n^6 in shells of |k| = 1, 2, ... fundamental modes; returns (k, P)."""
    n = delta.shape[0]
    power = np.abs(np.fft.fftn(delta)) ** 2 / n ** 6
    k = np.fft.fftfreq(n, d=1.0 / n)
    kmag = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    shell = np.rint(kmag).astype(int)
    keep = (shell >= 1) & (shell <= n // 2)
    sums = np.bincount(shell[keep], weights=power[keep], minlength=n // 2 + 1)
    counts = np.bincount(shell[keep], minlength=n // 2 + 1)
    return np.arange(1, n // 2 + 1), sums[1:] / counts[1:]


def core_properties(rho, dx):
    """Central density and half-density radius around the density maximum (periodic distances), angular-averaged profile."""
    n = rho.shape[0]
    peak = np.unravel_index(np.argmax(rho), rho.shape)
    axes = [np.minimum(np.abs(np.arange(n) - peak[i]), n - np.abs(np.arange(n) - peak[i])) * dx for i in range(3)]
    r = np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2)
    edges = np.arange(0.0, n * dx / 2, dx)
    index = np.digitize(r.ravel(), edges) - 1
    profile = np.bincount(index, weights=rho.ravel(), minlength=len(edges))[:len(edges)] / np.maximum(np.bincount(index, minlength=len(edges))[:len(edges)], 1)
    below = np.nonzero(profile <= rho.max() / 2)[0]
    r_half = edges[below[0]] if len(below) else np.nan
    return rho.max(), r_half


def potential_phase(rho, box, G, h, h_bar_tilde):
    """max |Phi| h / h_bar_tilde with Phi from the periodic Poisson equation (the phase picked up by psi in one step)."""
    n = rho.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=2 * box / n)
    k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    k2[0, 0, 0] = 1.0
    delta_k = np.fft.fftn(rho - rho.mean())
    delta_k[0, 0, 0] = 0.0
    phi = np.real(np.fft.ifftn(-4 * np.pi * G * delta_k / k2))
    return np.abs(phi).max() * h / h_bar_tilde


def collapse_time(time_series, ratio, threshold):
    above = np.nonzero(ratio >= threshold)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return time_series[0]
    return time_series[i - 1] + (threshold - ratio[i - 1]) / (ratio[i] - ratio[i - 1]) * (time_series[i] - time_series[i - 1])
