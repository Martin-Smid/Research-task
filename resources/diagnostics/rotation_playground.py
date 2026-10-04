"""Analytical GPU checks: python rotation_playground.py (playground branch only)."""

from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chats_playground import cp, np
from resources.Classes.Rotation_Curve_Class import RotationCurves


def main():
    simulation = SimpleNamespace(dim=3, boundaries=[(-10.0, 10.0)] * 3, dV=2.5 ** 3)
    rotation = RotationCurves(simulation)
    axis = cp.asarray([1.0, 2.0, 3.0])
    axis /= cp.linalg.norm(axis)
    u = cp.cross(axis, cp.asarray([0.0, 0.0, 1.0]))
    u /= cp.linalg.norm(u)
    v = cp.cross(axis, u)
    angle = cp.tile(cp.arange(16) * 2 * cp.pi / 16, 4)
    radius = cp.repeat(cp.asarray([1.0, 2.0, 3.0, 4.0]), 16)
    relative = radius[:, None] * (cp.cos(angle)[:, None] * u + cp.sin(angle)[:, None] * v)
    velocity = 2 * (-cp.sin(angle)[:, None] * u + cp.cos(angle)[:, None] * v)
    center = cp.asarray([9.8, -9.7, 1.2])
    bulk = cp.asarray([4.0, -3.0, 2.0])
    component = SimpleNamespace(N=len(radius), m_particle=1.0,
                                positions=(relative + center + 10) % 20 - 10,
                                velocities=velocity + bulk)
    data = rotation.baryon_data(component)
    R, mean, sigma, weight, frame = rotation.calculate(data, nbins=5, zmax=0.1, mass_fraction=None)
    assert frame["status"] == "ok"
    np.testing.assert_allclose(frame["center"], cp.asnumpy(center), atol=1e-12)
    np.testing.assert_allclose(frame["axis"], cp.asnumpy(axis), atol=1e-12)
    np.testing.assert_allclose(frame["bulk_velocity"], cp.asnumpy(bulk), atol=1e-12)
    np.testing.assert_allclose(mean[weight > 0], 2, atol=1e-12)
    np.testing.assert_allclose(sigma[weight > 0], 0, atol=1e-12)
    assert weight.sum() == component.N
    print("PASS: tilted, moving particle disk crossing periodic boundaries")

    opposite = rotation.baryon_data(component, axis=-axis)
    _, reverse_mean, _, reverse_weight, _ = rotation.calculate(opposite, mass_fraction=None)
    np.testing.assert_allclose(reverse_mean[reverse_weight > 0], -2, atol=1e-12)
    print("PASS: fixed opposite axis preserves signed rotation")
    _, _, _, _, half_frame = rotation.calculate(data, mass_fraction=0.5)
    np.testing.assert_allclose(half_frame["rmax"], 2, atol=1e-12)
    assert half_frame["enclosed_mass_fraction"] >= 0.5
    print("PASS: projected half-mass radius")

    samples = (cp.asarray([[2., 0, 0], [-2., 0, 0]]),
               cp.asarray([[0., 3, 0], [0., -3, 0]]), cp.ones(2), cp.asarray([0., 0, 1]))
    _, mean, _, weight, _ = rotation.calculate((samples, frame), nbins=2, rmax=2, mass_fraction=None)
    assert weight[-1] == 2 and mean[-1] == 3
    print("PASS: exact outer edge included")

    samples = (cp.asarray([[1., 0, 0], [-1., 0, 0]]),
               cp.asarray([[0., 1, 0], [0., -3, 0]]), cp.asarray([1., 3.]), cp.asarray([0., 0, 1]))
    _, mean, sigma, weight, _ = rotation.calculate((samples, frame), nbins=1, mass_fraction=None)
    np.testing.assert_allclose(mean, [2.5])
    np.testing.assert_allclose(sigma, [np.sqrt(0.75)])
    np.testing.assert_allclose(weight, [4])
    print("PASS: mass-weighted mean and velocity dispersion")

    coordinates = cp.arange(8) * 2.5 - 10
    simulation.grids = cp.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    x, y, z = simulation.grids
    # Four rotating cells; translate the ring across the x boundary.
    dx = (x - 7.5 + 10) % 20 - 10
    rho = ((dx ** 2 + y ** 2 == 2.5 ** 2) & (z == 0)).astype(cp.float64)
    gas = SimpleNamespace(rho=rho, vel=[-2 * y + 4, 2 * dx - 3, cp.full_like(x, 2)])
    _, mean, sigma, weight, frame = rotation.calculate(rotation.gas_data(gas), nbins=1)
    np.testing.assert_allclose(frame["center"], [7.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(frame["axis"], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(mean, [5], atol=1e-12)
    np.testing.assert_allclose(sigma, [0], atol=1e-12)
    np.testing.assert_allclose(weight, [4 * simulation.dV])
    print("PASS: gas frame, bulk subtraction and rho*dV weights")

    gas.rho = cp.zeros_like(rho)
    *curves, frame = rotation.calculate(rotation.gas_data(gas))
    assert frame["status"] == "empty" and all(len(a) == 0 for a in curves)
    gas.rho = cp.ones_like(rho)
    *curves, frame = rotation.calculate(rotation.gas_data(gas))
    assert frame["status"] == "center_ambiguous" and all(len(a) == 0 for a in curves)
    print("PASS: empty gas and ambiguous periodic center flagged")
    component.velocities = cp.broadcast_to(bulk, component.positions.shape)
    data = rotation.baryon_data(component)
    *curves, frame = rotation.calculate(data)
    assert frame["status"] == "axis_undefined" and all(len(a) == 0 for a in curves)
    data = rotation.baryon_data(component, axis=[0, 0, 1])
    _, mean, _, weight, frame = rotation.calculate(data)
    assert frame["status"] == "ok"
    np.testing.assert_allclose(mean[weight > 0], 0, atol=1e-12)
    print("PASS: undefined rotation flagged; manual axis supported")


if __name__ == "__main__":
    main()
