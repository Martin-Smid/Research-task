# plot_max_trajectory.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def _find_latest_sim_dir(root="resources/data"):
    """
    Finds the most recent 'simulation_*' directory under root.
    """
    sims = sorted(glob.glob(os.path.join(root, "simulation_*")), key=os.path.getmtime)
    if not sims:
        raise FileNotFoundError(f"No simulation_* directory found in {root!r}.")
    return sims[-1]

def _load_max_locations(sim_dir):
    """
    Loads max_locations.txt with columns:
    time, ix, iy, iz, x, y, z
    """
    path = os.path.join(sim_dir, "max_locations.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Did you call record_max_location during evolution?")
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data[None, :]  # handle single-row file
    if data.shape[1] < 7:
        raise ValueError("max_locations.txt does not have 7 columns (time, ix, iy, iz, x, y, z).")
    t, ix, iy, iz, x, y, z = (data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                              data[:, 4], data[:, 5], data[:, 6])
    return dict(t=t, ix=ix, iy=iy, iz=iz, x=x, y=y, z=z, path=path)

def plot_max_trajectory_2d(sim_dir=None, plane="xy", coords="physical",
                           show=False, fname=None):
    """
    2D projection of the arg-max trajectory.

    Parameters
    ----------
    sim_dir : str or None
        Simulation directory containing 'max_locations.txt'.
        If None, the latest 'resources/data/simulation_*' is used.
    plane : {'xy','xz','yz'}
        Which projection to draw.
    coords : {'physical','index'}
        Use (x,y,z) if 'physical', else (ix,iy,iz).
    show : bool
        Whether to call plt.show().
    fname : str or None
        Output filename. If None, auto-generated in sim_dir.

    Returns
    -------
    out_path : str
        Saved PNG path.
    """
    if sim_dir is None:
        sim_dir = _find_latest_sim_dir()

    d = _load_max_locations(sim_dir)
    if coords == "physical":
        X, Y, Z = d["x"], d["y"], d["z"]
        unit_lbl = " (physical)"
    else:
        X, Y, Z = d["ix"], d["iy"], d["iz"]
        unit_lbl = " (index)"

    if plane == "xy":
        a, b = X, Y
        labels = ("X", "Y")
    elif plane == "xz":
        a, b = X, Z
        labels = ("X", "Z")
    elif plane == "yz":
        a, b = Y, Z
        labels = ("Y", "Z")
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    # Color by time
    t = d["t"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(a, b, lw=1.25, alpha=0.8)
    sc = ax.scatter(a, b, c=t, s=12)
    ax.scatter(a[0], b[0], s=40, marker="o", edgecolor="k", label="start")
    ax.scatter(a[-1], b[-1], s=40, marker="s", edgecolor="k", label="end")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("time")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(labels[0] + unit_lbl)
    ax.set_ylabel(labels[1] + unit_lbl)
    ax.set_title(f"Max-density trajectory ({plane.upper()} projection)")

    ax.legend(loc="best")
    plt.tight_layout()

    if fname is None:
        fname = f"max_trajectory_{plane}_{coords}.png"
    out_path = os.path.join(sim_dir, fname)
    plt.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return out_path

def plot_max_trajectory_3d(sim_dir=None, coords="physical",
                           show=False, fname=None):
    """
    3D curve of the arg-max trajectory.

    Parameters
    ----------
    sim_dir : str or None
        Simulation directory containing 'max_locations.txt'.
    coords : {'physical','index'}
        Use (x,y,z) if 'physical', else (ix,iy,iz).
    show : bool
        Whether to call plt.show().
    fname : str or None
        Output filename. If None, auto-generated in sim_dir.

    Returns
    -------
    out_path : str
        Saved PNG path.
    """
    if sim_dir is None:
        sim_dir = _find_latest_sim_dir()

    d = _load_max_locations(sim_dir)
    if coords == "physical":
        X, Y, Z = d["x"], d["y"], d["z"]
        unit_lbl = "[kpc]"
    else:
        X, Y, Z = d["ix"], d["iy"], d["iz"]
        unit_lbl = " (index)"

    t = d["t"]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X, Y, Z, lw=1.2, alpha=0.85)
    # scatter colored by time
    p = ax.scatter(X, Y, Z, c=t, s=10)
    cb = fig.colorbar(p, ax=ax, pad=0.1)
    cb.set_label("time [Gyr]")

    ax.set_xlabel("X" + unit_lbl)
    ax.set_ylabel("Y" + unit_lbl)
    ax.set_zlabel("Z" + unit_lbl)
    #ax.set_title("Max-density trajectory (3D)")

    # Equal-ish aspect
    mins = np.array([X.min(), Y.min(), Z.min()])
    maxs = np.array([X.max(), Y.max(), Z.max()])
    centers = 0.5 * (mins + maxs)
    span = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - span, centers[0] + span)
    ax.set_ylim(centers[1] - span, centers[1] + span)
    ax.set_zlim(centers[2] - span, centers[2] + span)

    plt.tight_layout()
    if fname is None:
        fname = f"max_trajectory_3d_{coords}.png"
    out_path = os.path.join(sim_dir, fname)
    plt.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    # Example usage: will auto-pick the newest simulation_* dir
    p2d = plot_max_trajectory_2d(sim_dir=None, plane="xy", coords="physical", show=False)
    p3d = plot_max_trajectory_3d(sim_dir=None, coords="physical", show=False)
    print("Saved:", p2d)
    print("Saved:", p3d)
