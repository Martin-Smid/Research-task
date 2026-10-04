"""Compatibility entry point for the shared rotation-curve plotting script."""
from plot_rot_curves import main, plot_rotation_curves as _plot_rotation_curves


def plot_rotation_curves(directory, time=None, components=None, output=None, km_s=False):
    return _plot_rotation_curves(directory, time=time, components=components,
                                output=output, km_s=km_s, show=False)


if __name__ == "__main__":
    main()