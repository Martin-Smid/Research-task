# Rotation curves

Rotation curves measure mean azimuthal streaming velocity, with a mass-weighted
velocity dispersion. They do not calculate gravitational circular speed.

`Rotation_Curve_Class.py` has three public methods:
- `baryon_data`: particle positions, velocities and equal particle masses.
- `gas_data`: grid-cell positions, gas velocities and cell masses (`rho*dV`).
- `calculate`: one shared GPU binning calculation for either component.

Each component gets its own periodic center of mass, bulk velocity and rotation
axis from its net angular momentum. Velocities are measured after subtracting
bulk motion; distances use the nearest periodic image. A circular mean chooses
the periodic image before evaluating the ordinary center of mass. This assumes
one localized component, rather than identifying separate objects in multiple clumps.

The default cutoff is the cylindrical radius enclosing 50% of the component's
mass about that axis, before any height selection. Radius is also capped at half
the shortest box side. Ties at the cutoff can include more than 50% of the mass;
`enclosed_mass_fraction` records the actual fraction. Set `mass_fraction=None`
for the full usable radius, or another fraction in `(0, 1]`.

Height selection uses distance along the inferred axis, rather than global z.
The simulation currently selects heights within 1 for gas and 2 for particles,
in simulation length units. Components with ambiguous periodic centers or
undefined net axes produce a frame status and no curve. Axis coherence is the
magnitude of net angular momentum divided by the sum of individual magnitudes;
it measures cancellation, not a statistical confidence level.

Explicit frames are available through the component methods, for example:
```python
R, mean, sigma, mass = baryons.compute_rotation_curve(
    center=(0, 0, 0), axis=(0, 0, 1), mass_fraction=None, rmax=8, zmax=2,
)
```

`rotational_velocity.dat` retains its columns: time, component, bin-center radius,
mean velocity, velocity dispersion, weight. Weights now represent actual mass
for both particles and gas. `rotation_frames.csv` records the center, subtracted
bulk velocity, axis, cutoff, height selection, enclosed fraction, frame status
and units for each component and saved time. The inferred axis points along the
net angular momentum; use a fixed axis when comparing signed rotation reversals.

Plot the latest saved time with:
```text
python -m examples.plot_rotation_curves resources/data/simulation_... --km-s
```
Use `--time 0.5` for the nearest saved time, `--component gas` to select a
component, and `--output plot.png` to choose the output file. Bands or bars show
velocity dispersion, not the uncertainty of the mean. The new plotting entry point converts physical velocities to km/s by default;
use `--native-units` to keep simulation units. Legacy runs without frame
metadata can be plotted without unit conversion. Independently inferred frames
should be inspected before interpreting differences between component curves.

Verification tools and frozen references are available on the `working_with_chat` branch (switch to that branch to run these commands):
```text
python resources/diagnostics/rotation_playground.py
python chats_playground.py compare
python chats_playground.py diagnostics-check --allow-rotation-changes
```
The last command permits changes only to the two rotation output files while
requiring every energy sample, physical state array and other saved grid or
diagnostic output to match the pre-change references exactly.
## Convenient plotting entry point

```text
python plot_rot_curves.py
python plot_rot_curves.py -g
python plot_rot_curves.py -N
python plot_rot_curves.py simulation_20261004_...
python plot_rot_curves.py resources/data/simulation_... resources/data/simulation_...
```

No directory means the newest run with a rotation output file. Both gas and
N-body components are selected by default; `-g` selects gas and `-N` selects
N-body, while both flags select both types. Component types come from
`run_config.json`, with a name-based fallback for legacy runs. Each run uses
its own latest saved time, or the nearest time requested through `--time`.
The figure is saved in the first run directory and displayed; `--no-show` only
saves it. The examples entry point delegates to this same implementation.
