# User guide

A practical guide: how to set up a run, what each class does, where the output goes, and what to do when a run dies
or does not show what you need. Items marked **TODO** were not verified and should be checked by the author.

All paths are relative to the repository root. Units are **kpc, Msun, Gyr** (particle mass `m_s` in eV/c²).

## 1. A minimal run

```python
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class

sim = Simulation_Class(dim=3, boundaries=[(-10, 10)] * 3, N=32, total_time=0.01, h=0.001,
                       order_of_evolution=2, use_gravity=True, m_s=2.5e-22,
                       use_sponge=False, self_int=False, save_max_vals=True)

sim.add_wave_vector(Wave_vector_class(
    simulation=sim, packet_type="resources/solitons/GroundState(1).dat",
    means=[0, 0, 0], st_deviations=[0.5] * 3, momenta=[0, 0, 0],
    mass=1, omega=1, spin=0, desired_soliton_mass=53_090_064.0, random_seed=42))

sim.evolve(save_every=5)
```

This is `examples/initialize_simulation.py` (run `python -m examples.initialize_simulation`). The workflow is always:
**create `Simulation_Class` → add components → `evolve()`**.

## 2. Building blocks

| Class (file in `resources/Classes/`) | What it does |
|---|---|
| `Simulation_Class` | The box, grid (`N` points per axis), time (`total_time`, step `h`), units and the physics switches. Owns everything below. Entry points: `add_wave_vector`, `add_baryons`, `evolve`, `from_checkpoint`, `resume`, `start_new_segment`. |
| `Wave_vector_class` | A ULDM field of integer spin `s` (`2s+1` complex components; `spin=0` is the ordinary scalar field). `packet_type`: `"gaussian"`, `"LHO"`, `"blank"`, or a path to a profile file (e.g. a soliton `.dat`). `desired_soliton_mass` rescales a soliton profile. `random_seed` makes random initial fields reproducible; `independent_fields=True` gives each polarization its own random field. |
| `Baryons` (`Nbody_classes/Baryonic_N_body.py`) | Collisionless N-body particles (stars). `init_profile`: `hernquist`, `uniform`, `cold_clump`, `ring`, `spherical_clump`, `disk`, `from_file`. |
| `NBodyGas` (`Nbody_classes/NBodyGas.py`) | Gas on the same grid: finite volume, MUSCL reconstruction + Rusanov flux, periodic. Needs `order_of_evolution=2`. `free_fall_estimate(mass, radius)` prints the collapse time and a suitable `h`. |
| `SinkNBody` (`Nbody_classes/Sink_N_Body.py`) | Sink particles (black holes): analytic Gaussian-softened potential, sink–sink gravity, optional gas accretion and merging. |
| `Propagator_Class`, `Evolution_Class` | Internals: the FFT split-step propagator (orders 2, 4, 6), Poisson solver and the time loop (kick–drift–kick for gas; particles and sinks use the potential frozen at the start of the step). You normally do not call them. |
| `Diagnostics`, `Scribe`, `RotationCurves` | Energies, all file output and checkpoints, and rotation curves of baryonic components. |

Everything is added to the simulation before `evolve()`:

```python
sim.add_wave_vector(wave_vector)        # one or several ULDM fields
sim.add_baryons(particles)              # Baryons, NBodyGas and SinkNBody go through the same call
```

Other hooks: `static_potential=f` in `Simulation_Class` (a function that takes the simulation and returns the potential on the
grid), `sim.add_external_density(rho)` (replaces the density that sources gravity), `use_sponge` (absorbing boundary,
experimental), `self_int` / `a_s` (self-interaction; **experimental, see the known issues in `CLAUDE.md`**).
Note: `self_int` defaults to `True` in `Simulation_Class`; set `self_int=False` unless you want it.

### Sinks

```python
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
sink = SinkNBody(simulation=sim, N_sinks=1, initial_masses=[1e7],
                 initial_positions=[[0, 0, 0]], initial_velocities=[[0, 0, 0]])
sim.add_baryons(sink)
sim.sink_accretion = True      # off by default
sim.sink_merging = True        # off by default; optional sim.sink_merge_radius in kpc
```

Rules of thumb (from the tests): the sink field is usable up to r ≲ L/4 (periodic box); for friction use
`softening_bh ≲ 0.3 λ` with `λ = ħ/(m v)` and `λ ≥ 3` cells. `smooth_force=True` makes the sink feel the field with the same
Gaussian that acts on it (momentum balance exact). Results depend on the merge radius and on `N`, so report both.

## 3. Choosing the time step and the resolution

- Stability limit on `h` (from the dispersion relation, ∝ dx²) is checked when the simulation is initialised. A too large `h`
  only **warns** by default; set `sim.strict_time_step = True` to raise an error.
- Gas needs a CFL substep criterion (automatic); `free_fall_estimate` gives a start value for collapse problems.
- Memory grows as N³ (the wave function is `complex128`); N = 384 did not fit into a 16 GB P100.
- `evolve(save_every=1)` saves a snapshot **every step**, which is huge. Use e.g. `save_every=50` and
  `diagnostics_every` for energies.

## 4. Output

Each run writes a folder `resources/data/simulation_<timestamp>/` (the parent folder is `output_root` in `Scribe_Class.py`; **TODO:** document how to change it from a script).
`resources/data/` is ignored by git and can become very large.

| File | Content |
|---|---|
| `wf_<i>_snapshot_at_time_<t>.npy` | wave function `i` at time `t` |
| `total_density_…`, `baryons_…`, `gas_…` (`.npy`) | densities at the save times |
| `max_values.csv`, `max_locations.txt` | maximum density in time (written during the run) |
| `energy.csv` | energies from the diagnostics |
| `particle_trajectory.csv`, `rotation_frames.csv` | particle / sink trajectories, rotation-curve frames |
| `run_config.json`, `metadata.txt` | the run configuration and the list of saved times |
| `checkpoints/` (+ `latest.json`) | restart data, see below |

Plotting scripts are in `plots/` (`plot_density.py`, `plot_energies.py`, `plot_from_snapshots.py`, `Plot_max_vals.py`,
`plot_rot_curves.py`, …); they take the simulation folder as an argument. Rotation curves: `docs/rotation_curves.md`.
Analysis helpers without a GPU (power spectrum, core properties, spin density): `resources/Functions/spin_analysis.py`.
**TODO:** `examples/plot_max_values.py` and `examples/plot_rotation_curves.py` still import from the repository root; they must
be updated after the scripts moved to `plots/`.

## 5. When a run dies, or you need to go on

There are two different tools. Both start from the **simulation folder** of the earlier run.

### 5.1 The run crashed or was killed → `resume`

A checkpoint is written at every save step (`save_every`) and at the end, so at most `save_every` steps are lost.

```python
sim = Simulation_Class.from_checkpoint("resources/data/simulation_...")
sim.resume(save_every=50)          # calls evolve() itself; do not call evolve() again
```

`resume(until=T)` continues to another end time (must be a multiple of `h`). The run goes on in the same folder.
A checkpoint taken at the very **final** step cannot be extended to a longer time (the integrator has already applied
its closing half-step); in that case use 5.2 instead, or resume from an earlier checkpoint.

### 5.2 The run finished but did not show what we needed → `start_new_segment`

```python
sim = Simulation_Class.from_checkpoint("resources/data/simulation_...")      # a completed run
sim.start_new_segment(total_time=0.5, h=0.001, use_gravity=True)            # NEW duration, not absolute time
sim.evolve(save_every=50)                                                    # a new output folder
```

It takes the final state as the initial condition of a new run with a new folder. With unchanged physics it is equal to one
uninterrupted run up to rounding. What can be changed: `h`, `order_of_evolution`, `use_gravity`, `static_potential`,
`use_sponge`, `sponge_V0`, `save_max_vals`, `self_int`, `a_s`, `sink_formation`. The grid and the box **cannot** be changed.
You can also add components before `evolve()`, for example a sink after the soliton has relaxed
(`examples/add_sink_after_relaxation.py`); the standard recipe in this project is "relax first, then switch on baryons".

`start_new_segment` needs a completed endpoint; for an unfinished run use `resume`.

### 5.3 Typical problems

| Symptom | Likely cause / action |
|---|---|
| Warning "time step is too large" | `h` above the stability limit; use the advised `h` (or `strict_time_step = True` to stop). |
| Gas: `FloatingPointError`, NaN | the collapse is too violent for `h` or `max_substeps`; reduce `h` (`free_fall_estimate`) or raise `max_substeps`. |
| Energy or norm drifts with the number of steps | reduce `h`; since the switch to `complex128` the norm of ψ is conserved to rounding. |
| A sink "runs away" at small softening | softening below one cell without `smooth_force`; use softening ≥ 1 cell or `smooth_force=True` (see `tests/sink/`; **TODO** confirm the recommended default). |
| Out of memory | lower `N`; snapshots and checkpoints of ψ are `complex128` (twice the size of `complex64`). |
| `ImportError`/CUDA error on start | see *Installation* in the README; on Windows `examples/_runtime.py` adds the CUDA path. |

## 6. Reusing the pieces in other simulations

- Any combination of ULDM field(s) + particles + gas + sinks is just a set of `add_*` calls (section 2).
- `sim.propagator.compute_force_grids_from_potential(phi)` returns the force grids `a = -∇φ` for any potential on the grid;
  the tests use it to study the field of a sink or a soliton.
- `Baryons` and `SinkNBody` can be used with a fixed field, as in `tests/particle/` and `tests/sink/`; these tests are the
  best working examples of how to build a controlled setup (static potential, test particle, binary of sinks, friction).
- Output arrays are plain `.npy` files, so any analysis can run on a machine without a GPU.

## 7. Tests as documentation

`python tests/run_all.py` (about one minute) runs the regression checks. Folders: `tests/gas`, `tests/particle`,
`tests/sink`, `tests/other`; the spin tests are in `tests/spin`, the heavier measurement scripts (friction, energy
conservation) are in `tests/sink` and are not part of `run_all.py`. Benchmarks against reference sets
(`chats_playground.py compare`, `diagnostics-check`) are described in `CLAUDE.md`.
