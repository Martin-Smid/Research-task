# Simulation class

`Simulation_Class` (`resources/Classes/Simulation_Class.py`) handles the box, the grid, the time stepping, the units and the
physics switches. It owns every other component (wave functions, baryons, propagator, evolution, output).

```python
from resources.Classes.Simulation_Class import Simulation_Class
#minimal usecase:
sim = Simulation_Class(dim=3, boundaries=[(-10, 10)] * 3, N=64, total_time=1.0, h=1e-3)
```

<!-- claude: everything below this line was filled in and checked against the source (file:line in brackets). -->

Constructor signature: `Simulation_Class.py:100-103`. Arguments are type-checked by the `parameter_check` decorator
(`:24-82`): a missing or wrongly typed argument **prints an error and calls `sys.exit(0)`** (it does not raise an exception).
Note that `m_s`, `sponge_V0` and `a_s` must be `float` (an `int` is rejected).

## Simulation properties

| Property | Code name | Expected type | Default |
|---|---|---|---|
| dimension | `dim` | `int` | **required** |
| box size | `boundaries` | `list` of `dim` pairs `(min, max)` | **required** |
| resolution | `N` | `int` | **required** |
| final time | `total_time` | `int` or `float` | **required** |
| time step size | `h` | `int` or `float` | **required** |
| order of evolution | `order_of_evolution` | `int`, one of 2, 4, 6 | `2` |
| units of the simulation | `sim_units` | `dict` (see below) | `{"dUnits": "kpc", "tUnits": "Gyr", "mUnits": "Msun", "eUnits": "eV"}` |
| use physical units | `use_units` | `bool` | `True` |
| save maximal values on the grid | `save_max_vals` | `bool` | `False` |
| folder of the output folders | `output_root` | `str` | `"resources/data"` |

Notes:

- `boundaries`: a `list` (a tuple is rejected by the type check), one `(min, max)` per dimension, numbers only, `min < max`,
  length equal to `dim` (`:251-274`). The grid has `N` points per axis **without** the right end point
  (`np.linspace(..., endpoint=False)`, `:278`), so `dx = (max - min) / N` and the box is periodic.
- `num_steps = int(total_time / h)` (`:126`) truncates: if `total_time` is not a multiple of `h`, the run ends earlier
  than `total_time`.
- `order_of_evolution`: 2, 4 or 6 (`:129`). **Use 2 for anything with baryons.** Orders 4 and 6 are untested together with
  N-body particles and sinks (the particle drift is called with the kick coefficients of the higher-order scheme, so it is
  not a higher-order scheme for them, `CLAUDE.md` issue 2) and with gas `evolve` raises `ValueError` unless the order is 2 (`Evolution_Class.py:78-79`).
  <!-- claude: orders 4/6 for pure ULDM are not covered by a test I found; ask the author if that should be stated -->
  **TODO: confirm** whether orders 4/6 are tested for pure ULDM.
- `h` above the stability limit raises `ValueError` (default `sim.strict_time_step = True`); set it to `False` to only print a warning
  (`:138`, `:627-671`). See *Choosing the time step* in `docs/user_guide.md`.
- `save_max_vals=True` writes `max_values.csv` / `max_locations.txt` at the end of the run (`Evolution_Class.py:733-737`).
  (With `evolve()` in a batch job the run ends in `input()`: use `echo n | python -u script.py`, see `KOIOS.md`.)
- `output_root`: the parent folder of the `simulation_<timestamp>` run folders (created if missing). The `Scribe` reads it when
  the evolution is built, so set it in the constructor (or before `initialize_simulation()`). It is stored in checkpoints
  (`checkpointing.py`, key `output_root`): `from_checkpoint` restores it and `start_new_segment` keeps it, so a new segment is
  written next to the old runs. `resume` does not use it: it continues in the original run folder (or `output_directory`).
  Checkpoints written before this parameter existed load with the default.

### `sim_units`

A dictionary with four keys; each value is the **name of an astropy unit** (`:216-222`):

| Key | Meaning | Default |
|---|---|---|
| `dUnits` | length | `"kpc"` |
| `tUnits` | time | `"Gyr"` |
| `mUnits` | mass | `"Msun"` |
| `eUnits` | unit in which the particle mass `m_s` is given | `"eV"` |

From them the class derives `mass_s` (`m_s` converted to `mUnits`), `c`, `G`, `h_bar` and `h_bar_tilde = h_bar / mass_s`
(`:227-231`). `m_s` (default `2.5e-22`) is the particle mass in `eUnits`/c².
The rest of the code and these docs assume kpc / Msun / Gyr. Other units should work with any names that astropy accepts,
because all constants are converted through `astropy.units` / `astropy.constants` (`:15`, `:227-231`), but **only the
default set has been tested**.

### `use_units`

`use_units=False` sets `h_bar = h_bar_tilde = G = c = mass_s = 1` (`:234-239`) and switches off the soliton mass
rescaling of file profiles (`Wave_function_class.py:61`). Used for dimensionless tests, e.g. the Sod shock tube
(`tests/gas/test_gas_sod_shock_tube.py:83`). Intended for debugging and dimensionless runs (Sod, harmonic oscillator);
it probably works for the other setups too, but this is not guaranteed.

## Physical properties

| Property | Code name | Expected type | Default |
|---|---|---|---|
| gravity (Poisson solver) | `use_gravity` | `bool` | `True` |
| additional static potential | `static_potential` | callable `f(sim) -> array` | `None` |
| particle mass | `m_s` | `float`, in `eUnits`/c² | `2.5e-22` |
| absorbing boundary (sponge) | `use_sponge` | `bool` | **`True`** |
| sponge strength | `sponge_V0` | `float` | `0.6` |
| self-interaction | `self_int` | `bool` | `False` |
| scattering length | `a_s` | `float`, in cm | `-10e-80` |
| sink formation | `sink_formation` | `dict` or `None` | `None` |
| legacy baryon profile | `baryonic_model` | `str` or `None` | `None` |

Note the default in bold: **the sponge is on unless you switch it off.** Self-interaction is off by default (it is experimental).
Most examples and tests pass `use_sponge=False` (e.g. `examples/initialize_simulation.py`, `tests/gas/test_gas_free_fall_collapse.py`).

### `use_gravity`

If `False`, the gravitational potential is identically zero (`Propagator_Class.py:103-104`): ULDM evolves freely and
baryons feel only `static_potential`.

### `static_potential`

A function that takes **the simulation object** and returns the potential on the grid (array of shape `(N,)*dim`):

```python
def harmonic(sim):
    x, y, z = [cp.asarray(g) for g in sim.grids]
    return 0.5 * omega**2 * (x**2 + y**2 + z**2)

sim = Simulation_Class(..., static_potential=harmonic)
```

What the code does with it:

- ψ: multiplied by `exp(-i h V / h_bar_tilde)` (`Propagator_Class.py:76-79`), so $V$ is a potential **per unit mass**, the same
  quantity as the gravitational potential $\phi$ (kpc²/Gyr² in code units).
- Baryons (particles, gas, sinks): `V_static` is added to the gravitational potential, only its real part
  (`Evolution_Class.py:775-779`).
- It is evaluated **once**, when the propagators are built (`Evolution_Class.py:699-708`), so it is time independent.
  It is also called once in the time-step stability check (`Simulation_Class.py:640-641`).
- It is **not stored in checkpoints**: pass it again to `from_checkpoint(path, static_potential=f)`
  (`checkpointing.py:451-453`). `start_new_segment(..., static_potential=f)` can change it.

Sign: the code adds $V$ to $\phi$ (and $\phi < 0$ for attraction, `Propagator_Class.py:191`), so an **attractive potential is
negative**, e.g. a point mass $V = -GM/r$. Return a **CuPy** array: the propagator wraps it in `cp.asarray`
(`Propagator_Class.py:78`), but the baryon path calls `cp.real(V_stat)` directly (`Evolution_Class.py:779`), which fails
for a NumPy array. A working example is `gravity_potential` in `resources/Functions/Schrodinger_eq_functions.py`
(point mass, $V = -GM/r$, regularised at $r=0$ by the cell size; the mass is hard-coded to $10^6$ Msun and the centre
defaults to the origin; use `functools.partial` or a lambda to set `mass_multiplier` / `center`).
Not tested for a long time (author's note).

### `use_sponge`, `sponge_V0`

Imaginary absorbing potential near the box edge, to remove waves that would otherwise re-enter through the periodic
boundary (`:680-724`; reference given in the code: Phys. Rev. D 94, 043513 (2016)). Spherical around the box centre; acts for
$r > \tfrac78 r_N$ where $r_N$ is the largest half-length of the box; strength `sponge_V0`. Experimental (`docs/user_guide.md`).
The recommended `sponge_V0 = 0.6` is the value from that paper (`:706`).

### `self_int`, `a_s`

Self-interaction of the ULDM field, with scattering length `a_s` given in **cm** (converted at
`Propagator_Class.py:113-115`; the default is literally `-10e-80`, i.e. $-10^{-79}$ cm). **Experimental**: the potential
`pref*(psi^2 + 2 rho psi)` is not understood yet, see known issue 3 in `CONTINUATION.md` (archive). Off by default; switch it on only if you
want it.

### `sink_formation`

`None` (no sink formation) or a dictionary with the configuration of automatic sink creation
(`Simulation_Class._resolve_sink_thresholds`, `Evolution_Class.enable_sink_particle_formation`, `Evolution._check_and_form_sinks`,
`Sink_N_Body.check_and_create_sinks`). One criterion for particles, gas and ULDM together. Keys:

| Key | Default | Meaning |
|---|---|---|
| `source` | `"both"` | what counts in the density **and** is consumed: `"baryons"` (particles), `"gas"` or `"both"` |
| `density_threshold` | `None` | total density [Msun/kpc^3] above which a sink seed appears. **Overrides every automatic value.** If `None`: `gas_density_threshold`, else the particle value below |
| `gas_density_threshold` | `None` | if `None` and gas is present (`source` gas/both) and `gas_mode == "truelove"`: $\rho = \pi c_s^2 / [G (N_J\,dx)^2]$ with the constant `gas.cs`. A number given here replaces it |
| `baryon_sigma_floor`, `baryon_NJ` | –, `gas_NJ` | if `density_threshold` is `None` and this is given: $\pi \max(\sigma, \sigma_{floor})^2 / [G (N_J dx)^2]$, sigma = 1D velocity dispersion of the particles at initialisation |
| `include_uldm` | `True` | add the ULDM density $\sum|\psi|^2$ to the density that is compared with the threshold |
| `federrath_checks` | `False` | extra checks of Federrath et al. (2010) on the region of the seed: converging flow and gravitationally bound (E_th + E_kin + E_grav < 0, E_grav = -3/5 G M^2 / r, M includes the ULDM mass); the potential-minimum check is replaced by the density-maximum condition. Works for particles (kinetic energy relative to the centre of mass, noisy below ~10 particles in the region) and gas (E_th = 3/2 sum P dV) |
| `dynamic_threshold` | `False` | recompute the Truelove threshold at every check: $\pi c^2 / [G (N_J dx)^2]$ with $c^2 = \max(\gamma P/ho$ of the gas in that cell, $\sigma^2$ of the particles now, floor$^2)$, floor = `baryon_sigma_floor` / `gas_cs_floor`. Infinite where there is no gas and no particles. An explicit `density_threshold` wins; it replaces the automatic `gas_density_threshold` and the one-off sigma estimate. $\sigma$ is one global value (1D dispersion of all particles), not local |
| `consecutive_steps` | `1` | number of consecutive checks the threshold must be exceeded |
| `check_interval` | `10` | check every this many steps |
| `gas_r_acc_cells` | `2` | accretion/aggregation radius in cells, for particles and gas |
| `gas_NJ`, `gas_cs_floor`, `gas_cs_truelove`, `gas_mode` | `4`, –, –, `"truelove"` | inputs to the automatic gas threshold |
| `merge_r_cells`, `merge_interval` | `2`, `10` | sinks closer than `merge_r_cells` cells are merged every `merge_interval` steps (only after a formation event) |
| `enable_gas_sinks`, `gas_consecutive_steps`, `gas_check_interval` | – | **ignored** (kept so old configurations load): gas is handled by `source`, with the common `consecutive_steps` and `check_interval` |

Working example of a configuration: `examples/variety/testing.py`.

How a sink forms (validated by `tests/sink/test_sink_formation.py` and `tests/sink/test_boss_bodenheimer.py`, 2026-11-04):

- The check runs every `check_interval` steps; a cell is a candidate after `consecutive_steps` checks above the threshold.
- Candidates are the cells that are the density maximum within `gas_r_acc_cells` cells (one GPU filter pass), taken densest first;
  none within that radius of an existing sink or of one created in the same call. A skipped cell keeps its count and may form
  a sink later, e.g. after the neighbouring sink has moved away.
- **One sink consumes both**: all particles within the radius, and in every cell within the radius the gas excess of the total
  density over the threshold (at most the gas density of the cell; the cell energy scales with its mass).
- Boss & Bodenheimer benchmark (`tests/sink/test_boss_bodenheimer.py`, plots by `plot_boss_bodenheimer.py`): the m = 2 collapse forms two sinks
  on opposite sides at 1.0-1.1 t_ff. Without `federrath_checks` the collapsing cores keep re-forming sinks (7 at N = 64, 23 at N = 128);
  with them exactly 2 remain at both resolutions.
- Bondi accretion (`sim.sink_accretion = True`) uses the constant `gas.cs` by default. `sink.bondi_local = True` switches to the
  Bondi-Hoyle-Lyttleton rate with the local sound speed $\gamma P/ho$ and the gas speed relative to the sink, both interpolated to
  the sink position (`tests/sink/test_sink_bondi_local.py`).
- The sink is placed at the **centre of mass** of what it consumed and carries its total mass and momentum. The kinetic energy
  lost and the internal energy of the removed gas are stored in `E_diss_formation_total`.
- The criterion is the Truelove (Jeans resolution) condition, a gas criterion: there is no check that the region is bound or
  converging. A cell with ULDM only (no baryons) never forms a sink (there is nothing to consume).
- Cost: a few ms per created sink (145 sinks from a uniform sphere in 6 s), almost independent of the number of candidate cells.

### Class-level switches (set on the instance before `evolve`, not constructor arguments, not stored in checkpoints)

| Switch | Default | Meaning |
|---|---|---|
| `sim.sink_accretion` | `False` | sinks eat gas and particles (Bondi, `sink.bondi_local`) before every step |
| `sim.sink_accretion_uldm` | `False` | together with `sink_accretion`: sinks also absorb $\psi$ (`SinkNBody.accrete_from_uldm`, rate $16\pi (GM/c^2)^2 cho$, arXiv:2309.02482). `sink.uldm_cross_section_scale` multiplies the rate (1 = physical, which is negligible for a $10^8\,M_\odot$ soliton). The energy taken from $\psi$ is booked into `E_diss` (`sink.E_diss_uldm_total`). Example: `examples/uldm_accretion_bh.py` |
| `sim.sink_merging`, `sim.sink_merge_radius` | `False`, `None` | merge bound sinks closer than the radius [kpc] (`None` = capture radius) |
| `sim.baryon_time_centered_force` | `False` | particles and sinks feel the force averaged over the step (order 2 only): second order in $h$, ~1.7x dearer per step |
| `sim.strict_time_step` | `True` | a step above the stability limit raises `ValueError` |

**Energy bookkeeping and the sponge.** `E_tot_cons` counts the energy that sinks take from $\psi$ and gas, but **not** the energy
carried away by $\psi$ that the sponge absorbs (`E_diss` stays 0). With `use_sponge=True` and a collapsing, radiating
$\psi$ the logged `E_tot_cons` therefore drifts (about -4 % in 3 Gyr in `examples/dwarf_sandbox.py`, where the drift
follows the mass lost in the sponge, $\Delta E/\Delta M pprox 140\,({m kpc/Gyr})^2$; this is a correlation, not yet a
proof). Judge the conservation with `use_sponge=False` or a box large enough that nothing reaches the edge.

Example of a sandbox with everything Gaussian (ULDM + stars + gas, sink formation on, `include_uldm=False`):
`examples/dwarf_sandbox.py`.

### `baryonic_model`

Legacy shortcut: if a string is given, `Baryons(self, N_particles=5000, total_mass=53090, init_profile=baryonic_model)` is
added automatically. Prefer `sim.add_baryons(Baryons(...))` with explicit numbers.

## Objects owned by the simulation

| Attribute | Class | Created by |
|---|---|---|
| `sim.wave_functions` | list of `Wave_function`, the components of the added {py:class}`Wave_vector_class <resources.Classes.Wave_vector_class.Wave_vector_class>` | `add_wave_vector`, filled by `initialize_simulation` |
| `sim.baryonic_matter` | list of {py:class}`Baryons <resources.Classes.Nbody_classes.Baryonic_N_body.Baryons>`, {py:class}`NBodyGas <resources.Classes.Nbody_classes.NBodyGas.NBodyGas>`, {py:class}`SinkNBody <resources.Classes.Nbody_classes.Sink_N_Body.SinkNBody>` | `add_baryons` |
| `sim.propagator` | {py:class}`Propagator_Class <resources.Classes.Propagator_Class.Propagator_Class>`: FFT propagators, Poisson solver | `initialize_simulation` |
| `sim.evolution` | {py:class}`Evolution_Class <resources.Classes.Evolution_Class.Evolution_Class>`: the time loop | `initialize_simulation` |

The simulation owns the evolution, which in turn owns the {py:class}`Scribe <resources.Classes.Scribe_Class.Scribe>` (all
output) and the {py:class}`Diagnostics <resources.Classes.Diagnostics_Class.Diagnostics>` (energies). They are reached as
`sim.evolution.scribe` and `sim.evolution.diagnostics` (`Evolution_Class.py:32-33`).
`sim.propagator` and `sim.evolution` are `None` until `initialize_simulation()` has run (`Simulation_Class.py:168-169`).

## Methods

### `add_wave_vector(wave_vector)`

Adds a ULDM field (`:300-342`). Accepts

- a `Wave_vector_class` instance (the normal case), or
- a `list` of `Wave_function` objects; they are appended as they are, **without** summing. A single `Wave_function`
  outside a list is not accepted (raises `ValueError`).

A spin-$s$ wave vector has $2s+1$ components, each a separate wave function propagated on its own. You can add as many
wave vectors as you like; vectors of the **same spin are summed component by component** (`:316-323`, the first
object is modified in place), vectors of different spin stay separate (`:352-358`):

| Added | Wave functions in the run |
|---|---|
| spin 0 | 1 |
| spin 0 + spin 1 | 1 + 3 = 4 |
| 10 × spin 1 | 3 (summed) |

(From reading the code, not run; no test of the count exists. The comment at `tests/spin/spin_gravity_comparison.py:95`
agrees: "add_wave_vector merges equal spins".)

### `add_baryons(system)`

Accepts `Baryons`, `NBodyGas` or `SinkNBody`, or a list of them (`:726-754`). The object must have been created with this
simulation, otherwise `ValueError`.

### `add_external_density(rho)`

Replaces the density that sources gravity by a fixed array `rho` (`:758-760`). Every Poisson solve then uses `rho`
**instead of** the density of the wave functions and baryons (`Propagator_Class.py:187-188`), so ψ and the particles move in
the potential of `rho` only.

What it expects (nothing is validated, the code just uses it):

- a **CuPy** array (the solver calls `cp.mean` and `cp.fft.fftn` on it, `Propagator_Class.py:190`),
- shape `(N,) * dim`, the same as the grid,
- real, in Msun/kpc³ (the potential is $-4\pi G\,\rho_k / k^2$, `Propagator_Class.py:191`); the mean is subtracted
  before the solve, so a constant offset has no effect,
- fixed in time. It is saved in checkpoints and restored by `from_checkpoint` (`checkpointing.py:251-256`, `:488-493`).

### `initialize_simulation()`

Prepares everything that `evolve` needs (`:344-380`): collects the wave vectors into `sim.wave_functions`, runs the
time-step check, and creates `sim.propagator` and `sim.evolution` (which owns the `Scribe` that writes the output and the
`Diagnostics`). It is not necessary to call if the user runs the simulations with `Simulation_Class.evolve()` as `evolve()`calls it itself.

Exists for the possibility of modifing the objects it creates before the run starts. The output folder no longer needs it:
pass `Simulation_Class(..., output_root="my_runs")`. Overriding `Scribe.output_root` after this call still works (it
overrides the simulation value; it is not stored in checkpoints). The run folder is created when the
evolution starts (`Evolution_Class.py:112`), so the override must come before that. Then start the run through
`sim.evolution.evolve`, **not** `sim.evolve` (that would call `initialize_simulation` again and build a new `Scribe`):

```python
sim.initialize_simulation()
sim.evolution.scribe.output_root = "my_runs"
sim.evolution.evolve(sim.wave_functions, save_every=50)
```

This is the pattern used in `tests/spin/spin_gravity_comparison.py:240-241`.

### `evolve(save_every=1, diagnostics_every=3)`

Runs the simulation (`:389-414`): `initialize_simulation()` followed by `Evolution_Class.evolve`, and then stores the results
on the simulation object (`combined_psi`, `snapshot_directory`, `accessible_times`, `current_step`, `current_time`).
`save_every = 1` saves a snapshot every step; use a larger value (see `docs/user_guide.md`, section 3).
When you call `sim.evolution.evolve` directly these attributes are **not** set; `start_new_segment` needs
`sim.snapshot_directory` (`:497`).

## Restart and continuation

Two different situations, two tools. Both start from a **checkpoint** of an earlier run.

| | `resume` | `start_new_segment` |
|---|---|---|
| Use when | the run crashed or was killed | the run finished and you want to go on, or go on with something changed |
| Needs | any checkpoint | a checkpoint of the **final** step (`:490`) |
| Output | same folder (or `output_directory`) | a new folder |
| Time argument | `until`: absolute end time | `total_time`: **new duration**, counted from the end of the old run |
| Can change | only the end time, `save_every`, `diagnostics_every` | `h`, `order_of_evolution`, `use_gravity`, `static_potential`, `use_sponge`, `sponge_V0`, `save_max_vals`, `self_int`, `a_s`, `sink_formation` (`:516-527`) |

Examples: `examples/resume_aborted_simulation.py`, `examples/start_new_segment.py`,
`examples/add_sink_after_relaxation.py`. Both paths are exercised by the reference benchmarks
(`chats_playground.py:669`, `:738`); there is no dedicated test in `tests/`.

### Checkpoints

`evolve` writes a checkpoint at every save step (`save_every`) and at the final step
(`Evolution_Class.py:277`, `:345-355`) into `<run folder>/checkpoints/step_XXXXXXXX/`; `checkpoints/latest.json` points to the
newest one (`checkpointing.py:216-298`). A checkpoint holds the constructor arguments, all wave functions, the baryonic
components (`Baryons`, `NBodyGas`, `SinkNBody`; any other type raises `TypeError`, `:206-207`), the external density and the
state of the `Scribe`.

### `Simulation_Class.from_checkpoint(path, static_potential=None, output_directory=None)`

Rebuilds the simulation from a checkpoint (`:416`, `checkpointing.py:435-516`).

- `path`: the run folder, its `checkpoints/` folder, one `step_*` folder, `manifest.json` or `latest.json`. Run folder and
  `checkpoints/` mean the newest checkpoint (`checkpointing.py:301-320`).
- `static_potential`: **must be passed again** if the original run used one, otherwise `ValueError` (`:451-455`).
- `output_directory`: where the files written from now on go. By default the original run folder; a new folder is created
  if needed (`:501-506`). Earlier snapshots are not copied.
- `output_root` is stored and restored (see above); `output_directory` only changes where this continuation writes.
- **Not stored, set again after loading:** `sim.sink_accretion`, `sim.sink_accretion_uldm`, `sim.sink_merging`,
  `sim.sink_merge_radius`, `sim.baryon_time_centered_force`, `sim.strict_time_step` (they are not in `simulation_config`,
  `checkpointing.py:85-113`). Note that `from_checkpoint` therefore gives the class defaults for them.
- Complex64 checkpoints from older runs stay `complex64` (`CLAUDE.md`, issue 7).

### `resume(until=None, save_every=None, diagnostics_every=None)`

For a run that stopped. Continues from the loaded step in the same folder and **calls `evolve()` itself**
(`:428-478`):

```python
sim = Simulation_Class.from_checkpoint("resources/data/simulation_...")
sim.resume(save_every=50)             # to the original end time
sim.resume(until=2.0)                 # or to a later time
```

- `until` must be an integer multiple of `h` and later than the checkpoint step (`:435-453`).
- `save_every` and `diagnostics_every` default to the values of the original run (`:466-473`).
- A checkpoint of the **final** step cannot be extended to a longer time (the integrator has already applied its closing
  half-step); resume from an earlier checkpoint or use `start_new_segment` (`:439-447`).
- It raises `ValueError` unless the simulation was made by `from_checkpoint` (`:430-431`).

### `start_new_segment(total_time, **changes)`

Takes the **final state** of a finished run as the initial condition of a new run, with a new output folder
(`:480-610`). With unchanged physics it is the same as one uninterrupted run up to rounding (`:484-488`).

```python
sim = Simulation_Class.from_checkpoint("resources/data/simulation_...")      # a completed run
sim.start_new_segment(total_time=0.5, h=0.001, use_gravity=True)            # 0.5 Gyr more
sim.add_baryons(sink)                                                        # optional: add components
sim.evolve(save_every=50)                                                    # writes a new folder
```

- Only the parameters in the table above can be changed; anything else (`dim`, `N`, `boundaries`, units) raises `ValueError`
  because it would need a remapping of the state (`:528-534`).
- `total_time` must be an integer multiple of `h` (`:540-543`).
- It returns the simulation itself (`:610`), and the changes are remembered in `sim._run_parameter_changes` (`:587-604`).
- Components can be added before `evolve()`; the standard recipe in this project is "relax first, then switch on baryons"
  (`examples/add_sink_after_relaxation.py`).
- It also works on a simulation that has just finished `evolve()` in the same script; the source is then
  `sim.snapshot_directory` (`:496-498`). **TODO: confirm**, I only read this path in the code.
- For an unfinished run use `resume`: a loaded checkpoint with `current_step < num_steps` is refused (`:490-494`).
