# Examples

Run these files from the repository root. Replace `simulation_...` placeholders
with one of your output directories under `resources/data`.

- `initialize_simulation.py` — create and evolve a small ULDM simulation.
- `resume_aborted_simulation.py` — resume from the latest saved checkpoint.
- `start_new_segment.py` — branch a completed endpoint into a new run.
- `add_sink_after_relaxation.py` — add a central sink to a relaxed system.
- `plot_max_values.py` — plot one or several runs.
- `plot_rotation_curves.py` — plot saved component rotation curves; see [rotation curve documentation](../docs/rotation_curves.md).

`resume()` starts evolution itself. After `start_new_segment()`, call `evolve()`.

Runs containing `NBodyGas` require `order_of_evolution=2`. Orders 4 and 6
contain negative substeps, which the gas solver does not support; those runs
raise an error before evolution begins. Gas steps must be finite and non-negative.
The gas CFL/subcycling method is unchanged.

Example: `python -m examples.initialize_simulation`

For latest-run plotting and the -g/-N component filters, use the root plot_rot_curves.py entry point. Helper/test file descriptions are in [diagnostics README on working_with_chat](https://github.com/Martin-Smid/Research-task/blob/working_with_chat/resources/diagnostics/README.md).
