# Examples

Run these files from the repository root. Replace `simulation_...` placeholders
with one of your output directories under `resources/data`.

- `initialize_simulation.py` — create and evolve a small ULDM simulation.
- `resume_aborted_simulation.py` — resume from the latest saved checkpoint.
- `start_new_segment.py` — branch a completed endpoint into a new run.
- `add_sink_after_relaxation.py` — add a central sink to a relaxed system.
- `plot_max_values.py` — plot one or several runs.

`resume()` starts evolution itself. After `start_new_segment()`, call `evolve()`.

Example: `python -m examples.initialize_simulation`
