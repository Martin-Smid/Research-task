# Small regression checks

These files are intentionally plain Python scripts, so they can be opened and
run directly in PyCharm without configuring pytest.

Layout: `gas/`, `particle/`, `sink/` (tests plus longer measurement scripts such as `sink_friction*.py`),
`other/` (diagnostics, dressed soliton, time step, max values) and `spin/` (own README, not part of `run_all.py`).
`run_all.py` stays in this folder.

Run all checks by opening `run_all.py` and clicking the green Run button, or
run any `test_*.py` file separately to inspect one issue.

The expected result is one or more `PASS` lines. A regression raises an
`AssertionError` and PyCharm shows the failing line.

The scripts use CuPy and therefore need the same project interpreter and CUDA
environment as the simulation.

Heavier checks that are not part of run_all (GPU, minutes): other/test_dressed_soliton.py (1D and 3D soliton + black hole solvers), sink/sink_friction*.py and sink/sink_energy_conservation.py (measurement scripts).
