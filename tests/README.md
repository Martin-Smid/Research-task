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

Added 2026-10-30..11-02: `t5_bh_soliton.py` (T5: black hole released at rest in a soliton, set-up of arXiv:2403.09038; environment variables N, BOX, EPS_CELLS, SMOOTH, T_MYR, H, ...; run with
`sbatch --export=ALL,T_MYR=60 slurm/koios_t5.sbatch`, ~8 min on a P100 for 60 Myr at N = 192) and `t5_analyze.py <simulation folder> [M_BH]` (CPU: BH velocity envelope and stopping time, energy drift,
momentum of psi + BH from the snapshots, plot); `test_time_step_strict.py` (in `run_all.py`); the spin tests live in `tests/spin/` (see its README; not in `run_all.py`).
Everything that guards the code in one go on the cluster: `sbatch slurm/koios_tests_all.sbatch` (run_all, spin tests, bitwise `compare` and `diagnostics-check` against the P100 set, ~10 min; the last line
of its log lists the failed steps).
