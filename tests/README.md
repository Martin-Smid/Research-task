# Small regression checks

These files are intentionally plain Python scripts, so they can be opened and
run directly in PyCharm without configuring pytest.

Run all checks by opening `run_all.py` and clicking the green Run button, or
run any `test_*.py` file separately to inspect one issue.

The expected result is one or more `PASS` lines. A regression raises an
`AssertionError` and PyCharm shows the failing line.

The scripts use CuPy and therefore need the same project interpreter and CUDA
environment as the simulation.
