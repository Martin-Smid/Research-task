# CLAUDE.md — ULDM / Fuzzy Dark Matter simulation

Rules for working in this repository. State, history and results live elsewhere (see "Where things are").

## Project context
- Diploma-thesis code simulating ultralight (fuzzy) dark matter with a spectral (FFT) Schrödinger–Poisson solver, plus baryonic
  components (N-body particles, gas / NBodyGas, sinks). Goal: a **publishable simulation framework**. Correctness and
  reproducibility matter more than speed of development.
- GPU code uses CuPy. Priority is the ULDM solver; the gas solver is a supporting component (see "Accepted limitations").

## Working with the user
- The user writes Czech. Reply in Czech, short and concrete; the user gets tired of long reports. Yes/no question: start with yes or no.
- Read `CONTINUATION.md` first (Czech work log, "START HERE"). Do not claim more than the data show; report failures and gaps plainly.
- Do not publish, delete or overwrite results (`.npz`, plots) unasked. Before deleting, list the exact names and wait for OK; only delete what your own scripts created.

## Git workflow
- `working_with_chat` = development branch, do all work there. `main` = finished code: never commit or merge to it unless asked.
- Do not commit on your own: propose a commit message and wait for OK. Push only when asked.
- Commits carry **no** `Co-Authored-By` or other Claude/Anthropic lines.
- Stage only your own files. The user's uncommitted work (`main.py`, `plot_*.py`, notebooks, `docs/api.md`, `docs/index.md`, `resources/Functions/system_fucntions.py`, ...) must not be changed or committed.
  The index may already hold files staged by the user: run `git diff --cached --name-only` before committing and commit with explicit paths.
- The GitHub repository is public. Private, git-ignored: `plans.md` (exists only on Koios, keep a backup), `docs/paperBC_draft.tex`,
  `docs/clanek_BC_obsah.md`, `docs/paper1_outline.tex`, `docs/crosscode_comparison/`, `docs/t5/`. Do not put them back into git.

## How to change code
- **Write as little code as possible.** Smallest change that solves the problem. No refactors, renames, reformatting or "cleanups" not asked for.
- Prefer editing existing functions over adding new modules/abstractions. If a fix needs a bigger change, stop and explain the options first.
- Do not change physics, units or default parameters silently; always call it out (measurement scripts must not alter solver physics, units or defaults).
- **Everything written into code is in English**: comments, docstrings, printed messages, plot labels (chat stays Czech). Do not
  translate existing Czech comments unasked.

## After every change, report (always, in this format)
1. **What** changed and **why** (1–3 sentences).
2. **Where**: `file:line` for every edit.
3. **Before → after**: original and new code side by side (short snippets).
4. **How to verify**: exact command(s), what output/number/plot to look at, and what result means "it works".

## Tests and benchmarks
- After any change to solver/physics code, **run the existing benchmarks** and confirm the results did **not change** (or explain exactly
  why they changed and by how much, before/after numbers).
- For every bug fix or new feature, **add a test** that would have failed before (write it first, watch it fail, then fix).
- Never weaken or delete a test to make it pass. Do not overwrite reference data to "fix" a test.
- Still to build: convergence with resolution as a test, energy conservation vs. step size as a test.

## Commands (project venv; PowerShell/Git Bash)
- The venv is `venv/` on the RTX 3070 notebook and `.venv/` on the Quadro machine. Tests (≈15 min now; the user runs them locally):
  `venv/Scripts/python.exe tests/run_all.py` (each test also runs on its own and prints `PASS`). On Koios: `sbatch slurm/koios_tests.sbatch`.
- Benchmarks against a reference set (all must say `bitwise identical`; the P100 set `baselines_koios_p100/` is the maintained one, Quadro/RTX/GTX sets are stale):
  ```
  python chats_playground.py compare --baseline-dir resources/diagnostics/baselines_quadro_p4000/chat_regression_baselines
  python chats_playground.py diagnostics-check --baseline-dir resources/diagnostics/baselines_quadro_p4000/chat_regression_baselines
  python resources/diagnostics/check_optimization_rtx3070.py after resources/diagnostics/baselines_quadro_p4000/optimization_checks
  python chats_playground.py compatibility-check
  ```
  Regenerating a set: `chats_playground.py baseline --baseline-dir <dir> --overwrite` (the diagnostics set must be deleted first, then `diagnostics-baseline`);
  update `hashes_*.json` afterwards.
- Gas collapse planning: `free_fall_estimate(mass_Msun, radius_kpc)` from `resources/Classes/Nbody_classes/NBodyGas.py`.
- Simulation output goes to `resources/data/` (git-ignored, very large). Tests clean up after themselves; scripts that call `sim.evolve` do not:
  delete the generated `simulation_*` folders when done (listing the names first).

## GPU, Koios and data etiquette
- Details of the cluster (venv, sbatch, P100 reference set): `KOIOS.md`. GPUs gpu3/gpu4 (P100) are shared with other users (often only one is free; jobs of `pombo`); do not use CPU nodes.
- Local RTX 3070 notebook: use only after agreement, one job at a time. Any job estimated over ~15 minutes: write the estimate and wait for OK
  (count on 1.5–2× on a shared GPU).
- From Claude Code use `/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes koios '...'` and `scp.exe` (not `ssh` from Git Bash). Write scripts into a file and scp them.
  `Permission denied` = ssh-agent is not running: ask the user to run `Start-Service ssh-agent` (admin) and `ssh-add C:\Users\Macík\.ssh\id_ed25519_fzu`.
- Submit long jobs with `sbatch` (long `srun` over ssh drops). `--export` breaks on values with commas: put variables into the sbatch file.
  Chain jobs with `--dependency=afterany:<id>`. No `sleep` over a few seconds in the foreground: wait in a background command.
- Koios has another agent (files: `correct_params.py`, `testing.py`, `plans.md`, `tests/t5_*`, `slurm/koios_t5.sbatch`): do not touch them, and do not leave an interactive `srun --pty bash`.
- `resources/data/` on Koios has no backup. Never delete results there.

## Accepted limitations (decided by the user, 2026-10-16)
- Internal (thermal) energy of cold gas in an unresolved potential is not reliable (`E−K` absorbs the face/cell work difference; a guard
  keeps it above the floor, `gas.E_guard_injected`). Gas thermodynamics is not the priority.

## Where things are
- `CONTINUATION.md`: work log and current state (START HERE, TL;DR, article BC, release checklist). **Archive of the former CLAUDE.md** (known issues 1–16 with all
  numbers, session logs, lessons, handoff 2026-10-27) is at its end. Read the known issues there before touching sinks, gas, precision or the time step.
- `plans.md` (Koios only): research plan and numbers, sections 11.x = conclusions. `KOIOS.md`: cluster. `tests/README.md`, `tests/spin/README.md`: tests and measurement scripts.
- `docs/paperBC_draft.tex` (private): the working draft of article BC with a "Reader's guide" (files, data, jobs).

## Traps
- "Half-integer spin" in the scripts (`s = 1.5, 2.5`) means an N-component field (N = 2s+1 independent equal-mass components built from different integer spins), not spin 3/2.
- A circular orbit with v = sqrt(GM/r) is not circular in the periodic box; the sink field is usable only up to r ≲ L/4. Use an isolated W for virial checks.
- Friction tests need λ = ħ/(m v) ≥ 3 cells and sink softening ≲ 0.3 λ.
- Redirected stdout swallows your own prints; filtering with grep can hide the one line you need.

## Working lessons
- **Measure before explaining**: instrument per operation, compare old vs new by monkey-patching, compare with an exact solution.
- **Check the test, not only the code**: when a validation test disagrees, suspect the setup first.
- Energy diagnostics can be wrong independently of the dynamics: separate "dynamical energy" from the logged `E_total` before blaming the solver.
- Do not leave simulation output behind; `resources/data` fills the disk.
