# CLAUDE.md — ULDM / Fuzzy Dark Matter simulation

## Project context
- Diploma-thesis code simulating ultralight (fuzzy) dark matter with a spectral (FFT)
  Schrödinger–Poisson solver, plus baryonic components (N-body particles, gas / NBodyGas, sinks).
- Goal: grow it into a **publishable simulation framework**. Correctness and reproducibility
  matter more than speed of development.
- GPU code uses CuPy; simulations and tests run on this Windows machine (Quadro P4000).
- **Priority is the ULDM solver.** The gas solver is a supporting component; known weaknesses of it
  that do not affect ULDM results are accepted (see "Accepted limitations").
- The user writes Czech. Reply in Czech, short and concrete; the user gets tired of long reports.
  When asked a yes/no question, start with yes or no. Read `CONTINUATION.md` first (Czech, work log).

- Running on the Koios cluster (connection, venv, sbatch, P100 reference set): see `KOIOS.md`.

## Git workflow
- `working_with_chat` = development/testing branch. Do all work here.
- `main` = finished, tested code. Never commit or merge to `main` unless I explicitly ask.
- Do not commit on your own; propose a commit message and wait for my OK. Push only when I say so.
- Stage only your own files; `main.py`, `plot_*.py` and notebooks are often my uncommitted edits.

## How to change code
- **Write as little code as possible.** Smallest change that solves the problem.
  No refactors, renames, reformatting or "cleanups" I did not ask for.
- Prefer editing existing functions over adding new modules/abstractions.
- If a fix needs a bigger change, stop and explain the options first; let me choose.
- Do not change physics, units or default parameters silently — always call it out.

## After every change, report (always, in this format)
1. **What** changed and **why** (1–3 sentences).
2. **Where**: `file:line` for every edit.
3. **Before → after**: show the original code and the new code side by side (short snippets).
4. **How I can verify it**: exact command(s) to run, what output/plot/number to look at,
   and what result means "it works".

## Tests and benchmarks
- After any change to solver/physics code, **run the existing benchmarks** and confirm the
  results did **not change** (or explain exactly why they changed and by how much).
  Report the before/after numbers.
- For every bug fix or new feature, **add a test** that would have failed before the change
  (write it first, watch it fail, then fix).
- Physics validation tests: Sod, Jeans, free-fall collapse and energy-work balance exist; still to build:
  Kepler orbit, convergence with resolution as a test, energy conservation vs. step size as a test.
- Never weaken or delete a test to make it pass.

## Commands (use the project venv; PowerShell/Git Bash)
- Run tests (about 1 minute): `.venv/Scripts/python.exe tests/run_all.py`
  (each `tests/test_*.py` also runs on its own and prints `PASS`).
- Benchmarks against the Quadro P4000 reference set (all must say `bitwise identical`):
  ```
  .venv/Scripts/python.exe chats_playground.py compare --baseline-dir resources/diagnostics/baselines_quadro_p4000/chat_regression_baselines
  .venv/Scripts/python.exe chats_playground.py diagnostics-check --baseline-dir resources/diagnostics/baselines_quadro_p4000/chat_regression_baselines
  .venv/Scripts/python.exe resources/diagnostics/check_optimization_rtx3070.py after resources/diagnostics/baselines_quadro_p4000/optimization_checks
  .venv/Scripts/python.exe chats_playground.py compatibility-check
  ```
  Other GPUs have their own reference sets (`baselines/` GTX 1050 Ti, `baselines_rtx3070/`); their gas
  cases are stale since 2026-10-16 (gas coupling changed on purpose). Regenerating a set:
  `chats_playground.py baseline --baseline-dir <dir> --overwrite`; the diagnostics set must be deleted
  first (the tool refuses to overwrite), then `diagnostics-baseline`. Update `hashes_*.json` afterwards.
- Gas collapse planning: `from resources.Classes.Nbody_classes.NBodyGas import free_fall_estimate`;
  `free_fall_estimate(mass_Msun, radius_kpc)` prints the collapse time and the recommended `h`.
- Simulation output goes to `resources/data/` (ignored by git, very large). Tests clean up after themselves;
  scripts that call `sim.evolve` do not — delete the generated `simulation_*` folders when done.

## Known issues — status
1. Gas CFL / FloatingPointError in NBodyGas:
   - **Fixed 2026-10-16:** `dt_sub` was computed once per window; in a collapse the true Courant number grew
     from 0.5 to ~3 and the run ended in NaN. Now `_substeps` re-evaluates it before every substep.
     Emptied cells (below `rho_floor`) also lose their momentum/energy (`_reset_emptied_cells`).
   - **Fixed 2026-10-26 (the window disk case):** independent limiting of rho and momentum left a vacuum cell
     between two diverging flows with finite momentum but floored density, i.e. a face velocity ~1e11; Rusanov
     fluxes of 1e10 made negative densities, whose clamp *created* mass. `_sweep_axis` now falls back to the
     first-order cell states on faces whose velocity exceeds 10x (|v|+c) of the two adjacent cells. Regression
     baselines stay bitwise identical; test `tests/test_gas_vacuum_hole_between_flows.py`. Reproducer:
     `resources/diagnostics/cfl_disk_repro.py`.
   - MUSCL with a single forward-Euler stage is untouched (limits time accuracy of the hydro).
2. Baryon coupling in time: gas now uses a kick–drift–kick with a refreshed potential (second order for the
   gas); **particles and sinks still use the frozen potential**; orders 4/6 call particle drift with kick
   coefficients; gas supports order 2 only.
3. Self-interaction potential `pref*(psi^2 + 2 rho psi)` looks complex/non-unitary; expected `g|psi|^2`. Not touched.
4. Then: cleanup/packaging (requirements.txt is UTF-16 (tqdm added 2026-10-26); `.idea` tracked; duplicate
   `system_fucntions.py`) → docs → settle licence/citation with me.

5. Sinks (fixed 2026-10-26): (a) the analytic sink potential ignored the box's lower boundary, so in a [-L, L] box
   a sink at the centre pulled towards the corner (-L, -L, -L); (b) `accrete_from_gas` removed density but not the
   energy of the cell (specific energy and CFL exploded). Both fixed with tests; accretion is off unless
   `sim.sink_accretion = True` (the call in the main loop was commented out). 5 sink reference files per set
   changed; only `baselines_koios_p100/` was regenerated, the Quadro/RTX/GTX sets are stale for sink cases.
   Older BH runs in boxes that do not start at 0 were affected.
6. Open observations: the ULDM wave function loses 0.04-0.09 % of its mass per Gyr (unexplained); gas loses ~3.6 %
   of its angular momentum per orbit at N=128 (numerical viscosity of Rusanov, should fall with resolution);
   Bondi accretion uses the constant `gas.cs`; the code's `W` (periodic Poisson) is ~17 % off the isolated value in a
   20 kpc box, so use an isolated `W` for virial checks (isolated soliton: 2K/|W| = 1.004).

7. Wave-function precision (fixed 2026-10-26): psi was kept in complex64 (created as complex64, the in-place `*=` keeps
   the dtype), so every kick and drift rounded to float32 and the norm of psi drifted proportionally to the number of
   steps (about -4e-5 % per step; -0.3 % per Gyr at h=1.25e-4, N=256; energy drift +0.28 % -> +1.2 % when h shrank 4x).
   Now `complex128` (`Wave_function_class.py`, `Wave_Packet_Class.py`): the norm is conserved to rounding and the results no
   longer depend on h. Results changed by ~1e-7 (relative). Only `baselines_koios_p100/` was regenerated; the Quadro, RTX
   and GTX sets are stale for ALL cases (psi dtype). psi snapshots/checkpoints are twice as big; old complex64 checkpoints
   stay complex64 on restart.
8. Particle tests (2026-10-26, `tests/`): `test_particle_kepler_orbit.py` (NBody.drift with one substep is the textbook DKD,
   mean order 2.06; the real step is set by the substep criterion 0.25*dx/v, not by h; production error <1e-3 r0 after 3
   orbits), `test_sink_force_vs_analytic.py` (sink force on the grid vs the Gaussian + periodic-box formula: mean error ~1e-4;
   the max error grows as (r/L)^4 above ~4 kpc = cubic lattice anisotropy, so the sink field is usable up to r <= L/4),
   `test_particle_breathing_soliton.py` (time-varying source: first order in h because the force is frozen at the start of
   the step; error ~1e-3 r0 for a source that changes 3x; accepted).
9. A circular orbit started with the analytic v = sqrt(GM/r) is NOT circular in a periodic box: at r = L/4 the force is ~6.5 %
   weaker (background term -(4 pi/3) G M r / L^3) plus 1-3 % axis anisotropy. Use r0 <~ L/10 or the speed from the grid force
   (`circular_velocity` in `test_particle_kepler_orbit.py`). `testing_baryons.py` is the minimal orbit script; the old orbit
   figure of the thesis (r0 = 20 kpc) shows exactly this effect.

## Accepted limitations (decided by the user, 2026-10-16)
- The internal (thermal) energy of **cold gas in an unresolved potential** is not reliable: the difference between
  the face-flux gravity work and the cell-centred work lands in `E−K`; a guard keeps it above the floor and
  tracks the injected energy in `gas.E_guard_injected` (~1.5 % of |E0| in the disk test). This is a property of
  the chosen solver; gas thermodynamics is not the priority.

## Session log — 2026-10-16 (what was done; details in `CONTINUATION.md` and `git log`)
- Moved to a new machine (Quadro P4000), created `baselines_quadro_p4000/`; later regenerated its gas cases.
- Found that gas gravity did not conserve energy (+3.6 |E0| in 0.06 Gyr). Fixes in `NBodyGas.py` / `Evolution_Class.py`:
  gravity work through cell faces, refreshed potential (kick–drift–kick), internal-energy guard; and in
  `Diagnostics_Class.py` gas internal energy was counted once per baryonic system → once. Drift now ~0.1 % (N=64).
- Verified against analytic cases: Sod shock tube, Jeans instability (growth and oscillation), free-fall collapse
  of a cold sphere, work/energy balance. Old code did not converge with resolution on the disk run; the new does.
- Fixed the heavy-collapse NaN (stale CFL step, see above). Cost of the new machinery: a few percent per step.
- Added tests: work balance, positive internal energy, U_iso counted once, Sod, Jeans, free fall, vacuum cell,
  adaptive substeps. `tests/` is no longer git-ignored.

## What I (Claude) learned this session — apply next time
- **Measure before explaining.** Several first hypotheses were wrong (e.g. "floors cause the NaN", "old code
  converges in Jeans"). The decisive tools were: instrument per operation, compare old vs new by monkey-patching
  (`Evolution._set_gas_potential = lambda self, a, b, c: None` reproduces the old gas behaviour bitwise),
  and compare with an exact solution.
- **Check the test, not only the code.** Jeans first failed because my initial pressure was isothermal instead of
  adiabatic; Sod first failed because periodic waves met at t=0.15. When a validation test disagrees, suspect
  the setup first, then the code.
- A constant offset in a measured growth rate can be a time-label offset; use a local log-slope instead.
- Redirected stdout swallows your own prints; filtering output with grep can hide the one line you need.
- Report failures of your own change plainly (the new code gave NaN where the old one gave a clean error).
- Energy diagnostics can be wrong independently of the dynamics (the double counted U_iso): separate
  "dynamical energy" from the logged `E_total` before blaming the solver.
- Periodic box: a rotating disk IC is not periodic; the "cross/arms" in density slices come from box faces.
- Collapse planning: t_ff = sqrt(3π/(32 G ρ0)); use h ≈ t_ff/100; `max_substeps` may need raising.
- Do not leave simulation output behind; `resources/data` fills the disk (8 GB removed today).

## Session log — 2026-10-26 (Koios, details in `CONTINUATION.md`)
- Running on the Koios cluster: see `KOIOS.md` (ssh-agent, venv, sbatch). P100 reference set `baselines_koios_p100/`
  is bitwise reproducible; `compare_baseline_sets.py` compares two sets.
- Fixed: window-disk CFL case (see issue 1), sink potential offset and accretion energy (issue 5). New tests in
  `tests/` for all three; `run_all.py` passes.
- `main.py` is now a simplified dwarf galaxy (soliton 5.3e7 Msun + stars + gas + black hole, masses as fractions of
  the soliton) in two phases: relaxation with stars switched on slowly, then `from_checkpoint` + `start_new_segment`
  with gas and sink. Plain relaxation does not damp the soliton breathing (+-13 %, period ~0.75 Gyr); slow switching-on does.
- `plot_energies.py` / `plot_density.py` / `Plot_max_vals.py` / `plot_from_snapshots.py` save images into the simulation directory.
- Nothing is committed.

## Session log — 2026-10-26 (evening)
- Convergence runs N=64/128/256 (phase 2, 1 Gyr, complex64): soliton profile and inner gas rotation converge (128 vs 256
  within 1-2 %), gas angular momentum loss -31/-18/-10 %. Energy drift and psi mass loss grew with the NUMBER OF STEPS, not
  with N: cause = complex64 psi (issue 7). Fixed with complex128; `tests/run_all.py` passes; P100 reference set regenerated.
- New tests for particles and the sink field (issue 8); minimal orbit script `testing_baryons.py` (analytic Kepler speed).
- Next: convergence N=32/64/128/256 with complex128; sink oscillation test in the soliton; open points 2 and 3 above.
