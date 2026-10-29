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
- **Everything written into code is in English**: comments, docstrings, printed messages, plot labels (the chat with me
  stays Czech). This holds for new files and for lines I add; do not translate existing Czech comments unasked.

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
6. Open observations: (the ULDM wave-function mass loss of 0.04-0.09 % per Gyr was the complex64 precision, fixed, see 7); gas loses ~3.6 %
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

10. Convergence with complex128 (main.py, phase 2, 1 Gyr, 2026-10-26; h = 5e-4 for N = 64/128, 1.25e-4 for N = 256; N = 32
   relaxes but the gas goes NaN in phase 2, too coarse): psi mass change ~1e-10 %, energy drift -0.036 / -0.024 / -0.015 % for
   N = 64 / 128 / 256 (before: +0.25 / +0.28 / +1.16 %, growing with the number of steps). Soliton converges: rho_max(t=0)
   1.500 / 1.508 / 1.511 e7, profile at r >= 1 kpc within 1 % between 128 and 256; below ~0.5 kpc slower (order ~1). Breathing
   of rho_max (10.3 / 12.2 / 13.4 % for N = 64 / 128 / 256) is physical: the soliton oscillates after the baryons are
   switched on. Its growth with N is mostly the better resolved peak (max 1.662 / 1.704 / 1.727e7, increments 0.042 / 0.023),
   not a different physics. Gas: angular-momentum loss -30.9 / -18.2 / -9.9 %
   (falls slowly, apparent order ~0.6 from three points), vphi at 1-3 kpc 5.28 / 6.34 / 6.55 (128 vs 256 within 3 %), outer
   gas (3-6 kpc) not converged. complex128 costs nothing measurable on the P100 (N = 256: 2 h 46 vs 2 h 42). The remaining
   non-converged part is the gas numerical viscosity (single forward-Euler MUSCL stage).

11. Sink-sink gravity (added 2026-10-27): sinks used to move only in `forces_environment` (no sink potential, to avoid self-force),
   so they did not attract each other. `SinkNBody._pair_accelerations` now adds a direct pair force, applied through the
   `_interpolate_force_trilinear` override (only for N_sinks > 1; a single sink is bitwise unchanged). Each sink is a Gaussian
   source (BH part softening_bh, reservoir part softening_cusp) that the others feel as a point mass (the kernel of the sink
   potential and of the energy log; symmetric and momentum-conserving unless sinks carry different reservoir fractions),
   periodic minimum image, plus the mean-density term -(4 pi/3) G m r / V of the periodic
   Poisson solver (far field matches the grid; valid up to r ~ L/4). Test `tests/test_sink_binary_kepler.py` (circular binary:
   separation 1.6e-4, angular velocity 2.3e-4, momentum and angular momentum conserved; it failed before the change).
   P100 reference set and optimization checks stay bitwise identical (no reference case has 2+ sinks). Demos with plots:
   `tests/sink_grav1.py` (infall from rest, merge at 3.888 Gyr vs 3.89 expected for 2 x 1e7 Msun, 10 kpc, box 30 kpc) and
   `tests/sink_grav2.py` (circular orbit, period ~23 Gyr). Merging: `merge_close_sinks` exists but is only called from the
   disabled sink-formation path; the demos call it after every step. It averages positions without periodic wrapping.
   The kernel was first a symmetric s^2 = 2 eps^2 one; changed the same day to the point-receiver kernel so that the force is
   the gradient of the potential energy Diagnostics logs. Energy conservation along a binary orbit is not measured yet.
12. Sink merging (2026-10-27): `sim.sink_merging = True` (and optionally `sim.sink_merge_radius` in kpc, default = capture radius)
   makes Evolution call `SinkNBody.merge_close_sinks` before every step (`Evolution._perform_sink_merging`); off by default.
   It is a resolution-scale rule, not merger physics: bound sinks closer than the merge radius become one sink at their
   centre of mass (mass, momentum conserved; no recoil, no gravitational-wave loss). Three gaps fixed: (a) the energy lost in
   a merge (kinetic + softened potential, same model as Diagnostics) is added to `E_diss_merge_total` and Diagnostics now
   includes it in `E_diss` (before, `E_tot_cons` jumped at every merge); (b) the centre of mass uses the minimum-image
   displacement and is wrapped into the box (a pair across the box edge used to merge in the middle of the box); (c) the bound
   check uses the softened escape speed 2 G M erf(r / (sqrt(2) eps)) / r instead of the point-mass one. Test
   `tests/test_sink_merging.py` (written first, failed on the old code; the residual jump of E_total + E_diss at a merge is
   ~5 % of the energy lost, the grid-deposited vs analytic sink energy; tolerance 10 %). It cleans up its output folders.
   The merge radius sets when the merge happens, so report the sensitivity to it (and to N) in any result. Merging is
   cheap: a few small GPU->CPU copies per call, well under 0.1 % of a step.
13. Dynamical friction test (2026-10-27): `tests/sink_friction.py` sends a sink (3e6 Msun, 30 kpc/Gyr) through a uniform psi
   (3e5 Msun/kpc^3, m = 2.5e-22 eV, quantum Mach number M_Q = 17.4, 19.2 kpc box, half a box of travel) and measures the drag
   coefficient C = F / F_rel, F_rel = 4 pi rho (G M / v)^2, from the sink's dv/dt. References: (1) exact linear theory for this
   very box (discrete mode sum: periodic, Gaussian sink of width `softening_bh`, switched on at t = 0, psi self-gravity included;
   u'' + Omega^2 u = -psi0 k^2 Phi_k exp(-i k.v t), Omega^2 = hbar^2 k^4 / 4 - 4 pi G rho0), (2) Lancaster et al. 2020
   (arXiv:1909.06381) Eq. 4.26 for a point source in an infinite medium with b = v t (formula checked against the paper text).
   Mean C for v t / lambda > 7, simulation / linear theory / literature: softening 1.5 cells 2.70 / 2.71 / 3.27, 1.0 cell 2.95 / 2.97,
   0.5 cell 3.17 / 3.17, and N = 256 with 0.25 cell (run by the user) 3.28 / 3.25 / 3.27; rms difference simulation vs theory
   0.5-1.1 %. The theory itself changes by < 0.3 % below 0.25 cell, so smaller softening is pointless. Rule of thumb:
   softening_bh <~ 0.3 lambda (lambda = hbar / (m v)) and lambda >= 3 cells. Not covered: the nonlinear regime M_Q <~ 1,
   a soliton background, conservation of the total momentum sink + psi (the sink feels the wake at a point, psi feels the
   Gaussian sink). The N = 256 run used H = 4e-4, above the printed stability limit (1.5e-4), with no visible harm.
14. Friction scans (2026-10-29): `tests/sink_friction_scan.py` (SCAN = "mq" or "softening", N = 128, plots in
   `resources/data/sink_friction_scan_<SCAN>/`). T1: mean drag coefficient for v t / lambda > 7 at M_Q = 100, 30, 10, 3, 1, 0.3
   (softening 0.29 lambda): simulation 3.148, 3.159, 3.185, 3.130, 2.788, 2.222; exact linear theory for the box 3.174, 3.174, 3.174,
   3.152, 3.132, 3.125; exact steady-state point-source formula (Lancaster Eq. 4.21, own implementation, tends to Eq. 4.26 for large M_Q)
   3.267, 3.271, 3.275, 3.216, 2.783, 1.923. Simulation = linear theory within 0.8 % for M_Q >= 3; at M_Q <= 1 the coefficient oscillates
   strongly (mean equals the steady state at M_Q = 1, 15 % above it at 0.3). T3: the momentum gained by psi differs from the momentum lost by
   the sink by -13.0 / -8.0 / -1.9 / +0.3 % for softening 0.86 / 0.57 / 0.29 / 0.14 lambda (the sink feels the wake at a point, psi feels the
   Gaussian sink). Fix implemented 2026-10-29 (uncommitted): `SinkNBody(smooth_force=True)` filters the force on the sink with the same Gaussian,
   momentum balance error drops to +0.3...+2 % for all softening and M_Q (test `tests/test_sink_force_smoothing.py`, 4.6e-1 -> 2e-15); default False;
   friction at finite softening is lower (Gaussian squared in the linear theory), same point-source limit. Details and caveats in `plans.md`.

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

## Research plan (paper idea, literature numbers, test list): `plans.md` (git-ignored, exists only on Koios)

## Handoff 2026-10-27 — next tasks (read this first in a new session)
Goal of the next phase: **black-hole collisions inside a soliton** (ULDM soliton 5.3e7 Msun, sinks as black holes). Gas is NOT
needed for this (drop it; its numerical viscosity and outer-ring problems are then irrelevant).

**RESOLVED later on 2026-10-27 (issue 11 above): sink-sink gravity is implemented and tested. Text below is the original diagnosis.**
**Blocker found (2026-10-27): sinks do not attract each other.** Two sinks of 5e10 Msun, 4 kpc apart, no psi, 0.1 Gyr: the
separation stays exactly 4.0000 kpc (Newton would collapse them in ~0.01 Gyr). Sinks move only in `forces_environment`
(force from the density without any sink potential, `Evolution._evolve_order_2`), so there is no sink-sink gravity. Reproducer:
`~/conv/exp_twosinks.py` (run on a GPU node with `PYTHONPATH=$PWD`). Only this configuration (no psi) was measured; the cause
in the code was not investigated further. Changing it is a physics change: explain first, say which reference sets change.

Ordered plan (agreed with the user):
1. DONE (issue 11): add sink-sink gravity (softened, consistent with the analytic sink potential), plus a regression test with a two-sink
   Kepler binary (period, energy, centre of mass). Style of the existing tests: `tests/test_particle_kepler_orbit.py`.
2. Dynamical-friction test: a sink moving through psi (uniform background or soliton) against the literature formula (Hui et
   al. 2017, Lancaster et al. 2020 — recalled from memory, verify before using). This also becomes the validation set for GAMER.
3. Resolution check: BH of 2e5 Msun (0.4 % of the soliton) dominates only inside ~150 pc (rough estimate, constant core density)
   = 2 cells at N = 256 (dx = 78 pc). A BH of ~1e7 Msun dominates inside ~0.5 kpc (resolved). Prototype with the heavy BH.
   Uniform N = 512 would cost ~60 h per Gyr on a P100 (extrapolated from N = 128 -> 256: 7.4 min -> 166 min), so a scan needs AMR.
4. GAMER (https://github.com/gamer-project/gamer): GPU + AMR + ELBDM (wave and hybrid schemes, refinement flag for
   interference). Its wiki does not document ELBDM + particles or BH sinks (sink particles exist only for gas in a 2025 paper):
   ask the developers on GitHub before deciding. Koios has no `nvcc` (try pip CUDA toolkit wheels or Phoebe). Decide after 1-3.
5. Maintenance so the code is "functional/publishable": push (see below), `requirements.txt` is UTF-16, `.idea` is tracked,
   duplicate `system_fucntions.py`, README (units, minimal example), licence and citation, self-interaction potential (point 3:
   fix or switch off and document), regenerate the Quadro/RTX/GTX reference sets on Windows (stale for ALL cases since complex128).

Physics notes for this phase: an isolated soliton is stationary; the ~10 % breathing seen in main.py is a coherent mode excited by
the baryons (physical, user's view). The random "soliton walk" appears only if the soliton sits in a halo with interfering
waves; main.py has no such halo. A circular orbit with v = sqrt(GM/r) is not circular in the periodic box (use r0 <~ L/10 or the
grid force); the sink field is usable up to r <= L/4.

State of the repo and data (Koios, `~/projects/research_task`, branch `working_with_chat`):
- Commit `0d11459` (tests, minimal orbit script, docs) and `a236daa` (complex128) exist ONLY on Koios: `git push` over https needs
  a GitHub token (the user does it). Uncommitted: the convergence notes in CLAUDE.md/CONTINUATION.md and this handoff,
  `tests/test_sink_oscillation_in_soliton.py` (passes, registered in `tests/run_all.py`), untracked `slurm/koios_*_c128.sbatch`,
  `koios_convergence_h.sbatch` (one-off scripts; do not commit).
- `resources/data/` holds ~54 GB (home has no backup). Proposed deletions (not yet approved), exact names in
  `~/conv/delete_tier_A.txt` (34 small test runs, 0.6 GB), `_B.txt` (8 older exploratory runs, 8.6 GB), `_C.txt` (4 superseded
  complex64 N = 256 / N = 128 h = 1.25e-4 runs, 18 GB). Delete only by exact names the user approves.
- Helper scripts (outside the repo, in `~/conv/`): `conv_metrics.py` (convergence table), `plot_particle_tests.py` (plots of tests
  A-C), `plot_sink_osc.py`, `exp_twosinks.py`. Plots from this session are in `plots/particle_tests/` on the Windows machine.
- Connection: PowerShell `ssh koios` (the Windows ssh-agent already holds the key). Long jobs: submit with `sbatch` (a long
  `srun` over ssh drops the connection); only one P100 is usually free, so do not leave an interactive `srun --pty bash` open.
