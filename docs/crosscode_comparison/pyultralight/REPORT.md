# PyUltraLight reference run: dynamical friction in uniform ULDM (Test 1) and BH in a soliton (Test 2)

All files are in `~/projects/pyultralight` on Koios. Nothing outside this directory was touched (`research_task` was only read).

## 1. Code, version, resources
- `src/` = auckland-cosmo/PyUltraLight, commit f80c1f22 (public PyUltraLight): only a fixed central mass (-cmass/r); no moving particle, no uniform background. **Not usable for this test.**
- `nbody/` = Sifyrena/PyUL_NBody, commit 879b8e7e, `PyUltraLight2/Current.py` ("PyUltraLight 2"). This is the code actually used (it has particles + backreaction). Whether it is exactly the code of arXiv:2403.09038 I did not verify.
- venv: `venv/` (EESSI 2025.06 Python 3.12.3 + SciPy-bundle 2024.05, numexpr, numba, pyfftw, h5py, numpy<2, ipython 8.30.0 pinned, because IPython 9 breaks an import in PyUL).
- **CPU only** (numexpr/pyfftw/numba). The GPU was never used; jobs ran on the `gpu` partition without `--gres` only because the CPU partitions were drained. Templates `job.sbatch`/`job2.sbatch` now point to `rocky10_cpu` (per user request).
- Wall time (8 threads): N=128 about 10 min, N=192 about 17 min, N=256 about 75-90 min; Test 2 (N=192, 8510 steps) 2.0 h.

## 2. What PyUL_NBody supports (checked in the code)
| Needed | Status |
|---|---|
| Periodic box | yes, FFT Poisson with k=0 removed (the mean density is dropped) |
| Uniform background psi = sqrt(rho0) | yes (`Uniform=True`, `Density`, `UVel`) |
| Moving gravitating Plummer particle | yes (`particles`, `GenPlummer`) |
| Backreaction | yes: particle potential acts on psi, force from psi acts on the particle (4^3-point stencil, trilinear interpolation, RK4) |
| Trajectory / velocity output | yes (`NBody/NTM_#k`, acceleration `DF/DYF_#k`) |

Differences from our code / not supported:
- The potential of the particle acting on psi is **not periodic**: `-a m / sqrt(1 + a^2 r^2)` evaluated in the box; only the psi self-gravity is periodic.
- The force on the particle is set to zero within about 4 cells of the box edge.
- Time step is fixed to dx^2/pi (hbar/m = 1). I override it with a smaller step (H in the table) by choosing more save steps; PyUL only accepts at least its own step count.
- Softening is Plummer only, with the particle as "point receiver" (no smooth-receiver variant).
- Bug: `evolve()` crashes with `UnboundLocalError: egpcmMlist` unless "Energy" is in the save options. I did not modify PyUL; the driver always enables Energy.
- Grid is cell-centered at -L/2 + i dx (the origin is a node).

## 3. Test 1 setup
Dimensionless run mode (hbar/m = G = 1, length unit lambda, v unit 1), so M_Q, eps/lambda and L/lambda are exactly the requested ones. One run in physical units (kpc, Msun, Gyr via PyUL constants) gives the same C (2.808 vs 2.808 at M_Q 1, N=128), so constants mismatch (~0.2 %) is not visible at this precision.
Box [-9.6, 9.6]^3, start x = -4.8 kpc, v = 30 kpc/Gyr along +x, 0.32 Gyr. C(t) uses the same estimator as `tests/sink_friction_scan.py` (central difference over +-1.6e-3 Gyr), averaged for v t/lambda > 7.
Two softening choices per N: eps = 0.075 kpc = 0.287 lambda (same as our reference; = 0.5, 0.75, 1.0 cell for N = 128, 192, 256), and eps = 0.5 cell (0.075, 0.05, 0.0375 kpc; **different physical eps, not comparable to our reference**, shown only for the eps trend).

## 3b. Table (all runs)

| run | N | eps [kpc] | eps [cells] | eps/lambda | H [Gyr] | M_Q | C(PyUL, b>7) | C(PyUL, from DF) | C ours point | C ours smooth | PyUL/point-1 | slowdown | p_psi/lost-1 (end) | (p_psi+M dv)/(M v0) | wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q0.3_N128_e075_H4e-4 | 128 | 0.0750 | 0.50 | 0.287 | 4.00e-04 | 0.3 | 2.253 | 2.236 | 2.222 | 2.029 | +1.4% | 1.76% | +0.016 | +2.86e-04 | 588 |
| q0.3_N192_e05c_H2.5e-4 | 192 | 0.0500 | 0.50 | 0.191 | 2.50e-04 | 0.3 | 2.213 | 2.203 | 2.172 | 2.007 | +1.9% | 1.72% | +0.052 | +8.95e-04 | 998 |
| q0.3_N192_e075_H2.5e-4 | 192 | 0.0750 | 0.75 | 0.287 | 2.50e-04 | 0.3 | 2.292 | 2.286 | 2.172 | 2.007 | +5.5% | 1.79% | -0.004 | -7.89e-05 | 1002 |
| q0.3_N256_e05c_H1.4e-4 | 256 | 0.0375 | 0.50 | 0.143 | 1.40e-04 | 0.3 | 2.217 | 2.222 | 2.180 | 2.013 | +1.7% | 1.73% | +0.065 | +1.12e-03 | 4301 |
| q0.3_N256_e075_H1.4e-4 | 256 | 0.0750 | 1.00 | 0.287 | 1.40e-04 | 0.3 | 2.309 | 2.303 | 2.180 | 2.013 | +5.9% | 1.81% | -0.014 | -2.45e-04 | 5317 |
| q17.4_N128_e075_H4e-4 | 128 | 0.0750 | 0.50 | 0.287 | 4.00e-04 | 17.4 | 3.161 | 3.153 | 3.170 | 3.088 | -0.3% | 0.75% | +0.060 | +4.50e-04 | 593 |
| q17.4_N192_e05c_H2.5e-4 | 192 | 0.0500 | 0.50 | 0.191 | 2.50e-04 | 17.4 | 3.255 | 3.255 | 3.185 | 3.102 | +2.2% | 0.77% | +0.064 | +4.99e-04 | 1012 |
| q17.4_N192_e075_H2.5e-4 | 192 | 0.0750 | 0.75 | 0.287 | 2.50e-04 | 17.4 | 3.185 | 3.185 | 3.185 | 3.102 | +0.0% | 0.76% | +0.044 | +3.34e-04 | 1004 |
| q17.4_N256_e05c_H1.4e-4 | 256 | 0.0375 | 0.50 | 0.143 | 1.40e-04 | 17.4 | 3.295 | 3.299 | 3.191 | 3.108 | +3.3% | 0.79% | +0.067 | +5.29e-04 | 4464 |
| q17.4_N256_e075_H1.4e-4 | 256 | 0.0750 | 1.00 | 0.287 | 1.40e-04 | 17.4 | 3.196 | 3.198 | 3.191 | 3.108 | +0.1% | 0.76% | +0.039 | +2.98e-04 | 4483 |
| q1_N128_e075_H2e-4 | 128 | 0.0750 | 0.50 | 0.287 | 2.00e-04 | 1 | 2.801 | 2.791 | 2.788 | 2.630 | +0.5% | 2.26% | +0.048 | +1.08e-03 | 1166 |
| q1_N128_e075_H4e-4 | 128 | 0.0750 | 0.50 | 0.287 | 4.00e-04 | 1 | 2.808 | 2.791 | 2.788 | 2.630 | +0.7% | 2.26% | +0.045 | +1.03e-03 | 601 |
| q1_N128_e075_H4e-4_phys | 128 | 0.0750 | 0.50 | 0.287 | 4.00e-04 | 1 | 2.808 | 2.792 | 2.788 | 2.630 | +0.7% | 2.26% | +0.045 | +1.02e-03 | 0 |
| q1_N128_e075_Hdef | 128 | 0.0750 | 0.50 | 0.287 | 9.12e-04 | 1 | 2.822 | 2.789 | 2.788 | 2.630 | +1.2% | 2.28% | +0.039 | +8.80e-04 | 254 |
| q1_N192_e05c_H2.5e-4 | 192 | 0.0500 | 0.50 | 0.191 | 2.50e-04 | 1 | 2.836 | 2.830 | 2.820 | 2.647 | +0.6% | 2.30% | +0.053 | +1.21e-03 | 1028 |
| q1_N192_e075_H2.5e-4 | 192 | 0.0750 | 0.75 | 0.287 | 2.50e-04 | 1 | 2.869 | 2.861 | 2.820 | 2.647 | +1.8% | 2.33% | +0.011 | +2.57e-04 | 1028 |
| q1_N256_e05c_H1.4e-4 | 256 | 0.0375 | 0.50 | 0.143 | 1.40e-04 | 1 | 2.856 | 2.855 | 2.838 | 2.659 | +0.6% | 2.33% | +0.057 | +1.33e-03 | 5426 |
| q1_N256_e075_H1.4e-4 | 256 | 0.0750 | 1.00 | 0.287 | 1.40e-04 | 1 | 2.895 | 2.887 | 2.838 | 2.659 | +2.0% | 2.35% | +0.001 | +1.30e-05 | 5429 |
| test_MQ1_N128 | 128 | 0.0750 | 0.50 | 0.287 | 4.00e-04 | 1 | 2.808 | 2.791 | 2.788 | 2.630 | +0.7% | 2.26% | +0.045 | +1.03e-03 | 0 |

Columns: "C ours point/smooth" are the reference values of our code (`sink_friction_scan_t2_*`, eps 0.29 lambda); the correct comparison is "point". "PyUL/point-1" is only meaningful for rows with eps/lambda = 0.287. Plot: `friction_curves.png` (C(t) vs v t/lambda, PyUL runs coloured, ours black). Table also in `analysis_table.md`, raw results in `runs/*/result_*.npz`.

## 4. Test 1 results (same eps = 0.287 lambda, vs our point receiver)
| M_Q | N=128 | N=192 | N=256 | ours (128/192/256) | linear theory / Lancaster+20 |
|---|---|---|---|---|---|
| 17.4 | 3.161 (-0.3 %) | 3.185 (0.0 %) | 3.196 (+0.1 %) | 3.170 / 3.185 / 3.191 | 3.17 / 3.27 |
| 1 | 2.808 (+0.7 %) | 2.869 (+1.8 %) | 2.895 (+2.0 %) | 2.788 / 2.820 / 2.838 | - |
| 0.3 | 2.253 (+1.4 %) | 2.292 (+5.5 %) | 2.309 (+5.9 %) | 2.222 / 2.172 / 2.180 | - |

- M_Q 17.4 agrees to 0.3 % at all N. M_Q 1 agrees to 0.7-2 %, with the difference growing with N. M_Q 0.3 agrees to 1.4 % at N=128 but is 5.5-5.9 % higher at N=192/256; there our own values go slightly down with N and PyUL's go up. **I did not find the cause**; candidates (unverified): non-periodic particle potential, a different receiver/force interpolation, forced zero force near the edge. At M_Q 0.3 C(t) oscillates strongly (see plot), so the b>7 mean is also sensitive to oscillation phase.
- Time step (M_Q 1, N=128): H = 2e-4 / 4e-4 / 9.12e-4 (PyUL default) gives 2.801 / 2.808 / 2.822, so the default step alone biases C by about +0.5 %. Runs use H = 4e-4 (N=128), 2.5e-4 (192), 1.4e-4 (256).
- eps = 0.5 cell runs (eps = 0.075, 0.05, 0.0375 kpc): C rises as eps decreases at M_Q 17.4 (3.161 / 3.255 / 3.295) and M_Q 1 (2.808 / 2.836 / 2.856), as expected for less softening; for M_Q 0.3 it is 2.253 / 2.213 / 2.217.
- Momentum balance: (psi momentum + M dv)/(M v0) at the end is between -2.5e-4 and +1.3e-3 for all runs, i.e. total momentum is conserved to about 1e-3 of the initial point momentum. The momentum gained by psi relative to that lost by the point differs: for eps = 0.287 lambda between -1.4 % and +6 % (smallest at N=256, 0.1 to -1.4 %; largest at N=128, 4-6 % except M_Q 0.3), and +5 to +7 % for the 0.5-cell runs at N=192/256 (smaller eps). Likely due to the non-periodic particle potential (not verified).

## 5. Test 2 (setup of arXiv:2403.09038), one run only
Soliton 5.57e7 Msun, BH 4e6 Msun at rest at (0.125, 0.125, 0) kpc, box 2 kpc **periodic**, N=192 (dx = 10.4 pc), eps = 0.5 cell = 5.2 pc, 8510 steps, `isop=0` (PyUL's standard soliton radial profile loaded, not Schive Eq. 4 and not a numerically relaxed state).
- t_stop (BH speed < 5 % of its first peak) = 8.2 Myr; t_stop from sign change of the radial velocity = 8.4 Myr. Paper fit A/M + B = 46.9 Myr. **No agreement**; my definitions of t_stop may not be the paper's.
- max |BH - soliton COM| = 177 pc, final 23 pc; the soliton COM itself drifts about 26 pc (mean), which is not negligible compared to the 177 pc BH offset.
- Not tested: `isop=1`, other N or eps, non-periodic boundaries, relaxing the soliton first, other t_stop definitions. The difference to the paper could come from any of these (including the periodic box), so I do not draw a conclusion about the paper.

## 6. What is comparable and what is not
Comparable: M_Q, rho0, M, v, L, N, eps/lambda (0.287 lambda), run time, estimator for C(t) and window are identical to our reference; the "point receiver" column corresponds to the same measurement.
Not identical / not comparable: particle-to-psi potential is non-periodic in PyUL (periodic in ours, if our code uses the periodic one); receiver stencil and edge zeroing; time stepping (RK4 particle + split-step, fixed H vs ours); 0.5-cell runs use a different eps than our reference; constants (checked to give the same C in one case only).

## 7. Not tested
N=320; time-step convergence other than M_Q 1 at N=128; a periodic particle potential (would require modifying PyUL, which I did not do); GPU execution (the code is CPU only); the cause of the +5.5 % at M_Q 0.3; the cause of the psi/point momentum offset; seeds/phase of the initial psi (uniform, no noise); Test 2 variants (section 5).
