"""Soliton onset time versus the effective number of independent polarizations n_eff, from the saved cases of spin_gravity_comparison.py.

Usage (repository root, project venv, no GPU needed):
    python tests/spin/onset_times.py [--data resources/data] [--out resources/data/spin_onset_vs_neff.png]

Onset time: the baseline is the median of rho_max / rho_mean in 2-6 Gyr (halo formed, core not yet growing); the onset is the first time
after 6 Gyr at which rho_max / rho_mean exceeds k times the baseline, for k = 3 and 5. It is a proxy for the soliton formation time and
is reliable at N = 64 (the values of rho_max after the onset are not). n_eff = 1 / sum f_m^2 for the mass fractions f_m of the
polarizations. Every seed file case_s1_N64_seed*.npz of a result folder is used; add a result folder to CASES to include more cases."""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# n_eff -> TAG of resources/data/spin_gravity_comparison<TAG>/ (spin 1, N = 64, runs made with FRACTIONS = f_1,f_2,f_3)
CASES = {
    1.0: "_frac100",        # FRACTIONS=1,0,0        (one component = scalar field)
    1.0 / 0.66: "_frac80",  # FRACTIONS=0.8,0.1,0.1
    1.0 / 0.44: "_frac60",  # FRACTIONS=0.6,0.2,0.2
    3.0: "_long50",         # equal fractions, 50 Gyr
    5.0: "_long_s2",        # spin 2, equal fractions, 150 Gyr (case_s2_*)
    6.0: "_N6_long",        # 6 independent equal components (s = 2.5), 150 Gyr
    8.0: "_N8_long",        # 8 components (s = 3.5)
    9.0: "_N9_long",        # 9 components (s = 4)
}
K_VALUES = (3, 5)


def onset(case, k):
    t, rho_max = case["time"], case["rho_max"]
    baseline = np.median(rho_max[(t >= 2) & (t <= 6)])
    above = np.nonzero((t > 6) & (rho_max > k * baseline))[0]
    return t[above[0]] if len(above) else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join("resources", "data"))
    parser.add_argument("--out", default=os.path.join("resources", "data", "spin_onset_vs_neff.png"))
    args = parser.parse_args()

    results = {k: {} for k in K_VALUES}
    print(f"{'n_eff':>6s}  {'seeds':>5s}  " + "  ".join(f"onset(k={k}) mean +- std [values]".ljust(44) for k in K_VALUES))
    for n_eff, tag in sorted(CASES.items()):
        files = sorted(glob.glob(os.path.join(args.data, f"spin_gravity_comparison{tag}", "case_s*_N64_seed*.npz")))
        cases = [dict(np.load(f)) for f in files]
        row = f"{n_eff:6.2f}  {len(cases):5d}  "
        for k in K_VALUES:
            values = np.array([onset(c, k) for c in cases])
            results[k][n_eff] = values
            good = values[np.isfinite(values)]
            spread = good.std(ddof=1) if len(good) > 1 else np.nan
            row += f"{good.mean():6.1f} +- {spread:4.1f} {np.round(values, 1)}".ljust(46) if len(good) else "-".ljust(46)
        print(row)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for k, marker in zip(K_VALUES, ("o", "s")):
        xs = np.array(sorted(results[k]))
        means = np.array([np.nanmean(results[k][x]) for x in xs])
        stds = np.array([np.nanstd(results[k][x], ddof=1) if np.sum(np.isfinite(results[k][x])) > 1 else 0.0 for x in xs])
        for x in xs:
            ax.plot([x] * len(results[k][x]), results[k][x], marker, color="0.7", ms=4)
        ax.errorbar(xs, means, yerr=stds, fmt=marker + "-", capsize=3, label=f"onset at {k} x baseline (mean +- std over seeds)")
    ax.set(xlabel="n_eff = 1 / sum f_m^2", ylabel="soliton onset time [Gyr]", xscale="log", yscale="log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
