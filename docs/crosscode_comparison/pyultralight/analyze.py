import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.expanduser("~/projects/research_task/resources/data")
REF = {128: "sink_friction_scan_t2_N128_H4e-4_s0", 192: "sink_friction_scan_t2_N192_s0", 256: "sink_friction_scan_t2_N256_s0"}
def ref(N, q, smooth=False):
    d = REF[N] if not smooth else REF[N].replace("scan_t2_", "scan_t2_smooth_").replace("_s0", "_s1")
    f = glob.glob(f"{R}/{d}/case_MQ{q:g}_N{N}_*.npz")
    return np.load(f[0], allow_pickle=True) if f else None
runs = sorted(glob.glob(os.path.expanduser("~/projects/pyultralight/runs/*/result_*.npz")))
rows = []
for f in runs:
    r = np.load(f, allow_pickle=True)
    if "mq" not in r.files: continue
    rows.append((os.path.basename(f)[7:-4], r))
print("| run | N | eps [kpc] | eps [cells] | eps/lambda | H [Gyr] | M_Q | C(PyUL, b>7) | C(PyUL, from DF) | C ours point | C ours smooth | PyUL/point-1 | slowdown | p_psi/lost-1 (end) | (p_psi+M dv)/(M v0) | wall [s] |")
print("|" + "---|" * 16)
for tag, r in rows:
    N = int(r["N"]); q = float(r["mq"])
    rp = ref(N, q) if N in REF else None; rs = ref(N, q, True) if N in REF else None
    cp = float(rp["mean_sim"]) if rp is not None else np.nan; cs = float(rs["mean_sim"]) if rs is not None else np.nan
    print(f"| {tag} | {N} | {float(r['eps_kpc']):.4f} | {float(r['eps_cells']):.2f} | {float(r['eps_lambda']):.3f} | {float(r['H_gyr']):.2e} | {q:g} | {float(r['mean_c']):.3f} | {float(r['mean_c_df']):.3f} | {cp:.3f} | {cs:.3f} | {100*(float(r['mean_c'])/cp-1):+.1f}% | {100*float(r['slowdown']):.2f}% | {float(r['p_end_over_lost'])-1:+.3f} | {float(r['momentum_sum_end_rel']):+.2e} | {float(r['wall_s']):.0f} |")
# curves: production runs only (eps 0.075 kpc, tags with e075 and no H variant / phys)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, q in zip(axs, (17.4, 1.0, 0.3)):
    for tag, r in rows:
        if float(r["mq"]) != q or "phys" in tag or "Hdef" in tag or "H2e-4" in tag: continue
        N = int(r["N"])
        ax.plot(r["b"], r["c"], label=f"PyUL_NBody {tag.split('_')[1]} {tag.split('_')[2]}")
        for sm, ls in ((False, "--"), (True, ":")):
            rr = ref(N, q, sm) if N in REF else None
            if rr is not None:
                ax.plot(rr["b_tilde"], rr["c_sim"], ls, color="k", alpha=0.6, label=f"ours N={N} {'smooth' if sm else 'point'}")
    ax.set_title(f"M_Q = {q:g}"); ax.set_xlabel("v t / lambda"); ax.set_ylabel("C(t)"); ax.axvline(7, color="gray", lw=0.5); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.expanduser("~/projects/pyultralight/friction_curves.png"), dpi=130)