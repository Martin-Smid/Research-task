from pathlib import Path
import glob
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# =====================================================================================================
# CPU analysis of a T5 run (tests/t5_bh_soliton.py): python tests/t5_analyze.py <simulation folder> [M_BH]
# Prints: black hole speed peaks (the envelope of the oscillation) and the stopping time, black hole position relative to the density peak,
# energy drift, momentum of psi + BH from the wf_0 snapshots (spectral derivative). Saves t5_analysis.png in the simulation folder.
# The stopping time of arXiv:2403.09038 is the time at which the amplitude of the BH velocity oscillation is smallest before it grows again
# ("reheating", Fig. 6-8 of the paper); their exact algorithm is not known to me, here: the minimum of the running-maximum envelope of the speed.
# =====================================================================================================

KMS = 0.9778          # 1 kpc/Gyr in km/s
d = sys.argv[1]
M_BH = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0e6
cfg = json.load(open(d + "/run_config.json"))["simulation"]
N = cfg["N"]
lo, hi = cfg["boundaries"][0]
dx = (hi - lo) / N
HBAR_T = 1.9611    # hbar / m for m = 1e-21 eV [kpc^2/Gyr]

traj = np.genfromtxt(d + "/particle_trajectory.csv", delimiter=",", names=True)
t = traj["time"] * 1e3                                       # Myr
pos = np.array([traj["x"], traj["y"], traj["z"]])
vel = np.gradient(pos, traj["time"], axis=1)                 # kpc/Gyr
speed = np.linalg.norm(vel, axis=0) * KMS
signed = vel[0] * KMS

# the peak of the density (a cell-quantised proxy of the soliton centre)
loc = np.genfromtxt(d + "/max_locations.txt", delimiter=",", comments="#")
print(f"{d}: N {N}, {len(t)} steps, {t[-1]:.1f} Myr, peak speed {speed.max():.1f} km/s")
print("density peak position [pc]: min %s  max %s  (cell %.2f pc)" % (np.round(1e3 * loc[:, 4:7].min(0), 1), np.round(1e3 * loc[:, 4:7].max(0), 1), dx * 1e3))

# envelope of the speed oscillation
smooth_n = max(3, int(0.5 / (t[1] - t[0])))                  # 0.5 Myr running mean
sp = np.convolve(speed, np.ones(smooth_n) / smooth_n, mode="same")
peaks, _ = find_peaks(sp, prominence=0.2 * sp.max())
print("speed peaks [Myr: km/s]:", ", ".join(f"{t[p]:.1f}: {sp[p]:.1f}" for p in peaks))
# envelope = running maximum of the speed over a window of one oscillation (10 Myr); the stopping time is the minimum of the envelope after the first
# 15 Myr (the peak sequence itself is not monotonic: successive peaks can differ by +-5 km/s)
win = max(3, int(10.0 / (t[1] - t[0])))
env = np.array([sp[max(0, i - win // 2):i + win // 2 + 1].max() for i in range(len(sp))])
late = (t > 15.0) & (t < t[-1] - 5.0)
t_stop = t[late][np.argmin(env[late])] if late.any() else None
print("minimum of the speed envelope (stopping time estimate): " + (f"{t_stop:.1f} Myr, envelope {env[late].min():.1f} km/s" if t_stop else "not reached"))

en = np.genfromtxt(d + "/energy.csv", delimiter=",", names=True)
print("E_total relative change: max |dE/E0| = %.2e, last = %.2e" % (np.abs(en["E_total"] / en["E_total"][0] - 1).max(), en["E_total"][-1] / en["E_total"][0] - 1))

# momentum of psi + BH from the snapshots
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
files = glob.glob(d + "/wf_0_snapshot_at_time_*.npy")


def snap_time(f):
    m = re.search(r"time_([0-9.e+-]+)\.npy", f)
    return float(m.group(1)) if m else 0.0


files.sort(key=snap_time)
rows = []
for f in files:
    ts = snap_time(f)
    psi = np.load(f)
    ft = np.fft.fftn(psi)
    P = np.array([HBAR_T * np.sum((np.conj(psi) * np.fft.ifftn(1j * k.reshape([-1 if i == ax else 1 for i in range(3)]) * ft)).imag) * dx ** 3 for ax in range(3)])
    i = int(np.clip(np.searchsorted(traj["time"], ts), 0, len(t) - 1))
    rows.append((ts * 1e3, *P, *(M_BH * vel[:, i])))
rows = np.array(rows)
if len(rows):
    tot = rows[:, 1:4] + rows[:, 4:7]
    print("t[Myr]  |P_psi|  |P_BH|  |P_psi+P_BH|  [Msun kpc/Gyr]")
    for r, tt in zip(rows, tot):
        print("%7.2f  %.3e  %.3e  %.3e" % (r[0], np.linalg.norm(r[1:4]), np.linalg.norm(r[4:7]), np.linalg.norm(tt)))

fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
ax[0].plot(t, signed, label="v_x")
ax[0].plot(t, sp, "k", lw=0.8, label="speed (0.5 Myr mean)")
ax[0].plot(t[peaks], sp[peaks], "ro")
if t_stop:
    ax[0].axvline(t_stop, color="gray")
ax[0].set_ylabel("BH velocity [km/s]")
ax[0].legend()
ax[1].plot(t, 1e3 * pos[0], label="BH x")
ax[1].plot(loc[:, 0] * 1e3, 1e3 * loc[:, 4], label="density peak x", lw=0.8)
ax[1].set_ylabel("x [pc]")
ax[1].legend()
ax[2].plot(en["time"] * 1e3, en["E_total"] / en["E_total"][0] - 1)
ax[2].set_ylabel("E_total / E_0 - 1")
ax[2].set_xlabel("time [Myr]")
fig.tight_layout()
fig.savefig(d + "/t5_analysis.png", dpi=130)
print("plot saved:", d + "/t5_analysis.png")
