from resources.Functions.system_fucntions import *


snapshot_directory = 'resources/data/simulation_20251112_113630' # Replace with your path

#-------------------------JUST WFS -----------------------------------------------------------
'''
wave_function_number = 6# Which wave function to plot

# Plot single wave function
data = plot_wave_function_snapshots(
snapshot_dir=snapshot_directory,
wf_number=wave_function_number,
z_index=None,  # Will use middle slice for 3D data
save_plots=True,
show_plots=False
    )
    

plot_multiple_wave_functions(
     snapshot_dir=snapshot_directory,
     wf_numbers=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],  # List of wave functions to compare
     save_plots=True,
     show_plots=False)


plot_wave_function_panel(
    snapshot_dir="resources/data/simulation_20251029_133606",
    wf_number=0,
    times=[2.5,5, 7.5,10, 12.5,15],  # pick any 4+
    ncols=3,                 # 2x2 grid
    z_index=None,            # middle slice for 3D
    log_scale=True,          # shared LogNorm
    figsize=(10, 9),
    fontsize=16,             # bigger labels/ticks
    save_path="wf0_panel.jpg",  # perfect for LaTeX
    dpi=900,
    show=False
)


#------------------------------------------BOTH WFS AND BARYONS-----------------------------------------------------------------------


snapshot_dir = "resources/data/simulation_20251112_115531"   # your snapshot folder
                                     # choose time (matches filename)
wf_idx = 0                                            # which ψ to plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import os

boundaries = [(-20, 20), (-20, 20), (-20, 20)]           # same as used in Simulation_Class
wf_index = 0                                             # which ψ field to plot
slice_axis = 2                                           # 0=x,1=y,2=z
time_target = 16                                      # pick saved time
# ===================

# --- load metadata ---
meta_path = os.path.join(snapshot_dir, "metadata.txt")
with open(meta_path) as f:
    lines = f.readlines()

# find grid size by reading one wf file
wf_files = sorted(glob.glob(os.path.join(snapshot_dir, f"wf_{wf_index}_snapshot_at_time_*.npy")))
if not wf_files:
    raise FileNotFoundError("No wavefunction snapshots found in folder")

# pick closest time
def extract_time(fname):
    try:
        return float(fname.split("at_time_")[1].replace(".npy", ""))
    except Exception:
        return np.inf

times = np.array([extract_time(f) for f in wf_files])
chosen_i = np.argmin(abs(times - time_target))
wf_file = wf_files[chosen_i]
actual_time = times[chosen_i]

# --- load data ---
psi = np.load(wf_file)
rho_wf = np.abs(psi) ** 2

# baryons (if exists)
baryon_path = os.path.join(snapshot_dir, f"baryons_snapshot_at_time_{actual_time:.6f}.npy")
rho_b = np.load(baryon_path) if os.path.exists(baryon_path) else None

# --- build coordinate grid ---
N = rho_wf.shape[0]
x = np.linspace(boundaries[0][0], boundaries[0][1], N, endpoint=False)
y = np.linspace(boundaries[1][0], boundaries[1][1], N, endpoint=False)
z = np.linspace(boundaries[2][0], boundaries[2][1], N, endpoint=False)
x_mesh_2d, y_mesh_2d = np.meshgrid(x, y)

# --- pick slice ---
z_index = N // 2 if slice_axis == 2 else None
wf_slice = rho_wf[:, :, z_index]
if rho_b is not None:
    baryon_slice = rho_b[:, :, z_index]

# --- prepare contour levels ---
pos_vals = wf_slice[wf_slice > 0]
levels = np.logspace(np.log10(pos_vals.min()), np.log10(pos_vals.max()), 128)

# --- plotting ---
plt.figure(figsize=(8, 6))
plt.contourf(x_mesh_2d, y_mesh_2d, wf_slice.T, origin="lower",
             levels=levels, cmap="viridis", norm=LogNorm())

if rho_b is not None:
    pos_b = baryon_slice[baryon_slice > 0]
    if pos_b.size > 0:
        b_levels = np.logspace(np.log10(pos_b.min()), np.log10(pos_b.max()), 8)
        plt.contour(x_mesh_2d, y_mesh_2d, baryon_slice.T,
                    levels=b_levels, colors="cyan", linewidths=0.8)

plt.colorbar(label="|ψ|²")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"t = {actual_time:.3f}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

'''
#------------------------------------------JUST BARYONS-------------------------------------------------

snapshot_dir = "resources/data/simulation_20251112_141916"  # your snapshot folder
boundaries = [(-20, 20), (-20, 20), (-20, 20)]              # same as in Simulation_Class
slice_axis = 2                                              # 0=x,1=y,2=z
time_target = 0.5                                           # pick saved time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob, os

# --- find baryon density snapshots ---
baryon_files = sorted(glob.glob(os.path.join(snapshot_dir, "baryons_snapshot_at_time_*.npy")))
if not baryon_files:
    raise FileNotFoundError("No baryon snapshots found in folder")

def extract_time(fname):
    try:
        return float(fname.split("at_time_")[1].replace(".npy", ""))
    except Exception:
        return np.inf

times = np.array([extract_time(f) for f in baryon_files])
chosen_i = np.argmin(abs(times - time_target))
baryon_file = baryon_files[chosen_i]
actual_time = times[chosen_i]

# --- load baryon density grid ---
rho_b = np.load(baryon_file)   # shape (N, N, N) on a regular grid

# --- build coordinate grid (periodic, endpoint=False to match your sim) ---
N = rho_b.shape[0]
x = np.linspace(boundaries[0][0], boundaries[0][1], N, endpoint=False)
y = np.linspace(boundaries[1][0], boundaries[1][1], N, endpoint=False)
z = np.linspace(boundaries[2][0], boundaries[2][1], N, endpoint=False)
x_mesh_2d, y_mesh_2d = np.meshgrid(x, y)

# --- pick slice ---
if slice_axis == 0:
    idx = N // 2
    baryon_slice = rho_b[idx, :, :]
elif slice_axis == 1:
    idx = N // 2
    baryon_slice = rho_b[:, idx, :]
else:
    idx = N // 2
    baryon_slice = rho_b[:, :, idx]

# --- prepare contour levels (log) ---
pos_b = baryon_slice[baryon_slice > 0]
if pos_b.size == 0:
    raise ValueError("Selected slice has no positive baryon density values to plot.")
levels = np.logspace(np.log10(pos_b.min()), np.log10(pos_b.max()), 128)

# --- plotting (match your style/orientation) ---
plt.figure(figsize=(8, 6))
plt.contourf(x_mesh_2d, y_mesh_2d, baryon_slice.T, origin="lower",
             levels=levels, cmap="viridis", norm=LogNorm())

plt.colorbar(label=r"$\rho_{\mathrm{baryons}}$")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"t = {actual_time:.3f}  (baryons only)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
