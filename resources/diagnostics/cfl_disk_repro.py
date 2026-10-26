"""Reproducer + instrumentation for the open gas CFL blow-up (main.py disk IC times the cos^2 window).

Usage: python resources/diagnostics/cfl_disk_repro.py [window|nowindow|window_rho|window_v] [h]
Prints the CFL signal speed (|v|max + cs_max) at every call and where it sits, and stops at the first jump.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import chats_playground  # noqa: F401  (CUDA/CuPy runtime setup)
import cupy as cp
import numpy as np

from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Simulation_Class import Simulation_Class

variant = sys.argv[1] if len(sys.argv) > 1 else "window"
h = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
flags = sys.argv[3].split(",") if len(sys.argv) > 3 else []

sim = Simulation_Class(dim=3, boundaries=[(-10, 10)] * 3, N=128, total_time=0.05, h=h, order_of_evolution=2,
                       use_gravity="nograv" not in flags, use_units=True, use_sponge=False, self_int=False, save_max_vals=False)
disk_mass, scale_radius, scale_height, rotation_speed, sound_speed, core_radius = 1.0e9, 2.0, 0.5, 40.0, 10.0, 0.5
KMS = 1.022712165
x, y, z = [cp.asarray(g) for g in sim.grids]
height = (x + y) / cp.sqrt(2.0)
u = (x - y) / cp.sqrt(2.0)
R = cp.sqrt(u**2 + z**2)
rho = cp.exp(-R / scale_radius) * cp.exp(-0.5 * (height / scale_height) ** 2)
rho = cp.maximum(rho, 1e-12)
rho *= disk_mass / (cp.sum(rho) * sim.dV)
omega = rotation_speed * KMS / cp.sqrt(R**2 + core_radius**2)
vx, vy, vz = omega * z / cp.sqrt(2.0), -omega * z / cp.sqrt(2.0), -omega * u
window = (cp.cos(cp.pi * x / 20) * cp.cos(cp.pi * y / 20) * cp.cos(cp.pi * z / 20)) ** 2
if variant in ("window", "window_rho", "window_floor"):
    rho = rho * window
if variant == "window_floor":
    rho = cp.maximum(rho, 1e-12)  # window first, THEN the floor (main.py applies the floor before the window)
if variant in ("window", "window_v", "window_floor"):
    vx, vy, vz = vx * window, vy * window, vz * window
print(f"variant={variant} h={h}: rho min/max {float(rho.min()):.3e}/{float(rho.max()):.3e}, "
      f"cells below rho_floor 1e-12: {int((rho < 1e-12).sum())}, |v| max {float(cp.sqrt(vx**2+vy**2+vz**2).max()):.3e} kpc/Gyr")
gas = NBodyGas(simulation=sim, rho=rho, vx=vx, vy=vy, vz=vz, cs=sound_speed * KMS, gamma=5.0 / 3.0,
               cfl=0.4, max_substeps=100, name="gas")
sim.add_baryons(gas)

if "nocorr" in flags:
    from resources.Classes.Evolution_Class import Evolution_Class
    Evolution_Class._set_gas_potential = lambda self, a, b, c: None  # old gas behaviour: no face work / kick-drift-kick
if "trace" in flags:
    # Follow the specific internal energy eps = (E - K)/rho (as the solver sees it, rho floored) through
    # every operation of the FIRST hydro step; initially eps = cs^2/(gamma(gamma-1)) everywhere.
    state = {"step": 0, "on": False}

    def report(label, rho, moms, E):
        rs = cp.maximum(rho, gas.rho_floor)
        v2 = sum((m / rs) ** 2 for m in moms)
        eps = (E - 0.5 * rs * v2) / rs
        i = where(eps)
        nb = []
        for ax in range(3):
            for d in (-1, 1):
                j = list(i)
                j[ax] = (j[ax] + d) % rho.shape[ax]
                j = tuple(j)
                nb.append(f"ax{ax}{'+' if d > 0 else '-'}: rho={float(rho[j]):.1e} eps={float(eps[j]):.1e}")
        print(f"  [{label}] eps max {float(eps.max()):.3e} (min {float(eps.min()):.3e}) at {i}: rho={float(rho[i]):.2e} "
              f"E={float(E[i]):.2e} |m|={float(cp.sqrt(sum(m[i] ** 2 for m in moms))):.2e}; neighbours: " + "; ".join(nb),
              flush=True)

    orig_nd, orig_sweep, orig_reset = gas._hydro_step_nd, gas._sweep_axis, gas._reset_emptied_cells

    def nd(dt):
        state["step"] += 1
        if state["step"] == 1:
            print(f"\nFIRST hydro substep dt={dt:.3e}")
            report("initial", gas.rho, [gas.rho * v for v in gas.vel], gas.E)
        return orig_nd(dt)

    orig_pfc = gas._pressure_from_conserved

    def pfc(rho, moms, E):
        P = orig_pfc(rho, moms, E)
        if state["on"]:
            c = cp.sqrt(gas.gamma * P / rho)
            i = where(c)
            if float(c[i]) > state.get("cmax", (0,))[0]:
                state["cmax"] = (float(c[i]), i, float(rho[i]), float(P[i]), float(E[i]))
        return P

    gas._pressure_from_conserved = pfc

    def sweep(rho, moms, E, axis, dt, dx):
        state["cmax"] = (0.0,)
        out = orig_sweep(rho, moms, E, axis, dt, dx)
        if state["on"]:
            cm = state["cmax"]
            print(f"  [Courant axis {axis}] face sound speed max c={cm[0]:.3e} at {cm[1]}: face rho={cm[2]:.2e} P={cm[3]:.2e} E={cm[4]:.2e}"
                  f" -> dt/dx*c = {dt / dx * cm[0]:.3e} (stable only if < 1)", flush=True)
        if state["on"]:
            report(f"after sweep axis {axis}", *out)
            line = [(64, 64, k) for k in (125, 126, 127, 0, 1, 2, 3)]
            print(f"  [mass after sweep axis {axis}] {float(out[0].sum()) * gas.cell_volume:.4e} (before {float(rho.sum()) * gas.cell_volume:.4e}); "
                  f"z-line at x=y=64, z=125..3: rho before " + " ".join(f"{float(rho[i]):.1e}" for i in line)
                  + " | after " + " ".join(f"{float(out[0][i]):.1e}" for i in line)
                  + " | m_z before " + " ".join(f"{float(moms[2][i]):.1e}" for i in line), flush=True)
        return out

    def reset(rho, moms, E):
        out = orig_reset(rho, moms, E)
        if state["on"]:
            report("after reset      ", *out)
        return out

    gas._hydro_step_nd, gas._sweep_axis, gas._reset_emptied_cells = nd, sweep, reset

    def now(label):
        report(label, gas.rho, [gas.rho * v for v in gas.vel], gas.E)

    orig_drift, orig_kick, orig_corr, orig_hs = gas.drift, gas._gravity_kick, gas._add_gravity_energy_correction, gas._hydro_step

    def drift(*a, **k):
        if not state["on"] and state["step"] == 0:
            state["on"] = True
            now("drift start      ")
            state["rho0"] = gas.rho.copy()
            state["phi_old"] = gas.phi_total.copy()
            orig_pu = gas.potential_update

            def pu(d_rho):
                out = orig_pu(d_rho)
                state["phi_new"] = out[1]
                return out
            gas.potential_update = pu
        out = orig_drift(*a, **k)
        state["on"] = False
        state["step"] = max(state["step"], 2)
        return out

    def hs(dt):
        orig_hs(dt)
        if state["on"]:
            now("end of hydro step")

    def kick(ax, ay, az, dt):
        out = orig_kick(ax, ay, az, dt)
        if state["on"]:
            now(f"after kick dt={dt:.1e}")
        return out

    def corr(c):
        if state["on"]:
            i = where(cp.abs(c))
            print(f"  [correction to add] max |c| {float(cp.abs(c).max()):.3e} at {i}: rho={float(gas.rho[i]):.2e} "
                  f"c={float(c[i]):.2e} -> c/rho={float(c[i] / gas.rho[i]):.2e}; sum c*dV={float(c.sum()) * gas.cell_volume:.3e}", flush=True)
        if state["on"]:
            d_rho = gas.rho - state["rho0"]
            dphi = state["phi_new"] - state["phi_old"]
            tw = 0.5 * d_rho * dphi
            for tag, i in (("max|c| cell", where(cp.abs(c))), ("vacuum cell that blows up (116,12,127)", (116, 12, 127))):
                print(f"  terms at {tag} {i}: rho_old={float(state['rho0'][i]):.2e} rho_new={float(gas.rho[i]):.2e} "
                      f"d_rho={float(d_rho[i]):.2e} phi_old={float(state['phi_old'][i]):.3e} dphi={float(dphi[i]):.3e} "
                      f"face_work={float(gas._face_work[i]):.3e} -0.5*d_rho*dphi={-float(tw[i]):.3e} c={float(c[i]):.3e}", flush=True)
            print(f"  rho_new max {float(gas.rho.max()):.3e} at {where(gas.rho)}; |d_rho| max {float(cp.abs(d_rho).max()):.3e}; "
                  f"|dphi| max {float(cp.abs(dphi).max()):.3e}; |face_work| max {float(cp.abs(gas._face_work).max()):.3e}; "
                  f"|0.5 d_rho dphi| max {float(cp.abs(tw).max()):.3e} at {where(cp.abs(tw))}", flush=True)
        orig_corr(c)
        if state["on"]:
            now("after correction")

    gas.drift, gas._gravity_kick, gas._add_gravity_energy_correction, gas._hydro_step = drift, kick, corr, hs

calls = {"n": 0, "prev": None}
orig = gas._compute_cfl_timestep


def where(a):
    i = np.unravel_index(int(cp.argmax(a)), a.shape)
    return i


def logged():
    dt = orig()
    v2 = sum(vv * vv for vv in gas.vel)
    P = gas._compute_pressure(gas.rho, gas.E)
    cs = cp.sqrt(gas.gamma * P / cp.maximum(gas.rho, gas.rho_floor))
    iv, ic = where(v2), where(cs)
    sig = gas.cfl * gas.dx / dt
    calls["n"] += 1
    n, prev = calls["n"], calls["prev"]
    if n <= 3 or n % 20 == 0 or (prev and sig > 3 * prev):
        print(f"\ncall {n:5d}: signal {sig:10.3e} | vmax {float(cp.sqrt(v2.max())):10.3e} at {iv} rho={float(gas.rho[iv]):.2e} "
              f"| cs_max {float(cs.max()):10.3e} at {ic} rho={float(gas.rho[ic]):.2e} "
              f"| cells at floor {int((gas.rho <= gas.rho_floor * 1.0000001).sum())} "
              f"| mass {float(gas.rho.sum()) * gas.cell_volume:.4e} rho_max {float(gas.rho.max()):.2e}", flush=True)
    if prev and sig > 100 * prev:
        v = cp.sqrt(v2)
        floor = gas.rho <= gas.rho_floor * 1.0000001
        print("\n  EXPLODING. vmax cell is at rho floor:", bool(floor[iv]),
              "| max |v| over floor cells:", float(v[floor].max()) if bool(floor.any()) else None,
              "| max |v| over non-floor cells:", float(v[~floor].max()),
              "| E-K min:", float((gas.E - 0.5 * gas.rho * v2).min()))
        sl = tuple(slice(max(0, c - 2), c + 3) for c in iv)
        print("  rho around vmax cell:\n", cp.asnumpy(gas.rho[sl][:, :, 2 if gas.rho[sl].shape[2] > 2 else 0]))
        raise SystemExit("stopped at first blow-up")
    calls["prev"] = sig
    return dt


gas._compute_cfl_timestep = logged
sim.evolve(save_every=100000, diagnostics_every=100000)
print("finished without blow-up; calls:", calls["n"])
