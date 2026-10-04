import os
import csv
import datetime
import json
import subprocess
import uuid
from pathlib import Path

import numpy as np
import cupy as cp
import pandas as pd

from resources.Functions.checkpointing import _json_value, simulation_config


class Scribe:
    """
    Handles all data writing operations for the simulation.
    Separates I/O concerns from evolution logic.
    """

    def __init__(self, simulation):
        """
        Initialize the Scribe.

        Parameters:
            simulation: Simulation_Class instance (for accessing grid info, etc.)
        """
        self.simulation = simulation
        self.snapshot_directory = None
        self.max_vals_filename = None
        self.max_locations_path = None
        self.trajectory_path = None
        self.total_density = None

        # Data storage
        self.wave_values = []
        self.accessible_times = []
        self.energy_log = []
        self.max_location_log = []
        self.trajectory_log = []
        self.max_wave_vals_during_evolution = {}

    def setup_directories(self, num_wave_functions):
        """
        Create directory structure for saving data AND initialize energy log.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_dir = f"resources/data/simulation_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        self.snapshot_directory = save_dir

        self.max_vals_filename = os.path.join(self.snapshot_directory, "max_values.csv")

        self.energy_path = os.path.join(self.snapshot_directory, "energy.csv")

        header = "time,K_total,W,U_iso,E_total,K_flow,U_quantum,K_baryons,W_self,W_static,W_over_E,E_diss,E_tot_cons\n"

        with open(self.energy_path, "w") as f:
            f.write(header)

        # Initialize storage for each wave function
        self.wave_values = [[] for _ in range(num_wave_functions)]
        self.accessible_times.append(0)

        self.trajectory_path = os.path.join(self.snapshot_directory, "particle_trajectory.csv")
        with open(self.trajectory_path, "w") as f:
            f.write("time,x,y,z\n")

        # Initialize max location logger
        self.max_locations_path = os.path.join(self.snapshot_directory, "max_locations.txt")
        with open(self.max_locations_path, "w") as f:
            f.write("# time, ix, iy, iz, x, y, z\n")

        self.rotation_curve_path = os.path.join(self.snapshot_directory, "rotational_velocity.dat")
        with open(self.rotation_curve_path, "w") as f:
            f.write("time,component,R,vphi_mean,vphi_std,weight\n")


        return save_dir

    @staticmethod
    def _callable_name(value):
        if value is None:
            return None
        module = getattr(value, "__module__", None)
        name = getattr(value, "__qualname__", getattr(value, "__name__", None))
        return f"{module}.{name}" if module and name else str(value)

    @staticmethod
    def _portable_path(value):
        if value is None:
            return None
        project_root = Path(__file__).resolve().parents[2]
        try:
            return Path(value).resolve().relative_to(project_root).as_posix()
        except (OSError, ValueError):
            return str(value)

    @staticmethod
    def _git_state():
        project_root = Path(__file__).resolve().parents[2]
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip())
            return {"commit": commit, "dirty": dirty}
        except (OSError, subprocess.SubprocessError):
            return {"commit": None, "dirty": None}

    def save_run_config(self, status, current_step):
        """Create or update reproducibility metadata for this evolution call."""
        if self.snapshot_directory is None:
            raise ValueError("Cannot save run_config.json before creating an output directory")

        simulation = self.simulation
        if getattr(simulation, "_run_id", None) is None:
            simulation._run_id = uuid.uuid4().hex

        path = Path(self.snapshot_directory) / "run_config.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        lifecycle = data.get("lifecycle", [])
        event = next(
            (item for item in lifecycle if item.get("run_id") == simulation._run_id),
            None,
        )
        if event is None:
            event = {
                "run_id": simulation._run_id,
                "started_at": now,
            }
            lifecycle.append(event)

        event.update({
            "mode": getattr(simulation, "_run_mode", "new"),
            "status": status,
            "source_checkpoint": self._portable_path(
                getattr(simulation, "_source_checkpoint", None)
            ),
            "source_step": getattr(simulation, "_source_step", None),
            "source_time": getattr(simulation, "_source_time", None),
            "parameter_changes": getattr(simulation, "_run_parameter_changes", {}),
            "schedule": getattr(simulation, "_run_schedule", {}),
            "current_step": int(current_step),
            "current_time": float(current_step * simulation.h),
        })
        if status == "completed":
            event["completed_at"] = now

        config = simulation_config(simulation)
        config["static_potential"] = self._callable_name(simulation.static_potential)
        data = {
            "schema_version": 1,
            "simulation": config,
            "random_seeds": {
                "wave_vectors": getattr(simulation, "wave_vector_seeds", []),
            },
            "git": self._git_state(),
            "components": [
                {
                    "type": component.__class__.__name__,
                    "name": getattr(component, "name", None),
                    "count": getattr(component, "N", None),
                }
                for component in simulation.baryonic_matter
            ],
            "wave_functions": [
                {
                    "packet_type": str(getattr(wave, "packet_type", None)),
                    "means": getattr(wave, "means", None),
                    "momenta": getattr(wave, "momenta", None),
                    "st_deviations": getattr(wave, "st_deviations", None),
                    "omega": getattr(wave, "omega", None),
                    "multiplicity": getattr(wave, "multiplicity", None),
                    "desired_soliton_mass": getattr(
                        wave,
                        "desired_soliton_mass",
                        None,
                    ),
                }
                for wave in simulation.wave_functions
            ],
            "lifecycle": lifecycle,
        }

        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(_json_value(data), indent=2),
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def save_initial_states(self, wave_functions, total_density=None):
        """
        Save initial wave function states.

        Parameters:
            wave_functions: List of wave function objects
        """
        for wf_idx, wf in enumerate(wave_functions):
            initial_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_0.npy"
            np.save(initial_path, cp.asnumpy(wf.psi))
            self.wave_values[wf_idx].append(initial_path)

        # TODO: check this - keep the initial total-density snapshot aligned with wf snapshots.
        if total_density is not None:
            total_path = f"{self.snapshot_directory}/total_density_snapshot_at_time_0.npy"
            np.save(total_path, cp.asnumpy(total_density))

    def save_snapshots(self, wave_functions, current_time, total_density=None, baryon_density=None, gas_density=None):
        """
        Save wave function snapshots, baryon density, and total density at current step.

        Parameters:
            wave_functions: List of wave function objects
            step: Current simulation step
            h: Time step size
            total_density: (Optional) Total density grid (Waves + Baryons + Sinks)
            baryon_density: (Optional) Total baryon density grid (Gas + Sinks)
        """
        saved_wave_files = []
        # Save Wave Functions
        for wf_idx, wf in enumerate(wave_functions):
            snapshot_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_{current_time:.6f}.npy"
            np.save(snapshot_path, cp.asnumpy(wf.psi))
            saved_wave_files.append(os.path.basename(snapshot_path))
            self.wave_values[wf_idx].append(snapshot_path)

        # Save Baryon Density 
        if baryon_density is not None:
            try:
                baryon_path = f"{self.snapshot_directory}/baryons_snapshot_at_time_{current_time:.6f}.npy"
                np.save(baryon_path, cp.asnumpy(baryon_density))
            except Exception as e:
                print(f"[Scribe] Warning: could not save baryon snapshot at time {current_time:.6f}: {e}")

        if gas_density is not None:
            try:
                gas_path = f"{self.snapshot_directory}/gas_snapshot_at_time_{current_time:.6f}.npy"
                np.save(gas_path, cp.asnumpy(gas_density))
            except Exception as e:
                print(f"[Scribe] Warning: could not save gas snapshot at time {current_time:.6f}: {e}")

        # Save Total Density 
        if total_density is not None:
            try:
                total_path = f"{self.snapshot_directory}/total_density_snapshot_at_time_{current_time:.6f}.npy"
                np.save(total_path, cp.asnumpy(total_density))
            except Exception as e:
                print(f"[Scribe] Warning: could not save total density snapshot at time {current_time:.6f}: {e}")

        self.accessible_times.append(current_time)
        return saved_wave_files

    def save_final_state(self, wave_functions, num_steps, save_every, h, total_time,
                         total_density=None):
        """
        Save final state if it wasn't already saved.

        Parameters:
            wave_functions: List of wave function objects
            num_steps: Total number of steps
            save_every: Save frequency
            h: Time step size
            total_time: Total simulation time
        """
        saved_wave_files = []
        final_time = total_time
        # TODO: check this - snapshots are scheduled by completed steps, not zero-based indices.
        if num_steps % save_every != 0:
            for wf_idx, wf in enumerate(wave_functions):
                final_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_{final_time:.6f}.npy"
                np.save(final_path, cp.asnumpy(wf.psi))
                saved_wave_files.append(os.path.basename(final_path))
                self.wave_values[wf_idx].append(final_path)

            if total_density is not None:
                total_path = f"{self.snapshot_directory}/total_density_snapshot_at_time_{final_time:.6f}.npy"
                np.save(total_path, cp.asnumpy(total_density))

            if hasattr(self.simulation, "baryonic_matter") and self.simulation.baryonic_matter:
                try:
                    shape = (self.simulation.N,) * self.simulation.dim
                    rho_baryons = cp.zeros(shape, dtype=cp.float64)
            
                    for sys in self.simulation.baryonic_matter:
                        rho_baryons += sys.deposit_to_grid()
            
                    baryon_path = f"{self.snapshot_directory}/baryons_snapshot_at_time_{final_time:.6f}.npy"
                    np.save(baryon_path, cp.asnumpy(rho_baryons))
                except Exception as e:
                    print(f"[Scribe] Warning: could not save baryon snapshot at time {final_time:.6f}: {e}")
            self.accessible_times.append(final_time)
        else:
            # TODO: check this - restart metadata still needs the files saved by the regular schedule.
            for wf_idx in range(len(wave_functions)):
                final_path = f"{self.snapshot_directory}/wf_{wf_idx}_snapshot_at_time_{final_time:.6f}.npy"
                if os.path.exists(final_path):
                    saved_wave_files.append(os.path.basename(final_path))
        return saved_wave_files

    def save_metadata(self, num_steps, h, total_time, order, num_wave_functions):
        """
        Save simulation metadata.

        Parameters:
            num_steps: Total number of steps
            h: Time step size
            total_time: Total simulation time
            order: Method order
            num_wave_functions: Number of wave functions
        """
        with open(f"{self.snapshot_directory}/metadata.txt", "w") as f:
            f.write(f"Total steps: {num_steps}\n")
            f.write(f"Time step: {h}\n")
            f.write(f"Total time: {total_time}\n")
            f.write(f"Method order: {order}\n")
            f.write(f"Number of wave functions: {num_wave_functions}\n")
            f.write("Accessible times:\n")
            f.write(",".join([str(t) for t in self.accessible_times]))

    def record_max_location(self, ix, iy, iz, time_value):
        """
        Save the grid-index location of the current max density and its coordinates.

        Parameters:
            ix, iy, iz: Indices of the max-density cell
            time_value: Physical time corresponding to this step
        """
        # Get physical coordinates at that index
        gx, gy, gz = self.simulation.grids
        x = float(gx[ix, iy, iz])
        y = float(gy[ix, iy, iz])
        z = float(gz[ix, iy, iz])

        # Cache in memory and append to file
        self.max_location_log.append((float(time_value), int(ix), int(iy), int(iz), x, y, z))
        with open(self.max_locations_path, "a") as f:
            f.write(f"{time_value:.9e}, {ix:d}, {iy:d}, {iz:d}, {x:.9e}, {y:.9e}, {z:.9e}\n")

    def log_energy_detailed(self, time, K_total, W, U_iso, K_flow, U_quantum,
                            K_baryons, W_self, W_static, E_diss=0.0):
        """
        Log energy values at a given time.

        Parameters
        ----------
        time : float
            Current simulation time.
        K_total : float
            Total kinetic energy (waves + baryons).
        W : float
            Total potential energy (self-gravity + static).
        K_flow : float
            Wave kinetic "flow" part.
        U_quantum : float
            Wave "quantum pressure" term.
        K_baryons : float
            Kinetic energy of N-body baryons.
        W_self : float
            Self-gravitational energy from Poisson.
        W_static : float
            Energy in external static potential.
        """
        E_current = K_total + W + U_iso


        # Conserved Energy = Current Energy + Energy that was lost (Dissipated)
        E_conserved = E_current + E_diss

        if E_current != 0:
            W_over_E = W / abs(E_current)
        else:
            W_over_E = np.nan

        # Update internal log (optional, but good for consistency)
        self.energy_log.append({
            "time": float(time),
            "K_total": float(K_total),
            "W": float(W),
            "E_total": float(E_current),
            "K_flow": float(K_flow),
            "U_quantum": float(U_quantum),
            "K_baryons": float(K_baryons),
            "W_self": float(W_self) if W_self is not None else float("nan"),
            "W_static": float(W_static) if W_static is not None else float("nan"),
            "W/|E|": float(W_over_E),
            "E_diss": float(E_diss),
            "E_tot_cons": float(E_conserved)
        })

        line = (f"{float(time):.15e},{float(K_total):.15e},{float(W):.15e},{float(U_iso):.15e},{float(E_current):.15e},"
                f"{float(K_flow):.15e},{float(U_quantum):.15e},{float(K_baryons):.15e},"
                f"{float(W_self):.15e},{float(W_static):.15e},{float(W_over_E):.15e},"
                f"{float(E_diss):.15e},{float(E_conserved):.15e}\n")

        try:
            with open(self.energy_path, "a") as f:
                f.write(line)
        except Exception as e:
            print(f"Error writing energy log: {e}")

    def save_energy_log(self):
        """Save energy log to CSV file."""
        energy_path = os.path.join(self.snapshot_directory, "energy.txt")
        df = pd.DataFrame(self.energy_log)
        df.to_csv(energy_path, index=False, float_format="%.15e")
        print(f"Energy log saved to: {energy_path}")

    def save_radial_density_profile(self, bin_centers, rho_avg, current_time):
        """
        Save radial density profile data to CSV.

        Parameters:
            bin_centers: Array of radial bin centers
            rho_avg: Array of average densities per bin
            current_time: Current simulation time
        """
        save_dir = os.path.join(self.snapshot_directory, "density_profiles")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"density_data_at_time_{current_time:.6f}.csv")
        np.savetxt(save_path, np.column_stack((bin_centers, rho_avg)),
                   delimiter=",", header="radius,density", comments="")

    def track_max_value(self, step, max_value):
        """
        Track maximum value during evolution.

        Parameters:
            step: Current step number
            max_value: Maximum value to track
        """
        self.max_wave_vals_during_evolution[step] = float(max_value)

    def save_max_values(self, resolution, spin):
        """
        Save maximum values during evolution to CSV.

        Parameters:
            resolution: Grid resolution (N)
            spin: Spin value of the wavefunction
        """

        header = [
            "# This file contains the maximum values of wave functions at specific steps along the time evolution.",
            "# The first row contains the corresponding N - the resolution of the saved simulation.",
            "# The second row contains the spin of the wavefunction in the corresponding simulation.",
            "# The first column is the time step, and subsequent columns are max values for a given simulation.",
        ]

        # Prepare multi-column format
        col_tuples = [(f"N{resolution}", f"s={spin}")]
        columns = pd.MultiIndex.from_tuples(col_tuples)

        new_data = pd.DataFrame.from_dict(
            self.max_wave_vals_during_evolution,
            orient="index",
            columns=columns
        )

        with open(self.max_vals_filename, "w") as file:
            file.write("\n".join(header) + "\n")
        new_data.to_csv(self.max_vals_filename)

        print(f"{self.max_vals_filename} saved")

    def get_wave_function_at_time(self, time, wf_index=None, num_wave_functions=None):
        """
        Retrieve wave function(s) at a given time.

        Parameters:
            time: The time at which to retrieve the wave function values
            wf_index: Index of specific wave function. If None, returns sum of all.
            num_wave_functions: Total number of wave functions (needed if wf_index is None)

        Returns:
            cp.ndarray: The wave function (or sum) at the given time
        """
        # Find closest time
        closest_time_index = min(range(len(self.accessible_times)),
                                 key=lambda i: abs(self.accessible_times[i] - time))

        if wf_index is not None:
            file_path = self.wave_values[wf_index][closest_time_index]
            return cp.array(np.load(file_path))
        else:
            # Return sum of all wave functions
            if num_wave_functions is None:
                num_wave_functions = len(self.wave_values)

            summed_wave_function = None
            for wf_idx in range(num_wave_functions):
                file_path = self.wave_values[wf_idx][closest_time_index]
                wave_function = cp.array(np.load(file_path))

                if summed_wave_function is None:
                    summed_wave_function = wave_function.copy()
                else:
                    summed_wave_function += wave_function

            return summed_wave_function

    def log_single_particle(self, time, position):
        """
        Buffer the single particle position.
        position should be a CPU array/list of [x, y, z].
        """
        # Appending to a list is faster than writing to disk every step
        self.trajectory_log.append({
            "time": float(time),
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2])
        })

    def flush_trajectory_buffer(self):
        """
        Write currently buffered trajectory points to CSV and clear buffer.
        Call this periodically or at the end of simulation.
        """
        if not self.trajectory_log:
            return

        df = pd.DataFrame(self.trajectory_log)
        # Append to csv, avoiding rewriting header
        df.to_csv(self.trajectory_path, mode='a', header=False, index=False)
        self.trajectory_log = []  # Clear memory

    def save_rotation_curve(self, time, component_name, R_centers, vphi_mean, vphi_std, weights, frame=None):
        """
        Append one component's rotation curve to rotational_velocity.dat
    
        Parameters
        ----------
        time : float
            Simulation time.
        component_name : str
            Name of the component, e.g. "gas", "bulge", "stellar_disk".
        R_centers, vphi_mean, vphi_std, weights : array-like
            Rotation-curve data returned by compute_rotation_curve().
        """
        if frame is not None:
            path = Path(self.snapshot_directory) / "rotation_frames.csv"
            new_file = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if new_file:
                    writer.writerow(["time", "component", "center_x", "center_y", "center_z",
                                     "bulk_vx", "bulk_vy", "bulk_vz", "axis_x", "axis_y", "axis_z",
                                     "rmax", "zmax", "mass_fraction", "enclosed_mass_fraction", "axis_coherence",
                                     "status", "length_unit", "velocity_unit", "mass_unit", "time_unit"])
                length_unit = self.simulation.dUnits if self.simulation.use_units else "simulation"
                velocity_unit = f"{length_unit}/{self.simulation.tUnits}" if self.simulation.use_units else "simulation"
                mass_unit = self.simulation.mUnits if self.simulation.use_units else "simulation"
                writer.writerow([time, component_name, *frame["center"], *frame["bulk_velocity"],
                                 *frame["axis"], frame["rmax"], frame["zmax"], frame["mass_fraction"],
                                 frame["enclosed_mass_fraction"], frame["axis_coherence"], frame["status"],
                                 length_unit, velocity_unit, mass_unit,
                                 self.simulation.tUnits if self.simulation.use_units else "simulation"])
        if R_centers is None or len(R_centers) == 0:
            return
    
        try:
            with open(self.rotation_curve_path, "a") as f:
                for R, vm, vs, w in zip(R_centers, vphi_mean, vphi_std, weights):
                    if np.isfinite(R):
                        f.write(
                            f"{float(time):.15e},"
                            f"{component_name},"
                            f"{float(R):.15e},"
                            f"{float(vm):.15e},"
                            f"{float(vs):.15e},"
                            f"{float(w):.15e}\n"
                        )
        except Exception as e:
            print(f"[Scribe] Error writing rotation curve for {component_name} at t={time}: {e}")
