"""Test: max_values.csv is written while the simulation runs (together with the energy records), not only at the end.

A short run with save_max_vals = True and diagnostics every DIAGNOSTICS_EVERY steps; in the middle of the run (after step MID_STEP) the file is copied.
Checks
    1) the file exists in the middle of the run and already holds the records up to the last diagnostics step (before this change it did not exist),
    2) the copy taken in the middle is a prefix of the final file (same format, same numbers; the final file is rewritten by save_max_values),
    3) the final file has one row for every step 0 ... NUM_STEPS and can be read with the reader of plot_max_values_on_N,
    4) the start-up report prints the masses and no longer the old hard-coded 'Deviation [%]' / 'mass diff' lines."""

from contextlib import redirect_stdout
from pathlib import Path
import io
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class


# ---------------------------------------------------------------------------------------------------------
# TEST PARAMETERS
# ---------------------------------------------------------------------------------------------------------
N_GRID = 32
H = 1.0e-3                          # [Gyr], far below the stability limit of this coarse grid
NUM_STEPS = 100
DIAGNOSTICS_EVERY = 20
MID_STEP = 65                       # the file is copied after this step: records for steps 0 ... 60 must be on disk


def main():
    sim = Simulation_Class(

        dim=3,
        boundaries=[(-9.6, 9.6)] * 3,
        N=N_GRID,
        total_time=NUM_STEPS * H,
        h=H,
        order_of_evolution=2,
        use_gravity=True,
        use_units=True,
        use_sponge=False,
        self_int=False,
        save_max_vals=True,
        m_s=2.5e-22,

    )
    sim.add_wave_vector(Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat",
        means=[0, 0, 0],
        st_deviations=[0.5, 0.5, 0.5],
        simulation=sim,
        mass=1,
        omega=1,
        momenta=[0, 0.0, 0],
        spin=0,
        desired_soliton_mass=5.0e7,

    ))
    log = io.StringIO()
    with redirect_stdout(log):
        sim.initialize_simulation()
        scribe = sim.evolution.scribe
        copies = {}
        original_step = sim.evolution._perform_evolution_step

        def watching_step(wave_functions, total_density, step, save_step):
            result = original_step(wave_functions, total_density, step, save_step)
            if step + 1 == MID_STEP:
                path = Path(scribe.max_vals_filename)
                copies["exists"] = path.exists()
                copies["text"] = path.read_text() if path.exists() else ""
            return result

        sim.evolution._perform_evolution_step = watching_step
        sim.evolution.evolve(sim.wave_functions, save_every=10 ** 9, diagnostics_every=DIAGNOSTICS_EVERY)
    output = log.getvalue()
    final_path = Path(scribe.max_vals_filename)
    final_text = final_path.read_text()
    data = pd.read_csv(final_path, comment="#", header=[0, 1], index_col=0)         # the reader of plot_max_values_on_N
    shutil.rmtree(scribe.snapshot_directory, ignore_errors=True)                      # the test leaves no output behind

    assert copies["exists"], "max_values.csv does not exist in the middle of the run"
    mid_rows = [line for line in copies["text"].splitlines()[2:] if line]
    last_mid_step = int(mid_rows[-1].split(",")[0])
    print(f"in the middle of the run (after step {MID_STEP}) the file holds {len(mid_rows)} rows, the last one is step {last_mid_step}")
    assert last_mid_step >= 60 - 1, "the records of the first diagnostics steps are not on disk yet"
    assert final_text.startswith(copies["text"]), "the file written during the run is not a prefix of the final file"
    assert list(data.index) == list(range(0, NUM_STEPS + 1)), "the final file does not have one row for every step"
    assert "Deviation [%]" not in output and "mass diff" not in output, "the old temporary mass check is still printed"
    assert "Mass of the wave functions" in output and "Requested soliton mass" in output
    print("PASS: max_values.csv is written during the run, the final file is the same, the start-up report is clean")


if __name__ == "__main__":
    main()
