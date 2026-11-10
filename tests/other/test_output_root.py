"""Test: Simulation_Class(output_root=...) decides the folder that holds the simulation_<timestamp> output folders.

Checks
    1) the default is "resources/data" (the old behaviour) and the scribe takes the value from the simulation,
    2) with output_root=<tmp> a run writes its simulation_* folder under <tmp> (and nowhere else),
    3) the checkpoint stores output_root: from_checkpoint() gives the same value back,
    4) resume() continues in the original snapshot directory (it never creates a new folder),
    5) start_new_segment() on a loaded checkpoint keeps output_root: the new segment is a new simulation_* folder under <tmp>."""

from contextlib import redirect_stdout
from pathlib import Path
import io
import shutil
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class

N_GRID = 16
H = 1.0e-3
NUM_STEPS = 10


def build(output_root=None, total_time=NUM_STEPS * H):
    kwargs = {} if output_root is None else {"output_root": output_root}
    sim = Simulation_Class(

        dim=3, boundaries=[(-9.6, 9.6)] * 3, N=N_GRID, total_time=total_time, h=H, order_of_evolution=2,
        use_gravity=True, use_units=True, use_sponge=False, self_int=False, m_s=2.5e-22, **kwargs,

    )
    sim.add_wave_vector(Wave_vector_class(

        packet_type="resources/solitons/GroundState(1).dat", means=[0, 0, 0], st_deviations=[0.5, 0.5, 0.5],
        simulation=sim, mass=1, omega=1, momenta=[0, 0.0, 0], spin=0, desired_soliton_mass=5.0e7,

    ))
    return sim


def folders(root):
    return sorted(p.name for p in Path(root).glob("simulation_*"))


def main():
    default_sim = build()
    assert default_sim.output_root == "resources/data", "the default output_root must stay resources/data"
    log = io.StringIO()
    with redirect_stdout(log):
        default_sim.initialize_simulation()
    assert default_sim.evolution.scribe.output_root == "resources/data", "the scribe must take output_root from the simulation"

    tmp = tempfile.mkdtemp(prefix="output_root_test_")
    try:
        root = str(Path(tmp) / "custom_root")
        with redirect_stdout(log):
            sim = build(root)
            sim.initialize_simulation()
            sim.evolution.evolve(sim.wave_functions, save_every=5, diagnostics_every=5)
        assert len(folders(root)) == 1, f"expected one simulation_* folder under {root}, found {folders(root)}"
        first = Path(root) / folders(root)[0]
        assert Path(sim.evolution.scribe.snapshot_directory).resolve() == first.resolve()

        with redirect_stdout(log):
            loaded = Simulation_Class.from_checkpoint(first)
        assert loaded.output_root == root, f"from_checkpoint lost output_root: {loaded.output_root!r}"

        with redirect_stdout(log):
            earlier = sorted((first / "checkpoints").glob("step_*"))[0]              # intermediate checkpoint (step 5)
            resumed = Simulation_Class.from_checkpoint(earlier)
            resumed.resume()
        assert folders(root) == [first.name], "resume() must continue in the original folder, not create a new one"

        with redirect_stdout(log):
            segment = Simulation_Class.from_checkpoint(first)
            segment.start_new_segment(total_time=5 * H)
            assert segment.output_root == root, "start_new_segment lost output_root"
            segment.initialize_simulation()
            segment.evolution.evolve(segment.wave_functions, save_every=5, diagnostics_every=5)
        assert len(folders(root)) == 2, f"the new segment must create a second folder under {root}, found {folders(root)}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)                                       # the test leaves no output behind
    print("PASS: output_root is a Simulation_Class parameter, stored in checkpoints, kept by from_checkpoint, resume and start_new_segment")


if __name__ == "__main__":
    main()
