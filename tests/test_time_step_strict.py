"""Check: sim.strict_time_step turns the time-step stability warning into an error.

By default (strict_time_step = False) a step above the limit only prints a warning, as before. With sim.strict_time_step = True
check_time_step_restriction raises ValueError for a step above the limit and still accepts a step below it."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resources.Classes.Simulation_Class import Simulation_Class


def make_sim(h):
    return Simulation_Class(dim=3, boundaries=[(-5, 5)] * 3, N=16, total_time=0.01, h=h, order_of_evolution=2,
                            use_gravity=True, use_units=True, use_sponge=False, self_int=False)


def main():
    too_large = 1.0e3
    ok_step = 1.0e-9

    sim = make_sim(too_large)
    sim.calculate_physical_units()
    assert sim.strict_time_step is False, "strict_time_step must be off by default"
    assert sim.check_time_step_restriction() is True, "default behaviour must stay a warning"

    sim.strict_time_step = True
    try:
        sim.check_time_step_restriction()
    except ValueError as error:
        print("raised as expected:", str(error).splitlines()[0])
    else:
        raise AssertionError("strict_time_step = True must raise for a step above the limit")

    sim_ok = make_sim(ok_step)
    sim_ok.calculate_physical_units()
    sim_ok.strict_time_step = True
    assert sim_ok.check_time_step_restriction() is True, "a step below the limit must be accepted in strict mode"
    print("PASS: strict_time_step raises above the limit, warns by default, accepts a small step")


if __name__ == "__main__":
    main()
