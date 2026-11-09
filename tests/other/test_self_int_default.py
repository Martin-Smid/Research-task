"""Check: self-interaction is off unless the user asks for it (the potential is experimental, known issue 3)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resources.Classes.Simulation_Class import Simulation_Class


def main():
    kwargs = dict(dim=3, boundaries=[(-5, 5)] * 3, N=16, total_time=0.01, h=1e-4, use_units=True, use_sponge=False)
    assert Simulation_Class(**kwargs).use_self_int is False, "self_int must be off by default"
    assert Simulation_Class(self_int=True, **kwargs).use_self_int is True, "self_int=True must still switch it on"
    print("PASS: self_int is off by default and can be switched on")


if __name__ == "__main__":
    main()
