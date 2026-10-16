"""Run the small regression checks without pytest."""

from test_diagnostics_gas_internal_energy_once import main as test_diagnostics_gas_internal_energy_once
from test_gas_gravity_energy import main as test_gas_gravity_energy
from test_gas_gravity_work_conservation import main as test_gas_gravity_work_conservation
from test_gas_internal_energy_positive import main as test_gas_internal_energy_positive
from test_gas_jeans_instability import main as test_gas_jeans_instability
from test_gas_multidim_timestep import main as test_gas_multidim_timestep
from test_gas_sod_shock_tube import main as test_gas_sod_shock_tube
from test_sink_gravity_single_count import main as test_sink_gravity_single_count


def main():
    checks = [
        ("gas internal energy counted once", test_diagnostics_gas_internal_energy_once),
        ("gas gravity energy", test_gas_gravity_energy),
        ("gas gravity work balances dW", test_gas_gravity_work_conservation),
        ("gas internal energy stays positive", test_gas_internal_energy_positive),
        ("multidimensional gas timestep", test_gas_multidim_timestep),
        ("Sod shock tube vs exact solution", test_gas_sod_shock_tube),
        ("Jeans instability vs linear theory", test_gas_jeans_instability),
        ("single-count sink gravity", test_sink_gravity_single_count),
    ]

    for name, check in checks:
        print(f"\n--- {name} ---")
        check()

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
