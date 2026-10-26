"""Run the small regression checks without pytest."""

from test_diagnostics_gas_internal_energy_once import main as test_diagnostics_gas_internal_energy_once
from test_gas_adaptive_substeps import main as test_gas_adaptive_substeps
from test_gas_free_fall_collapse import main as test_gas_free_fall_collapse
from test_gas_gravity_energy import main as test_gas_gravity_energy
from test_gas_gravity_work_conservation import main as test_gas_gravity_work_conservation
from test_gas_internal_energy_positive import main as test_gas_internal_energy_positive
from test_gas_jeans_instability import main as test_gas_jeans_instability
from test_gas_multidim_timestep import main as test_gas_multidim_timestep
from test_gas_vacuum_cell_velocity import main as test_gas_vacuum_cell_velocity
from test_gas_vacuum_hole_between_flows import main as test_gas_vacuum_hole_between_flows
from test_gas_sod_shock_tube import main as test_gas_sod_shock_tube
from test_particle_breathing_soliton import main as test_particle_breathing_soliton
from test_particle_kepler_orbit import main as test_particle_kepler_orbit
from test_sink_gas_accretion_energy import main as test_sink_gas_accretion_energy
from test_sink_force_vs_analytic import main as test_sink_force_vs_analytic
from test_sink_gravity_single_count import main as test_sink_gravity_single_count
from test_sink_potential_position import main as test_sink_potential_position


def main():
    checks = [
        ("gas internal energy counted once", test_diagnostics_gas_internal_energy_once),
        ("gas gravity energy", test_gas_gravity_energy),
        ("gas gravity work balances dW", test_gas_gravity_work_conservation),
        ("gas internal energy stays positive", test_gas_internal_energy_positive),
        ("multidimensional gas timestep", test_gas_multidim_timestep),
        ("emptied cells keep no momentum", test_gas_vacuum_cell_velocity),
        ("vacuum hole between diverging flows", test_gas_vacuum_hole_between_flows),
        ("gas substeps follow the CFL estimate", test_gas_adaptive_substeps),
        ("Sod shock tube vs exact solution", test_gas_sod_shock_tube),
        ("cold sphere free-fall collapse", test_gas_free_fall_collapse),
        ("Jeans instability vs linear theory", test_gas_jeans_instability),
        ("single-count sink gravity", test_sink_gravity_single_count),
        ("sink potential centred on the sink", test_sink_potential_position),
        ("accretion removes energy with mass", test_sink_gas_accretion_energy),
        ("particle integrator on a circular orbit around a sink", test_particle_kepler_orbit),
        ("sink force on the grid vs analytic formula", test_sink_force_vs_analytic),
        ("particle in a breathing soliton converges in h", test_particle_breathing_soliton),
    ]

    for name, check in checks:
        print(f"\n--- {name} ---")
        check()

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
