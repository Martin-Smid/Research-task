"""Initialize and run a small ULDM simulation."""

from examples._runtime import configure_cuda_on_windows

configure_cuda_on_windows()

from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_vector_class import Wave_vector_class


sim = Simulation_Class(
    dim=3,
    boundaries=[(-10, 10)] * 3,
    N=32,
    total_time=0.01,
    h=0.001,
    order_of_evolution=2,
    use_gravity=True,
    static_potential=None,
    save_max_vals=True,
    m_s=2.5e-22,
    use_sponge=False,
    self_int=False,
)

wave_vector = Wave_vector_class(
    simulation=sim,
    packet_type="resources/solitons/GroundState(1).dat",
    means=[0, 0, 0],
    st_deviations=[0.5, 0.5, 0.5],
    momenta=[0, 0, 0],
    mass=1,
    omega=1,
    spin=0,
    desired_soliton_mass=53_090_064.0,
    random_seed=42,
)

sim.add_wave_vector(wave_vector)
sim.evolve(save_every=5)

print(f"Output saved to: {sim.snapshot_directory}")
