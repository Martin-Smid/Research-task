
from scipy.constants import gravitational_constant

from resources.Functions.Schrodinger_eq_functions import *
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_Packet_Class import Packet
from astropy import units, constants


#-----------------------------------------------------------------------------------------------------------------------


class Wave_function():  # Streamlined and unified evolution logic
    def __init__(self,simulation,packet_type="gaussian", momenta=[0], means=[0], st_deviations=[0.1],
                 potential=None, gravity_potential=None, mass=1, omega=1, **kwargs):
        """
        Initialize a wave function with optional gravitational potential.

        Parameters:
            potential (callable, optional): A user-defined potential function.
            gravity_potential (bool or None, optional): Whether to include dynamic gravitational potential. Defaults to None (no gravity).
            **kwargs: Additional keyword arguments.
        """
        self.simulation = simulation
        self.dim = simulation.dim
        self.boundaries = simulation.boundaries
        self.N = simulation.N
        self.means = means
        self.total_time = simulation.total_time
        self.h = simulation.h
        self.num_steps = int(self.total_time / self.h)
        self.dx = simulation.dx
        self.grids = simulation.grids
        self.momenta = momenta
        self.mass = self.simulation.mass_s
        self.h_bar_tilde = self.simulation.h_bar_tilde
        self.omega = omega
        self.packet_type = packet_type
        self.packet_creator = Packet(
            packet_type=self.packet_type,
            momenta=self.momenta,
            means=self.means,
            st_deviations=st_deviations,
            grids=self.grids,
            dx=self.dx,
            h_bar_tilde=self.h_bar_tilde,
            mass=self.mass,
            omega=self.omega,
            dim=self.dim, )
        self.psi = self.packet_creator.create_psi_0()

        self.potential = potential
        self.gravity_potential = gravity_potential


    def calclulate_soliton_mass(self):
        self.soliton_mass = self.packet_creator.compute_original_soliton_mass(self.packet_type)








