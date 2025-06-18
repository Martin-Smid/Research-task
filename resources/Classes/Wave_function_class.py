
from scipy.constants import gravitational_constant
from scipy.interpolate import interp1d
from resources.Functions.Schrodinger_eq_functions import *
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_Packet_Class import Packet
from astropy import units, constants


#-----------------------------------------------------------------------------------------------------------------------


class Wave_function():  # Streamlined and unified evolution logic
    def __init__(self,simulation,packet_type="gaussian", momenta=[0], means=[0], st_deviations=[0.1],
                 potential=None, gravity_potential=None, mass=1, omega=1,desired_soliton_mass=1e6, **kwargs):
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
        self.multiplicity = 1
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
        self.desired_soliton_mass = desired_soliton_mass
        print(f"passed mass is {desired_soliton_mass}")
        #self.rescale_psi_to_phys_units()
        self.soliton_mass = self.calculate_soliton_mass_from_spherical_data()
        print(f"calclulated mass is {self.soliton_mass}")
        self.scaling_lambda = self.desired_soliton_mass / self.soliton_mass
        print(self.scaling_lambda)
        print("above is lambda")
        self.psi = self._rescale_psi_to_new_scale_based_on_mass()

        massss = self.calclulate_soliton_mass()
        print(f"{massss} sol massss")

        self.potential = potential
        self.gravity_potential = gravity_potential


    def calclulate_soliton_mass(self):
        density = np.abs(self.psi) ** 2
        volume_element = np.prod(self.dx)
        mass = np.sum(density)*volume_element
        return mass

    def calculate_soliton_mass_from_spherical_data(self):
        data = self.packet_creator.read_ground_state_data(self.packet_type)
        r_values = data[:, 0]
        phi_values = data[:, 1]
        phi_values *= self.simulation.h_bar_tilde / np.sqrt(self.simulation.G)

        density = np.abs(phi_values) ** 2  # rho(r) = |psi(r)|^2

        # Use trapezoidal integration for M = ∫ rho(r) * 4πr² dr
        integrand = 4 * np.pi * r_values ** 2 * density
        mass = np.trapz(integrand, r_values)

        return mass

    def calculate_density(self):

        return cp.abs(self.multiplicity*self.psi).astype(cp.float32) ** 2

    def rescale_psi_to_phys_units(self):
        # Convert wave function: ψ_sol = ψ̂_sol * (ħ/√G)
        self.conversion_factor = self.simulation.h_bar_tilde / np.sqrt(self.simulation.G)
        self.psi *= self.conversion_factor


    def _rescale_psi_to_new_scale_based_on_mass(self):
        data = self.packet_creator.read_ground_state_data(self.packet_type)

        r_values = data[:, 0]
        r_values /= self.scaling_lambda
        print(max(r_values))
        phi_values = data[:, 1]
        phi_values *=  self.simulation.h_bar_tilde / np.sqrt(self.simulation.G)
        phi_values *= self.scaling_lambda**2

        interp_phi = interp1d(r_values, phi_values, kind='linear',
                              bounds_error=False, fill_value=0.0)

        # Compute r at each grid point
        r_distance = np.zeros_like(self.grids[0])
        for dim in range(self.dim):
            r_distance += (self.grids[dim] - self.means[dim]) ** 2
        r_distance = np.sqrt(r_distance)

        # Interpolate φ at each grid point
        psi = interp_phi(r_distance)

        return psi.astype(np.complex64)




