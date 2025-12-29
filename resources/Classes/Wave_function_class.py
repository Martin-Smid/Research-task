
from scipy.constants import gravitational_constant
from scipy.interpolate import interp1d
from resources.Functions.Schrodinger_eq_functions import *
from resources.Classes.Simulation_Class import Simulation_Class
from resources.Classes.Wave_Packet_Class import Packet
from astropy import units, constants
import os

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
        self.st_deviations = st_deviations
        self.packet_type = packet_type
        self.packet_creator = Packet(
            packet_type=self.packet_type,
            momenta=self.momenta,
            means=self.means,
            st_deviations=self.st_deviations,
            grids=self.grids,
            dx=self.dx,
            h_bar_tilde=self.h_bar_tilde,
            mass=self.mass,
            omega=self.omega,
            dim=self.dim, )


        self.psi = self.packet_creator.create_psi_0()
        self.desired_soliton_mass = desired_soliton_mass


        print(f"passed mass is {desired_soliton_mass}")
        if self.simulation.use_units and os.path.isfile(str(self.packet_type)):
            print(f"passed mass is {desired_soliton_mass}")

            try:
                self.soliton_mass = self.calculate_soliton_mass_from_spherical_data()
                print(f"calculated mass is {self.soliton_mass}")

                if self.soliton_mass > 0:
                    self.scaling_lambda = self.desired_soliton_mass / self.soliton_mass
                    print(self.scaling_lambda)
                    print("above is lambda")
                    self.psi = self._rescale_psi_to_new_scale_based_on_mass()
                #self._dealias_initial_psi(frac=2/3)
            except Exception as e:
                print(f"Warning: Could not rescale mass for packet {self.packet_type}. Using raw packet. Error: {e}")

            massss = self.calclulate_soliton_mass()  # Typo in original method name kept for consistency
            print(f"{massss} sol massss")




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

        return cp.abs(self.multiplicity*self.psi).astype(cp.float64) ** 2

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



    def softcopy_psi(self, new_psi, multiplicity=1):
        clone = Wave_function.__new__(Wave_function)  # hopefully this wont initialize new wf

        # Copy shared attributes without reinitializing
        for attr in ['simulation', 'dim', 'boundaries', 'multiplicity', 'N', 'means',
                     'total_time', 'h', 'num_steps', 'dx', 'grids', 'momenta',
                     'mass', 'h_bar_tilde', 'omega', 'packet_type', 'packet_creator',
                     'desired_soliton_mass', 'soliton_mass', 'scaling_lambda']:
            setattr(clone, attr, getattr(self, attr))

        # Now set ψ and multiplicity
        clone.psi = new_psi
        clone.multiplicity = multiplicity

        return clone

    def drift(self, kinetic_propagator):
        """
        Apply the drift (kinetic) step of split-step Fourier method to this wavefunction.
        ψ_k ← FFT(ψ)
        ψ_k *= kinetic propagator
        ψ ← IFFT(ψ_k)
        """
        psi_k = cp.fft.fftn(self.psi)
        psi_k *= kinetic_propagator
        self.psi = cp.fft.ifftn(psi_k)

    def _dealias_initial_psi(self, frac=2/3):
        """
        Low-pass filter psi in k-space to remove Nyquist/aliasing artifacts (e.g. axis cross at t=0).
        Keeps total mass (∫|psi|^2 dV) unchanged by renormalizing.
        Works with both numpy and cupy arrays.
        """
        psi = self.psi
        use_cupy = hasattr(cp, "ndarray") and isinstance(psi, cp.ndarray)

        xp = cp if use_cupy else np
        fftn = xp.fft.fftn
        ifftn = xp.fft.ifftn
        fftfreq = xp.fft.fftfreq

        # Save mass before filtering
        dV = float(np.prod(self.dx))
        mass0 = xp.sum(xp.abs(psi)**2) * dV

        # Build k-grid (use the same dx, N, periodic convention as your solver)
        k_components = [2 * np.pi * fftfreq(self.N, d=float(self.dx[i])) for i in range(self.dim)]
        k_mesh = xp.meshgrid(*k_components, indexing="ij")
        k2 = xp.zeros_like(k_mesh[0], dtype=xp.float64)
        for ki in k_mesh:
            k2 = k2 + ki.astype(xp.float64)**2
        k = xp.sqrt(k2)

        dx_min = float(min(self.dx))
        k_ny = np.pi / dx_min
        k_cut = frac * k_ny

        # Filter
        psi_k = fftn(psi)
        psi_k = psi_k * (k <= k_cut)
        psi_f = ifftn(psi_k)

        # Renormalize mass
        mass1 = xp.sum(xp.abs(psi_f)**2) * dV
        if float(mass1) > 0:
            psi_f = psi_f * xp.sqrt(mass0 / mass1)

        self.psi = psi_f.astype(psi.dtype, copy=False)
