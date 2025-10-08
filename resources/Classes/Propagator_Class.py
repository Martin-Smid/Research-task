import cupy as cp
import numpy as np
#np.random.seed(1)
from astropy import units, constants


class Propagator_Class:
    """
    Class to handle the creation and management of different types of propagators
    used in the Schrödinger equation evolution.
    """

    def __init__(self, simulation):
        """
        Initialize the propagator with references to the simulation parameters.

        Parameters:
            simulation: Simulation_Class instance containing necessary parameters
        """
        self.simulation = simulation
        self.dim = simulation.dim
        self.h = simulation.h  # Time step
        self.dx = simulation.dx
        self.N = simulation.N
        self.grids = simulation.grids
        self.k_space = simulation.k_space
        self.G = simulation.G
        self.h_bar = simulation.h_bar
        self.h_bar_tilde = self.simulation.h_bar_tilde
        # Placeholders for propagators
        self.kinetic_propagator = None
        self.static_potential_propagator = None
        self.gravity_propagator = None
        self.gravity_potential=None
        if not self.simulation.use_units:
            self.h_bar_tilde = 1
            self.h_bar = 1
            self.G = 1

    def compute_kinetic_propagator(self,time_factor=1):
        """
        Compute the kinetic propagator based on Fourier space components.
        Uses the split-step Fourier method formulation.

        Returns:
            cp.ndarray: The kinetic propagator in k-space
        """
        # Calculate k_squared_sum for the Laplacian operator
        k_squared_sum = cp.zeros_like(self.k_space[0], dtype=cp.float32)
        for k in self.k_space:
            k_squared_sum += k ** 2


        self.kinetic_propagator = cp.exp(((-1j * ((self.h*time_factor) / 2) * k_squared_sum )*(self.h_bar_tilde)), dtype=cp.complex64)
        return self.kinetic_propagator


    def compute_static_potential_propagator(self, potential_function,time_factor=1):
        """
        Compute the static potential propagator.

        Parameters:
            potential_function: Function that takes the simulation as input and returns potential values

        Returns:
            cp.ndarray: The potential propagator in real space
        """
        if potential_function is not None:
            potential_values = potential_function(self.simulation)
            self.static_potential_propagator = cp.exp((-1j * self.h*time_factor * potential_values)/(self.h_bar_tilde) , dtype=cp.complex64)

        else:
            # If no potential is provided, use unit propagator (no effect)
            self.static_potential_propagator = cp.ones(self.simulation.grids[0].shape, dtype=cp.complex64)

        return self.static_potential_propagator
    '''
    def compute_gravity_propagator(self, psi, density,first_step=False, last_step=False,time_factor=1):
        """
        Compute the gravitational potential propagator based on current wave function density.

        Parameters:
            psi (cp.ndarray): Current wave function
            first_step (bool): True if this is the first step in evolution
            last_step (bool): True if this is the last step in evolution

        Returns:
            cp.ndarray: The gravity propagator in real space
        """

        if not self.simulation.use_gravity:
            return cp.ones_like(psi, dtype=cp.complex64)

        #density = self.compute_density(psi)

        a_s = (self.simulation.a_s * units.cm).to(f"{self.simulation.dUnits}").value


        if not self.simulation.use_self_int:
            self_int_potential = cp.ones_like(psi, dtype=cp.complex64)
        elif self.simulation.use_self_int:
            self_int_potential = self.get_self_int_potential(density, psi, a_s)



        # Solve Poisson equation for gravitational potential
        gravity_potential = self.solve_poisson(density)
        self.gravity_potential = gravity_potential + self_int_potential

        if first_step or last_step:
            self.gravity_propagator = cp.exp((-1j * ((self.h*time_factor) / 2) * self.gravity_potential )/(self.simulation.h_bar_tilde), dtype=cp.complex64)
        else:
            self.gravity_propagator = cp.exp((-1j  * (self.h*time_factor) * self.gravity_potential)/(self.simulation.h_bar_tilde), dtype=cp.complex64)


        return self.gravity_propagator

    '''

    def get_self_int_potential(self, density, psi,a_s):
        lambda_param =  (32*cp.pi*a_s*self.simulation.c)/self.simulation.h_bar

        #print(lambda_param)
        psi_squared = psi * psi  # Ψ·Ψ
        #psi_conj_squared = cp.conj(psi) * cp.conj(psi)  # Ψ†·Ψ†
        my_prefactor = lambda_param*self.simulation.h_bar_tilde**2 / (4*self.simulation.c)



        potential =  my_prefactor * (psi_squared+2*density*psi)

        return potential


    def compute_gravity_potential(self, density):
        """Compute gravitational potential only (no propagator)."""
        if not self.simulation.use_gravity:
            return cp.zeros_like(density, dtype=cp.float32)


        return self.solve_poisson(density)

    def compute_self_interaction_potential(self, density, psi):
        """Compute self-interaction potential only."""
        if not self.simulation.use_self_int:
            return cp.zeros_like(psi, dtype=cp.float32)

        a_s = (self.simulation.a_s * units.cm).to(f"{self.simulation.dUnits}").value
        return self.get_self_int_potential(density, psi, a_s)

    def compute_sponge_potential(self):
        """Get sponge potential - computed in simulation class ."""
        if self.simulation.sponge_potential is not None:
            return self.simulation.sponge_potential
        return cp.zeros_like(self.grids[0], dtype=cp.complex64)

    def compute_total_potential(self, psi, density, include_static=False):
        """
        Combine all dynamic potentials into one.

        Parameters:
            psi: Wave function
            density: Total density
            include_static: Whether to include static potential (usually handled separately)

        Returns:
            Total potential (real for gravity/self-int, complex if sponge included)
        """
        # Start with gravity
        V_total = self.compute_gravity_potential(density)

        # Add self-interaction
        V_total += self.compute_self_interaction_potential(density, psi)

        # Add sponge (complex, so convert V_total first)
        V_sponge = self.compute_sponge_potential()
        V_total = V_total.astype(cp.complex64) + V_sponge

        # Optionally add static potential
        if self.simulation.static_potential is not None:
            V_static = self.simulation.static_potential(self.simulation)
            V_total += V_static

        self.total_potential = V_total
        return V_total

    def compute_total_propagator(self, psi, density, first_step=False, last_step=False, time_factor=1):
        """
        Compute propagator from total potential.

        Returns:
            Propagator exp(-i * dt * V_total / hbar)
        """
        V_total = self.compute_total_potential(psi, density, include_static=False)

        if first_step or last_step:
            dt = (self.h * time_factor) / 2
        else:
            dt = self.h * time_factor

        propagator = cp.exp((-1j * dt * V_total) / self.simulation.h_bar_tilde, dtype=cp.complex64)
        return propagator


    def solve_poisson(self, density):
        """
        Solve the Poisson equation ∇²V = 4πGρ.

        Parameters:
            density (cp.ndarray): Mass density distribution

        Returns:
            cp.ndarray: Gravitational potential
        """



        # FFT of density
        density_k = cp.fft.fftn((density- cp.mean(density)).astype(cp.complex64))

        # Calculate k_squared_sum for Laplacian
        k_squared_sum = sum(k ** 2 for k in self.k_space)

        # Avoid division by zero at k=0
        mask = k_squared_sum == 0
        k_squared_sum[mask] = 1

        # Calculate potential in k-space: -4πGρ/k²
        potential_k = (-4 * cp.pi * self.G* density_k) / k_squared_sum.astype(cp.complex64)
        potential_k[mask] = 0  # Set k=0 mode to zero

        # Transform back to real space
        potential = cp.fft.ifftn(potential_k).real.astype(cp.float32)

        return potential