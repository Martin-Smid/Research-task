import cupy as cp
import numpy as np


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

    def compute_kinetic_propagator(self):
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

        # Create propagator with time factor h/2 (for split-step)
        # --------------------------------------------------------------- ask about hbar/2 masses

        self.kinetic_propagator = cp.exp(((-1j * (self.h / 2) * k_squared_sum )*(self.h_bar_tilde)), dtype=cp.complex64)
        return self.kinetic_propagator

    def compute_static_potential_propagator(self, potential_function):
        """
        Compute the static potential propagator.

        Parameters:
            potential_function: Function that takes the simulation as input and returns potential values

        Returns:
            cp.ndarray: The potential propagator in real space
        """
        if potential_function is not None:
            potential_values = potential_function(self.simulation)
            self.static_potential_propagator = cp.exp((-1j * self.h * potential_values)*(self.simulation.h_bar_tilde**2) , dtype=cp.complex64)
        else:
            # If no potential is provided, use unit propagator (no effect)
            self.static_potential_propagator = cp.ones(self.simulation.grids[0].shape, dtype=cp.complex64)

        return self.static_potential_propagator

    def compute_gravity_propagator(self, psi, first_step=False, last_step=False):
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

        # Calculate density from wave function
        density = self.compute_density(psi)

        # Solve Poisson equation for gravitational potential
        gravity_potential = self.solve_poisson(density)

        # Store the potential for possible later use
        self.gravity_potential = gravity_potential

        # Create the propagator with appropriate time factor

        if first_step or last_step:
            # Half step for first and last steps (for split-step)
            self.gravity_propagator = cp.exp((-1j * (self.h / 2) * gravity_potential )/(self.simulation.h_bar_tilde), dtype=cp.complex64)

        else:
            # Full step for middle steps
            self.gravity_propagator = cp.exp((-1j  * self.h * gravity_potential)/(self.simulation.h_bar_tilde), dtype=cp.complex64)

        return self.gravity_propagator

    def compute_density(self, psi):
        """
        Calculate the density ρ = |ψ|².

        Parameters:
            psi: Wave function

        Returns:
            cp.ndarray: The density distribution
        """
        return cp.abs(psi).astype(cp.float32) ** 2

    def solve_poisson(self, density):
        """
        Solve the Poisson equation ∇²V = 4πGρ.

        Parameters:
            density (cp.ndarray): Mass density distribution

        Returns:
            cp.ndarray: Gravitational potential
        """
        # Remove mean density (required for periodic boundaries)
        density -= cp.mean(density)

        # FFT of density
        density_k = cp.fft.fftn(density.astype(cp.complex64))

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