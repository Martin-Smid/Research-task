from resources.Errors.Errors import *
import cupy as cp
from resources.Functions.Schrodinger_eq_functions import *

import functools
import sys
import inspect
import functools


def parameter_check(*types):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            parameters = list(sig.parameters.keys())

            # Check if it's a method by looking for 'self'
            is_method = parameters[0] == "self"

            # Get parameter names to check (excluding 'self' if it's a method)
            params_to_check = parameters[1:] if is_method else parameters

            # Make sure we don't try to check more parameters than types provided
            params_to_check = params_to_check[:len(types)]

            try:
                # Try to bind arguments
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                arg_dict = bound_args.arguments
            except TypeError as e:
                # Find the missing argument
                error_msg = str(e)
                if "missing a required argument" in error_msg:
                    # Extract the argument name from the error message
                    arg_name = error_msg.split("'")[1]
                    print(MissingArgumentError(arg_name, func.__name__))
                    sys.exit(0)
                raise  # Re-raise if it's a different TypeError

            # Skip 'self' in type checking if it's a method
            start_idx = 1 if is_method else 0

            # Check each parameter against its expected type
            for i, param_name in enumerate(params_to_check):
                if param_name not in arg_dict:
                    print(MissingArgumentError(param_name, func.__name__))
                    sys.exit(0)

                param_value = arg_dict[param_name]
                expected_type = types[i]

                if not isinstance(param_value, expected_type):
                    # Use the new TypeMismatchError
                    error = TypeMismatchError(
                        param_name,
                        expected_type,
                        type(param_value),
                        func.__name__
                    )
                    print(error)
                    sys.exit(0)

            return func(*args, **kwargs)

        return wrapper

    return decorator


#TODO: make a reliable way to pull wave function values at a given time
#TODO: make the plotting work for 2D and 3D
#TODO: figure out weher omega should go
#TODO: implement limit fo h
#TODO: make better descriptions of the methodes
#TODO: a lot more to check gravity function
#TODO: implement saving only certain snapshots in time
#TODO: check what happens when you include one more wave function on plot

class Simulation_class:
    """
    A class to manage simulation parameters and wave function evolution.

    Parameters:
        dim (int): Number of dimensions in the simulation (default = 1).
        boundaries (list[tuple]): List of tuples specifying the min and max for each dimension (e.g., [(-1, 1)]).
        N (int): Number of spatial points for each dimension (default = 1024).
        total_time (float): Total simulation time (default = 10).
        h (float): Time step size for propagation (default = 0.1).
        use_gravity (bool): Whether to include gravitational effects (default = False).
        static_potential (callable, optional): Function that returns static potential values.
    """

    @parameter_check(int, list, int, (int, float), (int, float), bool, object)
    def __init__(self, dim, boundaries, N, total_time, h, use_gravity=False, static_potential=None):
        # Setup parameters
        self.dim = dim
        self.boundaries = boundaries
        self.N = N
        self.total_time = total_time
        self.h = h  # Propagation parameter (time step size)
        self.num_steps = int(self.total_time / self.h)

        # Gravity and potential settings
        self.use_gravity = use_gravity
        self.static_potential = static_potential

        # Initialize spatial grids and wave function storage
        self.dx = []
        self.grids = []
        self.dx, self.grids = self.unpack_boundaries()

        # Initialize k-space for Fourier methods
        self.k_space = self.create_k_space()

        # Initialize wave functions storage
        self.wave_functions = []  # List to store Wave_function instances
        self.wave_masses = []  # List to store corresponding masses
        self.wave_momenta = []  # List to store corresponding momenta

        # Evolution data storage
        self.total_mass = 0  # Will be updated when wave functions are added
        self.combined_psi = None  # Will be initialized when evolution starts
        self.wave_values = []  # To store evolution snapshots

        # Will be computed before evolution starts
        self.static_potential_propagator = None
        self.kinetic_propagator = None

    def unpack_boundaries(self):
        """
        Validates the format of the boundaries and unpacks them into dx and multidimensional grids.
        Raises BoundaryFormatError if the boundaries are invalid.
        """
        if len(self.boundaries) != self.dim:
            raise BoundaryFormatError(
                message=f"Expected boundaries for {self.dim} dimensions but got {len(self.boundaries)}",
                tag="boundaries"
            )
        dx_values = []
        grids = []

        for i, (a, b) in enumerate(self.boundaries):
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise BoundaryFormatError(
                    message=f"Boundary {i} must be a tuple of two numbers, but got {(a, b)}",
                    tag=f"boundary_{i}"
                )
            if a >= b:
                raise BoundaryFormatError(
                    message=f"Boundary {i} values are invalid: {a} must be less than {b}",
                    tag=f"boundary_{i}"
                )
            # If the boundaries are valid, unpack them
            dx_dim = (b - a) / (self.N - 1)
            dx_values.append(dx_dim)
            grids.append(cp.linspace(a, b, self.N))

        # Generate multidimensional grids
        mesh = cp.meshgrid(*grids, indexing="ij")
        return dx_values, mesh

    def create_k_space(self):
        """Creates the k-space (wave vector space) for any arbitrary number of dimensions."""
        # Create k-space components with single-precision floats
        k_components = [
            2 * cp.pi * cp.fft.fftfreq(self.N, d=self.dx[i]).astype(cp.float32)
            for i in range(self.dim)
        ]
        # Create multidimensional k-space
        k_space = cp.meshgrid(*k_components, indexing='ij')
        return k_space

    def add_wave_function(self, wave_function):
        """
        Add a wave function to the simulation.

        Parameters:
            wave_function: A wave function object with psi, mass, and momenta attributes
        """
        self.wave_functions.append(wave_function)
        self.wave_masses.append(wave_function.mass)
        self.wave_momenta.append(wave_function.momenta)
        self.total_mass += wave_function.mass

        # Log that a wave function was added
        print(f"Added wave function with mass {wave_function.mass}. Total mass: {self.total_mass}")

    def initialize_simulation(self):
        """
        Initialize the combined wave function and propagators before evolution.
        This should be called after all wave functions have been added.
        """
        if not self.wave_functions:
            raise ValueError("No wave functions added to the simulation")

        # Combine all wave functions into one
        self.combined_psi = cp.zeros_like(self.wave_functions[0].psi, dtype=cp.complex64)
        for wave_func in self.wave_functions:
            self.combined_psi += wave_func.psi

        # Normalize the combined wave function
        norm = cp.sqrt(cp.sum(cp.abs(self.combined_psi) ** 2) * cp.prod(cp.array(self.dx)))
        self.combined_psi /= norm

        # Calculate combined momentum (weighted average)
        self.combined_momenta = []
        for i in range(self.dim):
            weighted_momentum = sum(wf.momenta[i] * wf.mass for wf in self.wave_functions) / self.total_mass
            self.combined_momenta.append(weighted_momentum)

        # Compute kinetic and static potential propagators
        self.kinetic_propagator = self.compute_kinetic_propagator()
        if self.static_potential:
            self.static_potential_propagator = self.compute_static_potential_propagator()
        else:
            self.static_potential_propagator = cp.ones_like(self.combined_psi, dtype=cp.complex64)

        # Save initial state
        self.wave_values = [self.combined_psi.copy()]

    def compute_kinetic_propagator(self):
        """Compute the kinetic propagator based on Fourier space components."""
        # Use single-precision floats to save memory
        k_shifted = [
            cp.array(k + (momentum / (2 * cp.pi)), dtype=cp.float32)
            for k, momentum in zip(self.k_space, self.combined_momenta)
        ]
        # Compute k_squared_sum in-place to save memory
        k_squared_sum = cp.zeros_like(k_shifted[0])
        for k in k_shifted:
            k_squared_sum += k ** 2
        return cp.exp(-1j * (self.h / 2) * k_squared_sum / self.total_mass, dtype=cp.complex64)

    def compute_static_potential_propagator(self):
        """Compute the static potential propagator."""
        potential_values = self.static_potential(self)
        return cp.exp(-1j * self.h * potential_values, dtype=cp.complex64)

    def compute_density(self, psi):
        """Calculate the density rho = m|psi|^2."""
        rho = self.total_mass * cp.abs(psi).astype(cp.float32) ** 2  # Ensure float32 precision
        return rho

    def solve_poisson(self, density):
        """
        Solve the Poisson equation ∇²V = 4πGρ, ignoring the zero mode.

        Parameters:
            density (cp.ndarray): Mass density ρ = m|ψ|².

        Returns:
            cp.ndarray: Gravitational potential V.
        """
        k_space = self.k_space  # Use existing k-space grids
        k_squared_sum = sum(k ** 2 for k in k_space)

        G = 1  # Gravitational constant
        density_k = cp.fft.fftn(density.astype(cp.complex64))

        # Set zero mode to 0 dynamically based on dimensions
        zero_mode_index = tuple([0] * self.dim)
        density_k -= cp.mean(density_k)

        # Protect against division by zero in k^2 sum
        k_squared_sum = cp.where(k_squared_sum == 0, 1e-12, k_squared_sum)

        # Compute potential in Fourier space with single precision
        potential_k = (4 * cp.pi * G * density_k) / k_squared_sum.astype(cp.complex64)

        # Transform back to real space, cast to real32
        potential = cp.fft.ifftn(potential_k).real.astype(cp.float32)
        return potential

    def update_gravity_potential(self, psi):
        """Compute gravitational potential propagator based on current density."""
        if self.use_gravity:
            density = self.compute_density(psi)
            gravity_potential = self.solve_poisson(density)
            return cp.exp(-1j * self.h * gravity_potential, dtype=cp.complex64)
        return cp.ones_like(psi, dtype=cp.complex64)

    def evolve_wavefunction_split_step(self, psi, step_index, total_steps):
        """
        Evolve the wavefunction using the split-step Fourier method.

        Parameters:
            psi (cp.ndarray): Initial wave function.
            step_index (int): Index of the current step (starting at 0).
            total_steps (int): Total number of steps in the simulation.

        Returns:
            cp.ndarray: Evolved wave function after one time step.
        """
        # Update gravity potential if enabled
        gravity_propagator = self.update_gravity_potential(psi)

        # If it's the first step, apply half the potential propagator
        if step_index == 0:
            psi *= cp.sqrt(self.static_potential_propagator * gravity_propagator)
        else:
            psi *= self.static_potential_propagator * gravity_propagator

        # Apply kinetic evolution in Fourier space
        psi_k = cp.fft.fftn(psi)
        psi_k *= self.kinetic_propagator
        psi = cp.fft.ifftn(psi_k)

        # If it's the last step, apply only a half potential step
        if step_index == total_steps - 1:
            psi *= cp.sqrt(self.static_potential_propagator * gravity_propagator)

        return psi

    def evolve(self, save_every=1):
        """
        Perform the full time evolution for the combined wave function.

        Parameters:
            save_every (int): Frequency of saving the wave function values. Defaults to 1 (saves every step).

        Returns:
            None
        """
        # Make sure simulation is initialized
        if self.combined_psi is None:
            self.initialize_simulation()

        psi = self.combined_psi.copy()  # Work with a copy of the initial state

        for step in range(self.num_steps):
            # Perform the evolution step
            psi = self.evolve_wavefunction_split_step(
                psi, step_index=step, total_steps=self.num_steps
            )

            # Save the wavefunction at the specified intervals
            if step % save_every == 0 and step > 0:  # Skip step 0 as it was saved during initialization
                self.wave_values.append(psi.copy())

            # Free memory when possible
            if step % save_every != 0:
                cp.get_default_memory_pool().free_all_blocks()

        # Update the final state
        self.combined_psi = psi

        # Ensure the last state is saved if it wasn't already
        if (self.num_steps - 1) % save_every != 0:
            self.wave_values.append(psi.copy())

        # Free GPU memory once the evolution is complete
        cp.get_default_memory_pool().free_all_blocks()

        print(f"Evolution completed with {len(self.wave_values)} saved states")
