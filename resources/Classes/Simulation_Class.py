from resources.Errors.Errors import *
import cupy as cp
from resources.Functions.Schrodinger_eq_functions import *
from resources.Classes.Propagator_Class import Propagator_Class
from resources.Classes.Evolution_Class import Evolution_Class
import pandas as pd
import functools
import sys
import inspect
import os
import datetime
from itertools import chain
import numpy as np
from astropy import units, constants
np.random.seed(1)

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


class Simulation_Class:
    """
    A class to manage simulation parameters and coordinate the wave function evolution.
    This class has been refactored to delegate evolution and propagator handling to
    separate classes.
    """

    @parameter_check(int, list, int, (int, float), (int, float),int, float, bool, object, bool, dict,bool,bool,float)
    def __init__(self, dim, boundaries, N, total_time, h,order_of_evolution = 2, m_s=10e-22, use_gravity=False,
                 static_potential=None, save_max_vals=False,
                 sim_units={"dUnits": "kpc", "tUnits": "Gyr", "mUnits": "Msun", "eUnits": "eV"},use_units=True,self_int=True,a_s=-10e-80):
        """
        Initialize the simulation parameters and setup.

        Parameters:
            dim (int): Number of dimensions
            boundaries (list[tuple]): List of tuples specifying the min and max for each dimension
            N (int): Number of spatial points for each dimension
            total_time (float): Total simulation time
            h (float): Time step size for propagation
            order_of_evolution (int): Order of evolution possible values are 2 and 4
            m_s (float): Mass parameter
            use_gravity (bool): Whether to include gravitational effects
            static_potential (callable): Function that returns static potential values
            save_max_vals (bool): Whether to save maximum values during evolution
            sim_units (dict): Simulation units configuration
        """
        # Setup parameters
        self.dim = dim
        self.boundaries = boundaries
        self.N = N
        self.total_time = total_time
        self.h = h
        self.num_steps = int(self.total_time / self.h)
        self.order_of_evolution = order_of_evolution

        if order_of_evolution not in (2, 4, 6):
            raise ValueError("order_of_evolution must be either 2, 4, 6. Raised while initializing Simulation_Class")

        # Gravity and potential settings
        self.use_gravity = use_gravity
        self.static_potential = static_potential
        self.save_max_vals = save_max_vals

        # Initialize spatial grids
        self.dx = []
        self.grids = []
        self.dx, self.grids = self.unpack_boundaries()

        # Initialize k-space for Fourier methods
        self.k_space = self.create_k_space()

        # Initialize wave functions storage
        self.wave_vectors = {}
        self.wave_functions = []  # List to store Wave_function instances
        self.wave_masses = []
        self.wave_momenta = []
        self.wave_omegas = []
        self.total_omega = 0
        self.combined_psi = None

        # Setup physical units
        self.use_units = use_units
        self.setup_units(sim_units, m_s)

        # Initialize helper classes
        self.propagator = None
        self.evolution = None
        self.snapshot_directory = None
        self.accessible_times = []
        self.wave_values = []

        self.use_self_int =self_int
        self.a_s = a_s

    def setup_units(self, sim_units, m_s):
        """
        Setup physical units for the simulation.

        Parameters:
            sim_units (dict): Dictionary with unit specifications
            m_s (float): Mass parameter
        """
        # Unpack the simulation units
        for key, value in sim_units.items():
            # Set the string value (like "kpc")
            setattr(self, key, value)

            # Create astropy unit objects
            if hasattr(units, value):
                setattr(self, f"{key}_unit", getattr(units, value))

        # Mass of the particle
        self.m_s = m_s
        self.mass_s = (m_s * self.eUnits_unit / constants.c ** 2).to(f"{self.mUnits}").value
        self.c = constants.c.to(f"{self.dUnits}/{self.tUnits}").value
        self.G = constants.G.to(f"{self.dUnits}3/({self.mUnits} {self.tUnits}2)").value
        self.h_bar = constants.hbar.to(f"{self.dUnits}2 {self.mUnits}/{self.tUnits}").value
        self.h_bar_tilde = (self.h_bar / self.mass_s)
        if not self.use_units:
            self.h_bar_tilde = 1
            self.h_bar = 1
            self.G = 1
            self.mass_s = 1

    def unpack_boundaries(self):
        """
        Validates the format of the boundaries and unpacks them into dx and multidimensional grids.

        Returns:
            tuple: (dx_values, grids)

        Raises:
            BoundaryFormatError: If boundaries are invalid
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
            grids.append(cp.linspace(a, b, self.N, endpoint=False))

        # Generate multidimensional grids
        mesh = cp.meshgrid(*grids, indexing="ij")
        return dx_values, mesh

    def create_k_space(self):
        """
        Creates the k-space (wave vector space) for the simulation.

        Returns:
            list: List of cupy arrays representing k-space components
        """
        # Create k-space components with single-precision floats
        k_components = [
            2 * cp.pi * cp.fft.fftfreq(self.N, d=self.dx[i]).astype(cp.float32)
            for i in range(self.dim)
        ]
        # Create multidimensional k-space
        k_space = cp.meshgrid(*k_components, indexing='ij')
        return k_space

    def add_wave_vector(self, wave_vector):
        """
        Add a wave function to the simulation.

        Parameters:
            wave_vector: A wave function object with psi, mass, and momenta attributes
        """
        if isinstance(wave_vector, list):
            for wave_function in wave_vector:
                self.wave_functions.append(wave_function)
        else:

            try:
                spin = wave_vector.spin
                if spin not in self.wave_vectors:
                    self.wave_vectors[spin] = wave_vector.wave_vector
                else:
                    for i, w_vect_component in enumerate(wave_vector.wave_vector):
                        self.wave_vectors[spin][i].psi += w_vect_component.psi
                self.wave_functions.append(wave_vector.wave_vector)

            except Exception as e:
                raise ValueError(f"Tried adding either a Wave_vector.wave_vector, list of Wave_functions or Wave_function but failed \n"
                                 f"got the error: {e}"
                                 f"try adding Wave_vector.wave_vector")

    def initialize_simulation(self):
        """
        Initialize the combined wave function and propagators before evolution.
        Sets up the Propagator and Evolution helper classes.
        """
        print(f"pracuji s {self.wave_functions}")
        if not self.wave_functions:
            raise ValueError("No wave functions added to the simulation")
        self.wave_functions = list(chain.from_iterable(self.wave_functions))

        if not self.check_time_step_restriction():
            return

        if self.use_units:
            self.calculate_physical_units()

        self.propagator = Propagator_Class(self)
        self.evolution = Evolution_Class(self, self.propagator,order=self.order_of_evolution)








    def evolve(self, save_every=1):
        """
        Start the evolution process.

        Parameters:
            save_every (int): How often to save the wave function during evolution
        """
        self.initialize_simulation()
        final_wave_functions = self.evolution.evolve(self.wave_functions, save_every)

        # Update simulation state
        self.combined_psi = final_wave_functions
        self.wave_values = self.evolution.wave_values
        self.accessible_times = self.evolution.accessible_times
        self.snapshot_directory = self.evolution.snapshot_directory

    def get_wave_function_at_time(self, time):
        """
        Retrieve the wave function at a specific time.

        Parameters:
            time (float): Time at which to retrieve the wave function

        Returns:
            cp.ndarray: Wave function at the specified time
        """
        if self.evolution is None:
            raise ValueError("Evolution has not been performed yet")

        return self.evolution.get_wave_function_at_time(time)

    def check_time_step_restriction(self):
        """
        Check if the time step satisfies the stability criteria.

        Returns:
            bool: True if time step is valid or user chooses to continue
        """
        # Get the minimum Δx (most restrictive)
        min_dx = min(self.dx)



        first_constraint = ((4) / (3 * cp.pi) * (1/self.h_bar_tilde)* min_dx ** 2)

        # Calculate second constraint based on potential
        if self.static_potential is not None:
            potential_values = self.static_potential(self)
            phi_max = cp.abs(potential_values).max()
        else:
            phi_max = 1e-10  # Small value if no potential is set

        # Avoid division by zero
        if phi_max < 1e-10:
            phi_max = 1e-10

        second_constraint = (2 * cp.pi * (self.h_bar_tilde) * (1 / phi_max))

        # Maximum allowed time step
        max_allowed_dt = 0.5 * min(float(first_constraint), float(second_constraint))

        # Check if current time step exceeds maximum allowed
        if self.h > max_allowed_dt:
            print(f"\nWARNING: Current time step h = {self.h} exceeds the stability criterion.")
            print(f"Maximum allowed time step: {max_allowed_dt}")
            print(f"  - From dispersion relation: {float(first_constraint)}")
            print(f"  - From potential term: {float(second_constraint)}")

            # Ask user what to do
            user_choice = input("Do you want to continue with the current time step anyway? (y/n): ")

            if user_choice.lower() != 'y':
                print("Simulation aborted due to time step restriction.")
                sys.exit(0)
                return False
            else:
                print("Continuing with user-specified time step despite stability concerns.")

        else:
            print(f"Time step h = {self.h} satisfies stability criterion (max allowed: {max_allowed_dt}).")

        return True

    def calculate_physical_units(self):
        """
        Convert the numerical wave function to physical units.
        """
        # Convert wave function: ψ_sol = ψ̂_sol * (ħ/√G)
        conversion_factor = self.h_bar_tilde / np.sqrt(self.G)
        for wf in self.wave_functions:
            wf.psi = wf.psi  * conversion_factor