from resources.Errors.Errors import *
import cupy as cp
from resources.Functions.Schrodinger_eq_functions import *
from resources.Classes.Propagator_Class import Propagator_Class
from resources.Classes.Evolution_Class import Evolution_Class
#from resources.Classes.Baryonic_liquid_Class import BaryonicMatter_Class
from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons
from resources.Classes.Nbody_classes.NBodyGas import NBodyGas
from resources.Classes.Nbody_classes.Sink_N_Body import SinkNBody
import functools
import sys
import inspect
from itertools import chain
import numpy as np
from astropy import units, constants
import math

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

    @parameter_check(int, list, int, (int, float), (int, float),int, float, float,bool, bool, object, (str, type(None)), bool,(type(None),dict), dict,bool,bool,float)
    def __init__(self, dim, boundaries, N, total_time, h,order_of_evolution = 2, m_s=2.5e-22, sponge_V0 =0.6 ,use_sponge=True, use_gravity=True,
                 static_potential=None, baryonic_model = None, save_max_vals=False, sink_formation=None,
                 sim_units={"dUnits": "kpc", "tUnits": "Gyr", "mUnits": "Msun", "eUnits": "eV"},use_units=True,self_int=True,a_s=-10e-80,):
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

        # initialize sponge potential
        self.use_sponge = use_sponge
        self.sponge_V0 = float(sponge_V0)
        self.sponge_potential = self._build_sponge_potential()

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
        self.num_of_w_vects_in_sim = 0


        self.use_self_int =self_int
        self.a_s = a_s

        self.baryonic_matter = []

        #I think this should make sim class backwards compatible (I aint testing that)
        self.baryonic_model = baryonic_model
        if self.baryonic_model is not None:
            self.add_baryons(
                Baryons(
                    self,
                    N_particles=5000,
                    total_mass=53090,
                    init_profile=self.baryonic_model,
                )
            )

        self.external_density = None
        self.overwrite_density = False

        self._sink_cfg = self._sink_cfg = None if sink_formation is None else dict(sink_formation) #configuration for sink particles

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
            self.c = 1

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


        self.dx = [(b - a) / self.N for (a, b) in self.boundaries]
        self.dV = np.prod(self.dx)
        self.cell_volume = self.dV  # Add this line to fix the AttributeError

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

            dx_dim = (b - a) / (self.N )
            dx_values.append(dx_dim)
            grids.append(np.linspace(a, b, self.N, endpoint=False))

        # Generate multidimensional grids
        mesh = np.meshgrid(*grids, indexing="ij")
        return dx_values, mesh

    def create_k_space(self):
        """
        Creates the k-space (wave vector space) for the simulation.

        Returns:
            list: List of cupy arrays representing k-space components
        """
        # Create k-space components with single-precision floats
        k_components = [
            2 * np.pi * cp.fft.fftfreq(self.N, d=self.dx[i]).astype(cp.float64)
            for i in range(self.dim)
        ]
        # Create multidimensional k-space
        k_space = np.meshgrid(*k_components, indexing='ij')
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
                print("nebo jdu sem")
        else:
            try:
                spin = wave_vector.spin
                wave_funcs = wave_vector.wave_vector[:]

                if spin not in self.wave_vectors:
                    self.wave_vectors[spin] = wave_funcs
                    print("existuju někdy tady?")
                    print(self.wave_vectors)
                else:
                    print("nebo dokonce sem")
                    for i, w_vect_component in enumerate(wave_funcs):
                        self.wave_vectors[spin][i].psi += w_vect_component.psi

                        print(f" i je {i}")
                        print(f"component {w_vect_component}")
                        print(self.wave_vectors)

                print(f"tohle končím s len {len(wave_vector.wave_vector)} {wave_vector}")
                print(f"a nejspíš bych měl pracovat s len {len(self.wave_vectors)} {self.wave_vectors}")
                #self.wave_functions.append(wave_vector.wave_vector)
                self.num_of_w_vects_in_sim +=1
                self.spin = wave_vector.spin

            except Exception as e:
                raise ValueError(f"Tried adding either a Wave_vector.wave_vector, list of Wave_functions or Wave_function but failed \n"
                                 f"got the error: {e}"
                                 f"try adding Wave_vector.wave_vector")

    def initialize_simulation(self):
        """
        Initialize the combined wave function and propagators before evolution.
        Sets up the Propagator and Evolution helper classes.
        """

        # Extract wave functions from the dictionary storage (used for Wave_vector_class)
        vlnky = []
        for spin in self.wave_vectors:
            vlnky.append(self.wave_vectors[spin])


        wave_vectors_flat = list(chain.from_iterable(vlnky))
        self.wave_functions.extend(wave_vectors_flat)


        self.wave_vectors = {}

        print(f"pracuji s len {len(self.wave_functions)} functions: {self.wave_functions}")

        # TODO: make it so that its possible to use sim without units
        if self.use_units:
            self.calculate_physical_units()

        if not self.check_time_step_restriction():
            return

        self.propagator = Propagator_Class(self)
        self.evolution = Evolution_Class(self, self.propagator, order=self.order_of_evolution)
        if self._sink_cfg is not None:
            try:
                # --- NEW: resolve auto-thresholds before passing to Evolution
                self._sink_cfg = self._resolve_sink_thresholds(dict(self._sink_cfg))
                self.evolution.enable_sink_particle_formation(**self._sink_cfg)
            except TypeError as e:
                raise TypeError(
                    f"Invalid sink_formation config keys/values: {self._sink_cfg}. "
                    f"Expected keys: density_threshold, consecutive_steps, check_interval."
                ) from e








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
        self.wave_values = self.evolution.scribe.wave_values
        self.accessible_times = self.evolution.scribe.accessible_times
        self.snapshot_directory = self.evolution.scribe.snapshot_directory

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
            phi_max = np.abs(potential_values).max()
        else:
            phi_max = 1e-10  # Small value if no potential is set

        # Avoid division by zero
        if phi_max < 1e-10:
            phi_max = 1e-10

        second_constraint = (2 * np.pi * (self.h_bar_tilde) * (1 / phi_max))

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

        scaling_lambda = 1

    def _build_sponge_potential(self):
        """ Builds spherical "sponge" potential as done in:PHYSICAL REVIEW D 94, 043513 (2016)
        V_s(r) = -i/2 * V0 * [ 2 + tanh((r - rs)/δ) - tanh(rs/δ) ] * Θ(r - rp).
        used to avoid spurious reheating from reflected waves emitted by soliton mergers
        carrying mass and energy toward box boundaries """
        if not self.use_sponge:
            return None



        grids_cpu = [cp.asnumpy(g) for g in self.grids]

        centers = []
        half_lengths = []
        for (a, b) in self.boundaries:
            c = 0.5 * (a + b)
            L = (b - a)
            centers.append(c)
            half_lengths.append(0.5 * L)


        r2 = np.zeros_like(grids_cpu[0], dtype=np.float64)
        for g, c in zip(grids_cpu, centers):
            r2 += (g - np.float64(c)) ** 2
        r = np.sqrt(r2, dtype=np.float64)

        #sponge params (paper values)
        rN = float(max(half_lengths))
        rp = (7.0 / 8.0) * rN
        rs = 0.5 * (rN + rp)
        delta = (rN - rp)
        V0 = np.float64(self.sponge_V0)

        # calar profile
        theta = (r > rp)
        S = 2.0 + np.tanh((r - rs) / delta) - np.tanh(rs / delta)
        Vs_cpu = (-0.5j * V0 * S * theta).astype(np.complex64)

        # freeing space
        del r2, r, S, theta

        # going to  GPU
        Vs_gpu = cp.asarray(Vs_cpu)

        return Vs_gpu

    def add_baryons(self, baryonic_system):
        """
        Attach one or more N-body baryonic components to this simulation.

        Parameters
        ----------
        baryonic_system : Baryons or list of NBodyBaryons
            Instance or list of resources.Classes.Baryonic_N_body.NBodyBaryons
            constructed with this Simulation_Class as the `simulation` argument.
        """
        from resources.Classes.Nbody_classes.Baryonic_N_body import Baryons

        # Handle list of baryonic systems
        if isinstance(baryonic_system, list):
            for baryon_sys in baryonic_system:
                self.add_baryons(baryon_sys)  # Recursive call for each
            return

        if not isinstance(baryonic_system, (Baryons, NBodyGas, SinkNBody)):
            raise TypeError(f"Expected Baryons, NBodyGas, or SinkNBody instance, got {type(baryonic_system)}")

        if baryonic_system.simulation is not self:
            raise ValueError(
                "Baryonic system was constructed with a different Simulation_Class "
                "instance. Construct NBodyBaryons with this simulation first."
            )

        # Append to list
        self.baryonic_matter.append(baryonic_system)



    def add_external_density(self, external_density):
        self.external_density = external_density
        self.overwrite_density = True

    def enable_sink_particle_formation(self, **cfg):
        """
        Store sink formation config and apply immediately if Evolution exists.

        use 'source' parameter for flexible sink creation.

        Parameters
        ----------
        **cfg : dict
            Configuration parameters including:
            - density_threshold : float
            - consecutive_steps : int
            - check_interval : int
            - source : str ('baryons', 'gas', or 'both')

        Example
        -------
         sim.enable_sink_particle_formation(
            density_threshold=1e5,
             consecutive_steps=5,
             source='baryons'  # Only create sinks from baryonic particles
         )
        """
        if getattr(self, "_sink_cfg", None) is None:
            self._sink_cfg = dict(cfg)
        else:
            self._sink_cfg.update(cfg)

        self._sink_cfg = self._resolve_sink_thresholds(dict(self._sink_cfg))


        if getattr(self, "evolution", None) is not None:
            try:
                self.evolution.enable_sink_particle_formation(**self._sink_cfg)
            except TypeError as e:
                raise TypeError(
                    f"Invalid sink_formation config keys/values: {self._sink_cfg}. "
                    f"Expected keys: density_threshold, consecutive_steps, check_interval, source. "
                    f"Update Evolution_Class.enable_sink_particle_formation signature accordingly."
                ) from e

    def _min_dx(self) -> float:
        return float(min(self.dx))

    def _truelove_rho_threshold(self, cs_eff: float, NJ: int) -> float:
        dx = self._min_dx()
        G = float(self.G)
        return cp.pi * (cs_eff ** 2) / (G * (NJ * dx) ** 2)

    def _get_first_gas_system(self):
        for b in self.baryonic_matter:
            # Evolution používá endswith("gas"); držím stejné pravidlo.
            if b.__class__.__name__.lower().endswith("gas"):
                return b
        return None

    def _estimate_gas_cs_eff(self, gas, cs_floor=None) -> float:
        # gas.cs může být scalar nebo pole; fallback na floor.
        cs = getattr(gas, "cs", None)
        cs_eff = None
        if cs is not None:
            try:
                cs_eff = float(cs)
            except TypeError:
                # pole -> vezmeme průměr (první rozumný default)
                cs_eff = float(cp.asnumpy(cp.mean(cs)))

        if cs_floor is not None:
            cs_eff = float(cs_eff) if cs_eff is not None else float(cs_floor)
            cs_eff = max(cs_eff, float(cs_floor))

        if cs_eff is None:
            raise ValueError("Cannot infer gas sound speed: gas.cs is missing and gas_cs_floor is None.")
        return cs_eff

    def _estimate_baryon_sigma_1d(self):
        # vezmeme všechny particle baryon systémy, co mají velocities
        systems = []
        for b in self.baryonic_matter:
            if b.__class__.__name__.lower().endswith("sink"):
                continue
            if hasattr(b, "N") and getattr(b, "N", 0) > 0 and hasattr(b, "velocities"):
                systems.append(b)

        if not systems:
            return None

        v = cp.concatenate([s.velocities for s in systems], axis=0)  # (Ntot,3)
        v_mean = cp.mean(v, axis=0)
        dv = v - v_mean[None, :]
        dv2 = cp.mean(cp.sum(dv * dv, axis=1))
        sigma_1d = cp.sqrt(dv2 / 3.0)
        return float(cp.asnumpy(sigma_1d))

    def _resolve_sink_thresholds(self, cfg: dict) -> dict:
        """
        Fill missing density thresholds from physics-based criteria.
        Gas:
          - if enable_gas_sinks and gas_density_threshold is None and gas_mode=='truelove':
              rho_thr = pi cs_eff^2 / [G (NJ dx)^2]
        Baryon particles:
          - if density_threshold is None and source includes baryons:
              use sigma_1d as cs_eff (pressure support ~ velocity dispersion)
              rho_thr = pi sigma_1d^2 / [G (NJ dx)^2]
        """
        source = cfg.get("source", "both")

        # --- GAS AUTO THRESHOLD
        if cfg.get("enable_gas_sinks", False):
            if cfg.get("gas_density_threshold", None) is None and cfg.get("gas_mode", "truelove") == "truelove":
                gas = self._get_first_gas_system()
                if gas is None:
                    pass
                else:
                    NJ = int(cfg.get("gas_NJ", 4))

                    cs_truelove = cfg.get("gas_cs_truelove", None)
                    cs_floor = cfg.get("gas_cs_floor", None)

                    if cs_truelove is not None:
                        cs_eff = float(cs_truelove)
                        if cs_floor is not None:
                            cs_eff = max(cs_eff, float(cs_floor))
                    else:
                        cs_eff = self._estimate_gas_cs_eff(gas, cs_floor=cs_floor)

                    cfg["gas_density_threshold"] = self._truelove_rho_threshold(cs_eff, NJ)


        if cfg.get("density_threshold", None) is None and source in ("baryons", "both"):
            sigma_1d = self._estimate_baryon_sigma_1d()
            if sigma_1d is None:
                pass
            else:
                NJ_b = int(cfg.get("baryon_NJ", cfg.get("gas_NJ", 4)))
                sigma_floor = cfg.get("baryon_sigma_floor", None)
                if sigma_floor is not None:
                    cfg["density_threshold"] = self._truelove_rho_threshold(float(sigma_1d), NJ_b)

        return cfg