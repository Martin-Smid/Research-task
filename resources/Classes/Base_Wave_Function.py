# resources/base_wave_function.py

class Base_Wave_function:
    """
    Base class for all wave functions.
    This contains functionality shared by `Wave_Packet_Class` and `Wave_function_class`.
    """

    def __init__(self, dim=1, boundaries=None, N=1024, h=0.01, total_time=1, potential=None, *args, **kwargs):
        """
        Initialize base wave function parameters.
        """
        self.dim = dim  # Number of dimensions
        self.boundaries = boundaries  # Domain boundaries
        self.N = N  # Number of spatial points
        self.h = h  # Time step size
        self.total_time = total_time  # Total simulation time
        self.potential = potential  # Potential function
        # Add other relevant shared attributes here

    def evolve_wave_function(self):
        """
        Placeholder to evolve the wave function in time (can be overridden by subclasses).
        """
        raise NotImplementedError("Base class does not implement this method.")
