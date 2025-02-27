from resources.Errors.Errors import *
import cupy as cp


class Simulation_class():
    """
        A class to manage simulation parameters.

        Parameters:
        dim (int): Number of dimensions in the simulation (default = 1).
        boundaries (list[tuple]): List of tuples specifying the min and max for each dimension (e.g., [(-1, 1)]).
        N (int): Number of spatial points for each dimension (default = 1024).
        total_time (float): Total simulation time (default = 10).
        h (float): Time step size for propagation (default = 0.1).
    """

    def __init__(
            self,
            dim=1,
            boundaries=[(-1, 1)],
            N=1024,
            total_time=10,
            h=0.1,
            dx=(1 + 1) / (1024 - 1)
    ):
        # Setup parameters
        self.dim = dim
        self.boundaries = boundaries
        self.N = N
        self.total_time = total_time
        self.h = h  # Propagation parameter (time step size)
        self.num_steps = int(self.total_time / self.h)
        self.dx = []
        self.grids = []

        # Compute dx and spatial grids for each dimension
        self.dx, self.grids = self.unpack_boundaries()

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
            self.dx.append(dx_dim)
            self.grids.append(cp.linspace(a, b, self.N))
        # Generate multidimensional grids
        mesh = cp.meshgrid(*self.grids, indexing="ij")
        return self.dx, mesh
