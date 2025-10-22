import cupy as cp
import numpy as np


class BaryonicMatter_Class:
    def __init__(self, simulation, M_b=1e9, a=0.5, model="hernquist"):
        self.simulation = simulation
        self.grids = simulation.grids
        self.G = simulation.G
        self.M_b = M_b  # total baryonic mass
        self.a = a      # scale radius (same units as simulation.dUnits)
        self.model = model.lower()


        self.rho_b = self.create_density_profiles()


    def drift_baryonic_matter(self):
        pass

    def compute_total_density(self):
        return self.rho_b


    def create_density_profiles(self):
        if self.model == "hernquist":
            return self.create_hernquist_profile()

    def create_hernquist_profile(self):
        """
        Create a Hernquist density profile:
            ρ(r) = (M_b / (2π)) * a / [r (r + a)^3]
        evaluated on the same grid as the simulation.
        taken from: https://articles.adsabs.harvard.edu/pdf/1990ApJ...356..359H
        eq: 2) and 5)
        """

        x, y, z = [np.asarray(g) for g in self.grids]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # avoids division by 0
        r[r == 0] = 1e-12

        Mb = self.M_b
        a = self.a

        rho = (Mb / (2 * np.pi)) * (a / (r * (r + a) ** 3))


        self.rho_b = np.asarray(rho.astype(np.float32))
        return self.rho_b