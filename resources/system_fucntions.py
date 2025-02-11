from resources.Functions.Schrodinger_eq_functions import *


def calculate_errors_between_num_and_analytical_evolution(wave_function, time_step):
    """
    Compares the numerical and analytical solutions of a wave function at a given time step
    and computes the error in both the wave function and the probability density.

    Parameters:
        wave_function (Wave_function): Instance of the Wave_function class, representing the simulated wavefunction.
        time_step (int): The time step at which the comparison will be made.

    Returns:
        dict: A dictionary with the following keys:
            - "numerical_psi": Numerical wave function at this time step.
            - "analytical_psi": Analytical wave function at this time step.
            - "wave function_error": The error between analytical and numerical wave functions.
            - "probability_density_error": The error between the probability densities.
            - "analytical_norm": Norm of the analytical solution at this step.
            - "numerical_norm": Norm of the numerical solution at this step.
    """
    # Get the time corresponding to the time step
    t = time_step * wave_function.h

    # Analytical solution at time t
    analytical_wave = cp.asnumpy(
        wave_function.psi_0 * cp.exp(-1j * energy_nd([0] * wave_function.dim, omega=1, hbar=1) * t)
    )

    # Numerical solution at time t
    numerical_wave = cp.asnumpy(wave_function.wave_function_at_time(t))

    # Compute errors between analytical and numerical solutions
    wavefunction_error = np.linalg.norm(analytical_wave - numerical_wave) / np.linalg.norm(analytical_wave)

    # Compute probability densities
    prob_density_analytical = np.abs(analytical_wave) ** 2
    prob_density_numerical = np.abs(numerical_wave) ** 2

    # Compute error in probability density
    probability_density_error = np.linalg.norm(prob_density_analytical - prob_density_numerical) / np.linalg.norm(
        prob_density_analytical)

    # Compute the norms of the wavefunctions for diagnostics
    analytical_norm = np.linalg.norm(prob_density_analytical)
    numerical_norm = np.linalg.norm(prob_density_numerical)

    return {
        "numerical_psi": numerical_wave,
        "analytical_psi": analytical_wave,
        "wave_function_error": wavefunction_error,
        "probability_density_error": probability_density_error,
        "analytical_norm": analytical_norm,
        "numerical_norm": numerical_norm
    }
