from resources.Functions.Schrodinger_eq_functions import *
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_max_values_on_N(simulation_class_instance):
    import pandas as pd
    import matplotlib.pyplot as plt

    # File name
    filename = simulation_class_instance.max_vals_filename

    # Read the CSV file, skipping comment lines (lines starting with '#')
    data = pd.read_csv(filename, comment='#')

    # Extract the first row (N values) for column names
    n_values = data.columns[1:]  # Skip the first column (time step)
    n_values = [float(n) for n in n_values]  # Convert to float for proper naming

    # Rename columns for clarity
    data.columns = ['Time Step'] + [f"N = {n}" for n in n_values]

    # Normalize the data: (value / first_value) - 1
    for col in data.columns[1:]:
        first_val = data[col].iloc[0]
        data[col] = (data[col] / first_val) - 1  # Now starts at 0

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot each column of data (except the first column, which is the x-axis)
    for col in data.columns[1:]:
        plt.plot(data['Time Step'], data[col], label=col)

    # Add labels, title, and legend
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("Normalized Max Values (Start = 0)", fontsize=18)
    plt.legend(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    # Show the plot
    plt.show()