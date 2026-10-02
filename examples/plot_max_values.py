"""Plot normalized maximum-density histories."""

from Plot_max_vals import plot_max_values


# Leave empty to plot the newest run, or add paths to compare selected runs.
simulation_directories = [
    # "resources/data/simulation_...",
    # "resources/data/simulation_...",
]

plot_max_values(*simulation_directories)
