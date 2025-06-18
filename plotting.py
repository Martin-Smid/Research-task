from resources.Functions.system_fucntions import *


snapshot_directory = 'resources/data/simulation_20250611_101250' # Replace with your path

'''
wave_function_number = 6# Which wave function to plot

# Plot single wave function
data = plot_wave_function_snapshots(
snapshot_dir=snapshot_directory,
wf_number=wave_function_number,
z_index=None,  # Will use middle slice for 3D data
save_plots=True,
show_plots=False
    )
    '''

plot_multiple_wave_functions(
     snapshot_dir=snapshot_directory,
     wf_numbers=[0, 1, 2],  # List of wave functions to compare
     save_plots=True,
     show_plots=False)