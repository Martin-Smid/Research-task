from resources.Functions.system_fucntions import *


snapshot_directory = 'resources/data/simulation_20250618_151117' # Replace with your path

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
     wf_numbers=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],  # List of wave functions to compare
     save_plots=True,
     show_plots=False)