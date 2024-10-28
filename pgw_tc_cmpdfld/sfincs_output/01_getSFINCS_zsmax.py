import sys
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\pgw_tc_cmpdfld\sfincs_output')
from pgw_utils import *

''' 
Script:
Author: L Grimley
Last Updated: 8/20/24

Description: 
    This script is used for plotting water level differences for the PGW ensmean for coastal, runoff, and compound drivers
    This script reads in the netCDF of zsmax and plots the difference between the future and present

'''

# Load SFINCS model
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')
mod.read()

# Directory of the SFINCS model results
results_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\02_sfincs_models_future'

# Output directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3')

''' Loop through all model runs and save the peak water level output to a single netcdf'''

# Get peak water levels across all model runs
da_zsmax_list = []
event_ids = []

# Loop through all the model dirs to read the results
for dir in os.listdir(results_dir):
    try:
        # Read SFINCS model results
        mod.read_results(fn_map=os.path.join(results_dir, dir, 'sfincs_map.nc'))
        # Get maximum water surface elevation
        zsmax = mod.results["zsmax"].max(dim='timemax')
        # Append this data array to a list
        da_zsmax_list.append(zsmax)
        event_ids.append(dir)
    except:
        print(dir)

# Combine data array using xarray with new dimension "run" with model name
da_zsmax = xr.concat(da_zsmax_list, dim='run')
da_zsmax['run'] = xr.IndexVariable('run', event_ids)

# Write this to a netcdf
da_zsmax.to_netcdf('pgw_zsmax.nc')
print('Done writing zsmax for all runs!')
