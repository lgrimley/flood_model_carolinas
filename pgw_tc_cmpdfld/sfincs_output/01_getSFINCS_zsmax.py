import os.path
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
depfile = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_20m.tif'
dep_da = mod.data_catalog.get_rasterdataset(depfile)

# Directory of the SFINCS model results
results_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\02_sfincs_models_future'

# Output directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3')

''' Loop through all model runs and save the peak water level output to a single netcdf'''

# Get peak water levels across all model runs
da_zsmax_list = []
da_hmax_list = []
da_vmax_list = []
da_tmax_list = []
event_ids = []

# Loop through all the model dirs to read the results
for dir in os.listdir(results_dir):
    try:
        # Read SFINCS model results
        mod.read_results(fn_map=os.path.join(results_dir, dir, 'sfincs_map.nc'))

        '''Get maximum water surface elevation'''
        zsmax = mod.results["zsmax"].max(dim='timemax')
        # Append this data array to a list
        da_zsmax_list.append(zsmax)

        ''' Get max depth '''
        if 'compound' in dir and os.path.exists(rf'.\flood_depths_20m\{dir}.nc') is False:
            # Downscale results to get depth
            hmax = utils.downscale_floodmap(
                zsmax=zsmax,
                dep=dep_da,
                hmin=0.1,
                gdf_mask=None,
                reproj_method='nearest',
                floodmap_fn=None
            )
            hmax.name = dir
            hmax.to_netcdf(rf'.\flood_depths_20m\{dir}.nc')

        ''' Get the maximum velocity '''
        vmax = mod.results["vmax"].max(dim='timemax')
        # Append this data array to a list
        da_vmax_list.append(vmax)

        ''' Get the time of inundation above twet_threshold '''
        tmax = mod.results['tmax'].max(dim='timemax')
        # Create a mask of the NaT values
        mask = np.isnat(tmax.values)
        # Convert timedelta64[ns] to float
        tmax = tmax.astype(float)
        # Mask out the NaT values
        tmax = tmax.where(~mask, np.nan)
        # Convert nanoseconds to hours
        tmax = tmax / (3.6 * 10 ** 12)
        # Subset the data further with a minimum time threshold of interest
        twet_min = 0  # hours
        tmax = xr.where(tmax >= twet_min, tmax, np.nan)
        da_tmax_list.append(tmax)

        # Keep a list of the event ids
        event_ids.append(dir)
        print(dir)
    except:
        print(f'Something mess up for {dir}')

# Write this to a netcdf
# Combine data array using xarray with new dimension "run" with model name
da_zsmax = xr.concat(da_zsmax_list, dim='run')
da_zsmax['run'] = xr.IndexVariable('run', event_ids)
#da_zsmax.to_netcdf('pgw_zsmax.nc')

da_vmax = xr.concat(da_vmax_list, dim='run')
da_vmax['run'] = xr.IndexVariable('run', event_ids)
da_vmax.to_netcdf('pgw_vmax.nc')

da_tmax = xr.concat(da_tmax_list, dim='run')
da_tmax['run'] = xr.IndexVariable('run', event_ids)
da_tmax.to_netcdf('pgw_tmax.nc')

print('Done writing zsmax, tmax, vmax for all runs!')

test = xr.open_dataset(r'C:\Users\lelise\Desktop\tmp\depth_New Hanover.tif')
depth = test.sel(band=2)
depth = depth['band_data'].where(depth['band_data']>0.1)
depth.raster.to_raster(r'C:\Users\lelise\Desktop\tmp\depth_NewHanover_2.tif', nodata=-9999.0)