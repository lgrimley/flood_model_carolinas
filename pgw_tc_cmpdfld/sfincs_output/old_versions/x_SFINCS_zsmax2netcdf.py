import os
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils

# This script loops through SFINCS models and saves the peak water levels for each simulation to a single netcdf

''' Loop through all model runs and save the peak water level output to a single netcdf'''


def get_hmax_da(mod_results_dir, depfile=None, hmin=0.05):
    da_hmax_list = []
    event_ids = []
    for dir in os.listdir(mod_results_dir):
        mod.read_results(fn_map=os.path.join(results_dir, dir, 'sfincs_map.nc'))
        zsmax = mod.results["zsmax"].max(dim='timemax')
        da_zsmax_list.append(zsmax)
        event_ids.append(dir)

        # Downscale results to get depth
        hmax = utils.downscale_floodmap(
            zsmax=zsmax,
            dep=mod.data_catalog.get_rasterdataset(depfile),
            hmin=hmin,
            gdf_mask=None,
            reproj_method='bilinear',
            floodmap_fn=os.path.join(out_dir, f'{dir}_hmax.tif'))
        da_hmax_list.append(hmax)

    da_hmax = xr.concat(da_hmax_list, dim='run')
    da_hmax['run'] = xr.IndexVariable('run', event_ids)
    return da_hmax


root = r'Z:\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence' \
       r'\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')

os.chdir(r'Z:\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_2')
out_dir = os.path.join(os.getcwd(), 'zsmax')
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)
os.chdir(out_dir)
if os.path.exists(os.path.join(out_dir, 'pgw_zsmax.nc')) is False:
    # Get peak water levels across all model runs
    da_zsmax_list = []
    event_ids = []
    for dir in os.listdir(results_dir):
        try:
            # Read SFINCS model results
            mod.read_results(fn_map=os.path.join(results_dir, dir, 'sfincs_map.nc'))
            # Get maximum water surface elevation
            zsmax = mod.results["zsmax"].max(dim='timemax')
            # Append to list
            da_zsmax_list.append(zsmax)
            event_ids.append(dir)
        except:
            print(dir)

    da_zsmax = xr.concat(da_zsmax_list, dim='run')
    da_zsmax['run'] = xr.IndexVariable('run', event_ids)
    da_zsmax.to_netcdf(os.path.join(out_dir, 'pgw_zsmax.nc'))
    print('Done writing zsmax for all runs!')
else:
    da_zsmax = xr.open_dataarray(os.path.join(out_dir, 'pgw_zsmax.nc'))




def calculate_fut_ensmean_zsmax(da_zsmax, storm, climate, scenario, type='mean'):
    nruns = 8
    if storm == 'matt':
        nruns = 7

    # Run IDs for Runoff
    run_ids = [f'{storm}_{climate}_SF{ii}_{scenario}' for ii in np.arange(1, nruns, 1)]

    # Run IFs for Coastal and Compound w/ sea level rise
    if scenario in ['compound', 'coastal']:
        for ii in range(1, nruns):
            run_ids = [f'{storm}_{climate}_SF{ii}_SLR{i}_{scenario}' for i in np.arange(1, 6, 1)]

    print(run_ids)

    if type == 'mean':
        meanOfMembers = da_zsmax.sel(run=run_ids).mean('run')
    elif type == 'max':
        meanOfMembers = da_zsmax.sel(run=run_ids).max('run')
    meanOfMembers.name = f'{storm}_{climate}_{scenario}_{type}'

    return meanOfMembers


for type in ['mean', 'max']:
    if os.path.exists(f'ensemble_zsmax_{type}.nc') is False:
        ensmean_list = []
        run_ids = []
        storms = ['flor', 'matt', 'floy']
        scenarios = ['coastal', 'runoff', 'compound']
        for storm in storms:
            for climate in ['fut']:
                for scenario in scenarios:
                    ensmean = calculate_fut_ensmean_zsmax(da_zsmax=da_zsmax,
                                                          storm=storm,
                                                          climate=climate,
                                                          scenario=scenario,
                                                          type=type
                                                          )
                    ensmean_list.append(ensmean)
                    run_ids.append(ensmean.name)
                    print(ensmean.name)

        da_ensmean = xr.concat(ensmean_list, dim='run')
        da_ensmean['run'] = xr.IndexVariable('run', run_ids)
        da_ensmean.to_netcdf(f'fut_ensemble_zsmax_{type}.nc')
    else:
        da_ensmean = xr.open_dataarray(f'fut_ensemble_zsmax_{type}.nc')



