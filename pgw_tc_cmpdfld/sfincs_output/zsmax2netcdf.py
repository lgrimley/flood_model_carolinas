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

root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')
fileout = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\pgw_zsmax.nc'
# Loop through model results and delete subgrid file
results_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\arch'
for root, dirs, files in os.walk(results_dir):
    for name in files:
        if name.endswith("sfincs.sbg"):
            os.remove(os.path.join(root, name))


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


os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis')
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
            mod.read_results(fn_map=os.path.join(results_dir, dir, 'sfincs_map.nc'))
            zsmax = mod.results["zsmax"].max(dim='timemax')
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


''' Calculate SFINCS ensemble mean/max water levels '''


def calculate_ensmean_zsmax(da_zsmax, storm, climate, scenario, type='mean'):
    nruns = 8
    if storm == 'matt':
        nruns = 7

    ids = [f'{storm}_{climate}_ens{ii}_{scenario}' for ii in np.arange(1, nruns, 1)]
    if climate == 'presScaled':
        if scenario in ['compound', 'coastal']:
            for ii in range(1, nruns):
                ids = [f'{storm}_{climate}_ens{ii}_SLR{i}_{scenario}' for i in np.arange(1, 6, 1)]
    print(ids)
    if type == 'mean':
        meanOfMembers = da_zsmax.sel(run=ids).mean('run')
    elif type == 'max':
        meanOfMembers = da_zsmax.sel(run=ids).max('run')
    meanOfMembers.name = f'{storm}_{climate}_{scenario}_{type}'

    return meanOfMembers


for type in ['mean', 'max']:
    if os.path.exists(f'pgw_ensmean_zsmax_{type}.nc') is False:
        ensmean_list = []
        run_ids = []
        storms = ['flor', 'floy', 'matt']
        climates = ['pres', 'presScaled']
        scenarios = ['coastal', 'runoff', 'compound']
        for storm in storms:
            for climate in climates:
                for scenario in scenarios:
                    ensmean = calculate_ensmean_zsmax(da_zsmax=da_zsmax, storm=storm, climate=climate,
                                                      scenario=scenario, type=type)
                    ensmean_list.append(ensmean)
                    run_ids.append(ensmean.name)
                    print(ensmean.name)

        da_ensmean = xr.concat(ensmean_list, dim='run')
        da_ensmean['run'] = xr.IndexVariable('run', run_ids)
        da_ensmean.to_netcdf(f'pgw_ensmean_zsmax_{type}.nc')
    else:
        da_ensmean = xr.open_dataarray(f'pgw_ensmean_zsmax_{type}.nc')
