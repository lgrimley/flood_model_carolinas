import os
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils


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


root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')
fileout = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\pgw_zsmax.nc'
# Loop through model results and delete subgrid file
results_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\arch'
for root, dirs, files in os.walk(results_dir):
    for name in files:
        if name.endswith("sfincs.sbg"):
            os.remove(os.path.join(root, name))

# Get peak water levels across all model runs
da_zsmax_list = []
event_ids = []
for dir in os.listdir(results_dir):
    mod.read_results(fn_map=os.path.join(results_dir, dir, 'sfincs_map.nc'))
    zsmax = mod.results["zsmax"].max(dim='timemax')
    da_zsmax_list.append(zsmax)
    event_ids.append(dir)
da_zsmax = xr.concat(da_zsmax_list, dim='run')
da_zsmax['run'] = xr.IndexVariable('run', event_ids)
da_zsmax.to_netcdf(fileout)
print('Done writing zsmax for all runs!')