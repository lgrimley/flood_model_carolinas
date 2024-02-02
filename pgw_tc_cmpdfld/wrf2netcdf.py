import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel, utils

# Script for reading WRF model output and writing to combined netcdf
# Author: Lauren Grimley
# Last updated: 08/21/2023

path = r'Z:\users\lelise\projects\Carolinas\Chapter2\wrf_output_20231006\matt_ensmean'
dir_out = path


for root, dirs, files in os.walk(path):
    for _dir in dirs:
        fileout = os.path.join(dir_out, _dir) + '.nc'
        print(fileout)

        if os.path.exists(fileout):
            continue
        else:
            print("Working on writing " + _dir)
            dfs = xr.open_mfdataset(paths=os.path.join(os.path.join(root, _dir) + '/*.nc'),
                                    combine='nested')

            ds = xr.Dataset(
                data_vars=dict(
                    u_10m_gr=(["time", "y", "x"], dfs['u_10m_gr'].data),
                    v_10m_gr=(["time", "y", "x"], dfs['v_10m_gr'].data),
                    precip_g=(["time", "y", "x"], dfs['precip_g'].data),
                    mslp=(["time", "y", "x"], dfs['slp_int'].data),
                    # T_2m=(["time", "y", "x"], dfs['T_2m'].data),
                    # precip_pmm=(["time", "y", "x"], dfs['precip_pmm'].data),
                ),
                coords=dict(
                    x=(["x"], dfs['lon'].values[0, :]),
                    y=(["y"], dfs['lat'].values[:, 0]),
                    time=dfs['time'].values
                ),
                attrs=dfs.attrs,
            )

            ds['precip_r'] = ds['precip_g'].diff(dim='time')

            ds2 = ds.rio.write_crs("epsg:4326").rio.set_spatial_dims(x_dim='x',
                                                                     y_dim='y').rio.write_coordinate_system(
                inplace=True)

            ds2.to_netcdf(os.path.join(dir_out, fileout))

            # Sanity Check
            ds['precip_r'].sum(dim='time').plot()
            plt.savefig(os.path.join(dir_out, _dir) + '_sum_precip_r.png')
            plt.close()

            ds['precip_g'][-1].plot()
            plt.savefig(os.path.join(dir_out, _dir) + '_precip_g.png')
            plt.close()

            print('Done!')

# fileout = 'flor_ensmean_present.nc'
# tmp = os.path.join(path) + "/*.nc"
# dfs = xr.open_mfdataset(paths=os.path.join(tmp), combine='nested')


