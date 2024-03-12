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

cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog_LEG.yml')
cat = hydromt.DataCatalog(yml)

scenarios = [
    'present_florence',
    'future_florence',
    "present_floyd",
    "future_floyd",
    "present_matthew",
    "future_matthew"
]
wrf_output_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_output'
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input'

yml_str = f""" meta:
    root: {root}
"""
data_lib = os.path.join(root, 'data_catalog_pgw.yml')
with open(data_lib, mode="a") as f:
    f.write(yml_str)

for scen in scenarios:
    dir_out = os.path.join(root, 'met', scen)
    if os.path.exists(dir_out) is False:
        os.mkdir(dir_out)

    subfolders = [f.name for f in os.scandir(os.path.join(wrf_output_dir, scen)) if f.is_dir()]
    for folder in subfolders:
        leaveout = ['ensmean_shifted', 'MSL_runs']
        if folder not in leaveout:
            wrf_dir = os.path.join(wrf_output_dir, scen, folder)
            fileout_name = scen + '_' + folder + '.nc'
            fileout_path = os.path.join(dir_out, fileout_name)

            if os.path.exists(fileout_path):
                continue
            else:
                print("Working on writing:", fileout_name)

                dfs = xr.open_mfdataset(paths=os.path.join(wrf_dir, '*.nc'),
                                        combine='nested',
                                        # compat='override'
                                        )
                print('Done reading in WRF outputs')

                if 'slp_int' in list(dfs.keys()):
                    ds = xr.Dataset(
                        data_vars=dict(
                            wind_u=(["time", "y", "x"], dfs['u_10m_gr'].data),
                            wind_v=(["time", "y", "x"], dfs['v_10m_gr'].data),
                            precip_g=(["time", "y", "x"], dfs['precip_g'].data),
                            mslp=(["time", "y", "x"], dfs['slp_int'].data),
                        ),
                        coords=dict(
                            x=(["x"], dfs['lon'].values[0, :]),
                            y=(["y"], dfs['lat'].values[:, 0]),
                            time=dfs['time'].values
                        ),
                        attrs=dfs.attrs,
                    )
                    ds['precip'] = ds['precip_g'].diff(dim='time')
                else:
                    ds = xr.Dataset(
                        data_vars=dict(
                            wind_u=(["time", "y", "x"], dfs['u_10m_gr'].data),
                            wind_v=(["time", "y", "x"], dfs['v_10m_gr'].data),
                            precip_g=(["time", "y", "x"], dfs['precip_g'].data),
                            mslp=(["time", "y", "x"], dfs['mslp'].data),
                            precip=(["time", "y", "x"], dfs['precip_r'].data),
                        ),
                        coords=dict(
                            x=(["x"], dfs['x'].values),
                            y=(["y"], dfs['y'].values),
                            time=dfs['time'].values
                        ),
                        attrs=dfs.attrs,
                    )

                ds2 = ds.rio.write_crs("epsg:4326").rio.set_spatial_dims(x_dim='x',
                                                                         y_dim='y').rio.write_coordinate_system(
                    inplace=True)
                ds2.to_netcdf(fileout_path)
                print('Done writing netcdf')

                # Check Plot
                ds['precip'].sum(dim='time').plot()
                figout = fileout_name.split('.')[0] + '_totalrain.png'
                plt.savefig(os.path.join(dir_out, figout))
                plt.close()
                print('Done plotting')

                # Add to Hydromt Data Catalog
                path_dc = f'met/{scen}/{fileout_name}'
                yml_str = f"""
{fileout_name.split('.')[0]}:
    path: {path_dc}
    data_type: RasterDataset
    driver: netcdf
    crs: 4326
    meta:
        category: meteo"""

                data_lib = os.path.join(root, 'data_catalog_pgw.yml')
                with open(data_lib, mode="a") as f:
                    f.write(yml_str)
                print('Done writing dataset to data catalog')
