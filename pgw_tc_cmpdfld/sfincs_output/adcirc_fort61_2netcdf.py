#!/usr/bin/env python3
# coding: utf-8
#
# -------------- Script Overview -------------------
#
# Converting ADCIRC fort.61.nc to Hydromt compatible netcdf
#
#   Author: Lauren Grimley, lauren.grimley@unc.edu
#   Last edited by: LEG 2/6/24
#
# Inputs:
#   fort.61.nc - contains ADCIRC water level outpoints at specified points
#
# Outputs:
#   netcdf that can be read by hydromt 
#
# Dependencies:
#
# Needed updates:
#   alternative to using a template dataset (?) 

import os
import numpy as np
from datetime import datetime
import pandas as pd
from netCDF4 import Dataset, date2num
import matplotlib.pyplot as plt
import xarray as xr
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel, utils

# Filepath to data catalog yml
yml = os.path.join(r'Z:\users\lelise\data', 'data_catalog_SFINCS_Carolinas.yml')
cat = hydromt.DataCatalog(yml)

subfolder = [
    # 'present_florence',
    # 'future_florence',
    "present_floyd",
    # "future_floyd",
    # "present_matthew",
    # "future_matthew"
             ]

for sb in subfolder:
    path = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\adcirc_output', sb)
    dir_out = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\waterlevel', sb)

    if os.path.exists(dir_out) is False:
        os.mkdir(dir_out)

    for root, dirs, files in os.walk(path):
        for _dir in dirs:

            climate = sb.split('_')[0]
            storm = sb.split('_')[-1]

            if storm == 'floyd':
                ss = 'floy'
            elif storm == 'florence':
                ss = 'flor'
            elif storm == 'matthew':
                ss = 'matt'

            if climate == 'present':
                cc = 'pres'
            elif climate == 'future':
                cc = 'fut'

            rr = _dir.split('_')[-1]
            name = f'{ss}_{cc}_{rr}.nc'
            fileout = os.path.join(dir_out, name)
            if os.path.exists(fileout):
                continue

            ds = xr.open_dataset(os.path.join(root, _dir, 'fort.61.nc'))
            template = cat.get_geodataset('usgs_waterlevel_florence')
            ds_out = xr.Dataset(
                data_vars=dict(
                    waterlevel=(["time", "index"], ds['zeta'].data),
                ),
                coords=dict(
                    x=(['index'], ds['x'].data),
                    y=(['index'], ds['y'].data),
                    index=(["index"], ds['station'].data),
                    time=ds['time'].data,
                    spatial_ref=template['spatial_ref'],
                ),
                attrs=ds.attrs,
            )

            ds_out.to_netcdf(fileout)
            print('Done!')


# HINDCAST RUN PROCESSING
sb = 'floy_ERA5'
path = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\adcirc_output\floy_ERA5')
dir_out = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\hindcast\waterlevel')

name = f'{sb}.nc'
fileout = os.path.join(dir_out, name)

ds = xr.open_dataset(os.path.join(path, 'fort.61.nc'))
template = cat.get_geodataset('usgs_waterlevel_florence')
ds_out = xr.Dataset(
    data_vars=dict(
        waterlevel=(["time", "index"], ds['zeta'].data),
    ),
    coords=dict(
        x=(['index'], ds['x'].data),
        y=(['index'], ds['y'].data),
        index=(["index"], ds['station'].data),
        time=ds['time'].data,
        spatial_ref=template['spatial_ref'],
    ),
    attrs=ds.attrs,
)

ds_out.to_netcdf(fileout)
print('Done!')

