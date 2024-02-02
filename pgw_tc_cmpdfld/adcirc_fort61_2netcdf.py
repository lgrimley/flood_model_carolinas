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
yml = os.path.join(r'Z:\users\lelise\data', 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\adcirc_output\matt_ensmean\future_0.13m')
ds = xr.open_dataset('fort.61.nc')
fileout = 'adcirc_wl_wrf_matt_ensmean_future_0.13m.nc'

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

ds_out.max()

ds_out.to_netcdf(fileout)
print('Done!')


