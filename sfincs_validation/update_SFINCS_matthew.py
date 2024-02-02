import os
import datetime
import hydromt
import rasterio.merge
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Filepath to data catalog yml
#cat_dir = '/projects/sfincs/data'
cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

# Working directory and model root
os.chdir(r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2018_Florence\mod_v6')
#os.chdir('/projects/sfincs/')
root = 'flo_hindcast_v6_200m_LPD2m_avgN'
mod = SfincsModel(root=root, mode='r', data_libs=yml)
out_dir = r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2016_Matthew\matt_hindcast_v6_200m_LPD2m_avgN'
mod.update(out_dir)

# Updating config
mod.setup_config(
    **{
        'crsgeo': mod.crs.to_epsg(),
        "tref": "20161003 000000",
        "tstart": "20161003 000000",
        "tstop": "20161015 000000",
    }
)
print(mod.config)
mod.write_config(config_fn='sfincs.inp')


# Setup water level forcing
mod.setup_waterlevel_forcing(geodataset='adcirc_waterlevel_matthew',
                             offset='lmsl_to_navd88',
                             timeseries=None,
                             locations=None,
                             buffer=2000,
                             merge=False)
mod.write_forcing(data_vars='bzs')
gdf_locs = mod.forcing['bzs'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['bzs'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'bnd.shp'))
print('Write bzs')

# Setup discharge forcing
mod.setup_discharge_forcing(geodataset='usgs_discharge_matthew',
                            merge=False,
                            buffer=2000)
mod.write_forcing(data_vars='dis')
gdf_locs = mod.forcing['dis'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['dis'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'src.shp'))
print('Write dis')

# Setup gridded precipitation forcing
mod.setup_precip_forcing_from_grid(precip='mrms_tc_matthew',
                                   aggregate=False)
mod.write_forcing(data_vars='precip')
print('Write precip')

# Write wind forcing
mod.setup_wind_forcing_from_grid(wind='owi_matthew_winds')
mod.write_forcing(data_vars='wind')
print('Writing wind')
mod.write_forcing()

mod.write()

