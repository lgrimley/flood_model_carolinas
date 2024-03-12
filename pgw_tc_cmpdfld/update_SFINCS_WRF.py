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
# cat_dir = '/projects/sfincs/data'
cat_dir_pgw = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input'
yml_pgw = os.path.join(cat_dir_pgw, 'data_catalog_pgw.yml')
cat_pgw = hydromt.DataCatalog(yml_pgw)

cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog_LEG.yml')
cat = hydromt.DataCatalog(yml)

# Working directory and model root
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\present_matthew\ensmean\matt_ensmean_present'
mod = SfincsModel(root=root, mode='r', data_libs=[yml, yml_pgw])

mod.update(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\future_matthew\future_matthew_ensmean')
met_file = 'future_matthew_ensmean'
wl_file = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\waterlevel\future_matthew' \
          r'\future_matthew_ensmean.nc'

region = mod.mask.where(mod.mask == 2, 0).raster.vectorize()
wl_df = cat.get_geodataset(data_like=wl_file, geom=region, buffer=500)
mod.setup_waterlevel_forcing(geodataset=wl_df,
                             offset='lmsl_to_navd88',
                             timeseries=None,
                             locations=None,
                             buffer=500,
                             merge=False)
mod.write_forcing(data_vars='bzs')

bzs = mod.forcing['bzs']
index_to_remove = bzs['index'][bzs.argmax().values.item()].values.item()
cleaned_bcs = bzs.drop_sel(index=index_to_remove)
mod.setup_waterlevel_forcing(geodataset=cleaned_bcs,
                             #offset='lmsl_to_navd88',
                             timeseries=None,
                             locations=None,
                             buffer=500,
                             merge=False)
mod.write_forcing(data_vars='bzs')
# to_remove = []
# counter = 0
# for ind in bzs['index']:
#     zs = bzs.sel(index=ind)
#     if zs.max().values.item() > 10:
#         to_remove.append(ind)
#     counter += 1
#     print(counter)
# cleaned_bcs = wl_df.drop_sel(index=to_remove)

gdf_locs = mod.forcing['bzs'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['bzs'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'bnd.shp'))
print('Write bzs')

# Setup discharge forcing
# mod.setup_discharge_forcing(geodataset='usgs_discharge_florence',
#                             merge=False,
#                             buffer=2000)
# mod.write_forcing(data_vars='dis')
# gdf_locs = mod.forcing['dis'].vector.to_gdf()
# gdf_locs['name'] = mod.forcing['dis'].index.values
# gdf_locs.to_file(os.path.join(mod.root, 'gis', 'src.shp'))
#print('Write dis')

# Setup gridded precipitation forcing
mod.setup_precip_forcing_from_grid(precip=met_file,
                                   aggregate=False)
print('Write precip')

# Write wind forcing
mod.setup_wind_forcing_from_grid(wind=met_file)
print('Writing wind')

_ = mod.plot_forcing(fn_out='forcings_hydro.png',
                     forcings=['bzs', 'dis'])
plt.close()

_ = mod.plot_forcing(fn_out='forcings_meteo.png',
                     forcings=['precip_2d', 'wind_u', 'wind_v'])
plt.close()

mod.write_forcing()
mod.write()
