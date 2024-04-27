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
yml = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
yml_Base = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
cat = hydromt.DataCatalog(yml)

# Working directory and model root
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\present_florence\ensmean\flor_ensmean_present'
mod = SfincsModel(root=root, mode='r', data_libs=[yml, yml_pgw, yml_Base])


mod.update(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\test')

mod.setup_structures('levees_carolinas',
                     stype='weir',
                     dep=None,
                     buffer=None,
                     merge=False,
                     dz=0)
mod.write()
# Updating config
# mod.setup_config(
#     **{
#         'crsgeo': mod.crs.to_epsg(),
#         "tref": "19990913 000000",
#         "tstart": "19990913 000000",
#         "tstop": "19990922 000000",
#     }
# )
# print(mod.config)
# mod.write_config(config_fn='sfincs.inp')

wl_file = 'flor_pres_spinup_waterlevel'
region = mod.mask.where(mod.mask == 2, 0).raster.vectorize()
wl_df = mod.data_catalog.get_geodataset(data_like=wl_file, crs=4326)
mod.setup_waterlevel_forcing(geodataset=wl_df,
                             #offset='lmsl_to_navd88',
                             timeseries=None,
                             locations=None,
                             #buffer=5000,
                             merge=False)
mod.write_forcing(data_vars='bzs')

bzs = mod.forcing['bzs']
index_to_remove = bzs['index'][bzs.argmax().values.item()].values.item()
# index_to_remove = [8665530, 8670870, 8720030]
cleaned_bcs = bzs.drop_sel(index=index_to_remove)
mod.setup_waterlevel_forcing(geodataset=cleaned_bcs,
                             #offset='lmsl_to_navd88',
                             timeseries=None,
                             locations=None,
                             buffer=500,
                             merge=False)
mod.write_forcing(data_vars='bzs')
gdf_locs = mod.forcing['bzs'].vector.to_gdf()
gdf_locs['name'] = mod.forcing['bzs'].index.values
gdf_locs.to_file(os.path.join(mod.root, 'gis', 'bnd.shp'))
print('Write bzs')

# # Setup discharge forcing
# mod.setup_discharge_forcing(geodataset='usgs_discharge_floyd',
#                             merge=False,
#                             buffer=2000)
# mod.write_forcing(data_vars='dis')
# gdf_locs = mod.forcing['dis'].vector.to_gdf()
# gdf_locs['name'] = mod.forcing['dis'].index.values
# gdf_locs.to_file(os.path.join(mod.root, 'gis', 'src.shp'))
# print('Write dis')

# # Setup gridded precipitation forcing
# mod.setup_precip_forcing_from_grid(precip='era5_1999',
#                                    aggregate=False)
# print('Write precip')
#
# # Write wind forcing
# mod.setup_wind_forcing_from_grid(wind='era5_1999')
# print('Writing wind')
# mod.write_forcing()

figout = os.path.join(mod.root, 'bcs.png')
fig, axes = mod.plot_forcing()
axes = axes.flatten()
for ax in axes:
    ax.legend().remove()
plt.subplots_adjust(left=0.15, wspace=0.05, hspace=0.25, top=0.925, bottom=0.05)
plt.margins(x=0, y=0)
plt.savefig(figout)
plt.close()


mod.write()
