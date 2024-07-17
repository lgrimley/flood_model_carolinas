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
import shutil
import matplotlib as mpl
import cartopy.crs as ccrs
import scipy.stats as ss

# Filepath to data catalogs yml
yml_pgw = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\wrf\data_catalog_pgw.yml'
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'
yml = r'Z:\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'

# Working directory and model root
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_wrf\AGU2023\present_florence\ensmean' \
       r'\flor_ensmean_present'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_pgw, yml_base, yml])
mod.update(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\tmp')
mod.write()

''' Update time of reference depending on the storm '''
# Florence
# "tref": "20180913 000000",
# "tstart": "20180913 000000",
# "tstop": "20180930 000000",
# Matthew
# "tref": "20161007 000000",
# "tstart": "20161007 000000",
# "tstop": "20161015 000000",
# Floyd
# "tref": "19990913 000000",
# "tstart": "19990914 000000",
# "tstop": "19990922 000000",

mod.setup_config(
    **{
        'crsgeo': mod.crs.to_epsg(),
        "tref": "19990914 000000",
        "tstart": "19990914 000000",
        "tstop": "19990922 000000",
    }
)

''' Loop through the events and write the SFINCS input files '''
storm = 'floy'  # update for each storm: floy, flor, matt
met_id_rain = 'era5_1999'
met_id_wind = 'era5_1999'
wl_id = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\hindcast\waterlevel\floy_ERA5.nc'
runs = ['pres']
wd = 'sfincs_models_obs'

# Creating meteo inputs from WRF output for pgw runs
for run in runs:
    met_id = f'{storm}_{run}'

    # Set up the SFINCS precip and wind and write to netcdf
    mod.setup_precip_forcing_from_grid(precip=met_id_rain, aggregate=False)
    mod.setup_wind_forcing_from_grid(wind=met_id_wind)
    mod.write_forcing()  # this creates the files precip_2d.nc and wind_2d.nc in the SFINCS model folder

    # Take these newly created files and rename and move them to be associated with the correct event
    variables = ['precip_2d', 'wind_2d']
    for var in variables:
        vout = var.replace('_', '')  # remove the underscore
        # Create a directory for the nc files to be saved in
        dir_out = os.path.join(rf'Z:\users\lelise\projects\ENC_CompFld\Chapter2\{wd}', vout)
        if os.path.exists(dir_out) is False:
            os.makedirs(dir_out)
        shutil.move(src=os.path.join(mod.root, f'{var}.nc'), dst=os.path.join(dir_out, f'{met_id}_{vout}.nc'))

    # Creating water level inputs from ADCIRC output for pgw runs
    region = mod.mask.where(mod.mask == 2, 0).raster.vectorize()
    wl_df = mod.data_catalog.get_geodataset(data_like=wl_id, geom=region, buffer=500)
    mod.setup_waterlevel_forcing(geodataset=wl_df,
                                 offset='lmsl_to_navd88',  # converts mean sea level to NAVD88 datum SFINCS is in
                                 timeseries=None,
                                 locations=None,
                                 buffer=500,
                                 merge=False)
    mod.write_forcing(data_vars='bzs')
    # Remove bad data points
    # (Why? there are some points where there is no data in the offset raster available)
    bzs = mod.forcing['bzs']
    index_to_remove = bzs['index'][bzs.argmax().values.item()].values.item()
    cleaned_bcs = bzs.drop_sel(index=index_to_remove)
    
    #cleaned_bcs = bzs
    mod.setup_waterlevel_forcing(geodataset=cleaned_bcs, timeseries=None, locations=None, buffer=500, merge=False)
    mod.write_forcing()  # this creates the sfincs.bnd and sfincs.bzs input files

    var = 'waterlevel'
    dir_out = os.path.join(rf'Z:\users\lelise\projects\ENC_CompFld\Chapter2\{wd}', var)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)
    shutil.move(src=os.path.join(mod.root, 'sfincs.bnd'), dst=os.path.join(dir_out, f'{met_id}.bnd'))
    shutil.move(src=os.path.join(mod.root, 'sfincs.bzs'), dst=os.path.join(dir_out, f'{met_id}.bzs'))

    mod.setup_discharge_forcing(geodataset='usgs_discharge_floyd',
                                merge=False, buffer=3000)
    mod.write_forcing()

    # Plot the output for the event to make sure it makes sense
    dir_out = os.path.join(rf'Z:\users\lelise\projects\ENC_CompFld\Chapter2\{wd}\forcing_figs')
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)
    figout = os.path.join(dir_out, f'{met_id}.png')
    fig, axes = mod.plot_forcing()
    axes = axes.flatten()
    for ax in axes:
        ax.legend().remove()
    plt.subplots_adjust(left=0.15, wspace=0.05, hspace=0.25, top=0.925, bottom=0.05)
    plt.margins(x=0, y=0)
    plt.suptitle(met_id)
    plt.savefig(figout)
    plt.close()

# fig, axes = mpl.pyplot.subplots()
# era5.plot()
# plt.subplots_adjust(left=0.15, wspace=0.05, hspace=0.25, top=0.925, bottom=0.05)
# plt.margins(x=0, y=0)
# plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_hindcast\forcing_figs\era5_floyd.png')
# plt.close()