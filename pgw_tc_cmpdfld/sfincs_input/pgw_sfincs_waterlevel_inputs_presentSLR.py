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

# Sea level rise
slr_df = pd.DataFrame()
dir = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\slr_projections\ipcc_ar6')
for filename in os.listdir(dir):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(dir, filename))
        slr_df = pd.concat([slr_df, df])
        print(filename)
slr_df.reset_index(inplace=True, drop=True)

slr_sub = slr_df[(slr_df['scenario'] == 'ssp585') & (slr_df['confidence'] == 'medium')]
stations = slr_sub['psmsl_id'].unique()
sta_data = pd.DataFrame()
for sta in stations:
    percentiles = slr_sub[slr_sub['psmsl_id'] == sta]['2100']

    # Determine the parameters (minimum and maximum) of the uniform distribution
    a = min(percentiles)
    b = max(percentiles)
    uniform_sample = ss.uniform.rvs(loc=a, scale=b - a, size=1000)

    # Plot the empirical cumulative distribution function (ECDF)
    sorted_data = np.sort(uniform_sample)
    sta_data[f'{sta}'] = sorted_data
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

sta_data.columns = ['Springmaid Pier SC', 'Cape Hatteras NC', 'Beaufort NC', 'Oregon Inlet NC']
sta_data.index = ecdf

# Plot the cum prob for each site
fig, ax = plt.subplots()
plt.plot(sta_data, sta_data.index, lw=2, label=sta_data.columns)
plt.xlabel('Sea Level Rise (m)')
plt.ylabel('Cumulative Probability')
plt.title('IPCC AR6 projections for SSP5-8.5 in 2100 (~4 degC warming)\nrelative to 1995-2014 baseline')
plt.legend()
plt.grid(True)
plt.savefig(fr'Z:\users\lelise\projects\ENC_CompFld\slr_projections\ipcc_ar6\ipcc_ar6_ssp585_medium.png',
            tight_layout=True)
plt.close()

# Sample random and apply to model water level BCs
percentiles = slr_sub[slr_sub['psmsl_id'] == 2295]['2100']
a = min(percentiles)
b = max(percentiles)

# Filepath to data catalogs yml
yml_pgw = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\data_catalog_pgw.yml'
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\present_florence\ensmean' \
       r'\flor_ensmean_present'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_pgw, yml_base])
mod.update(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\tmp')
mod.write()
region = mod.mask.where(mod.mask == 2, 0).raster.vectorize()

n_slr = 5
storms = ['floy', 'matt', 'flor']
slr_event_tracker = []
for storm in storms:
    climates = ['pres']
    runs = ['ensmean', 'ens1', 'ens2', 'ens3', 'ens4', 'ens5', 'ens6', 'ens7']
    # Update model time before writing new boundary conditions
    if storm == 'flor':
        # Florence
        tref = '20180913 000000'
        tstart = tref
        tstop = '20180930 000000'
    elif storm == 'matt':
        # Matthew
        tref = '20161007 000000'
        tstart = tref
        tstop = '20161015 000000'
        runs = ['ensmean', 'ens1', 'ens2', 'ens3', 'ens4', 'ens5', 'ens6']
    elif storm == 'floy':
        # Floyd
        tref = '19990913 000000'
        tstart = '19990914 000000'
        tstop = '19990922 000000'
    
    mod.setup_config(
        **{'crsgeo': mod.crs.to_epsg(),
            "tref": tref,
            "tstart": tstart,
            "tstop": tstop})
    
    slr_scenarios = []
    slr_values = []
    for climate in climates:
        for run in runs:
            # Creating water level inputs from ADCIRC output for pgw runs
            wl_id = f'{storm}_{climate}_{run}_waterlevel'
            wl_df = mod.data_catalog.get_geodataset(data_like=wl_id, geom=region, buffer=500)
            mod.setup_waterlevel_forcing(geodataset=wl_df,
                                         offset='lmsl_to_navd88',  # converts mean sea level to NAVD88 datum SFINCS
                                         timeseries=None, locations=None, buffer=500, merge=False)
            mod.write_forcing(data_vars='bzs')
            # Remove bad data points
            # (Why? there are some points where there is no data in the offset raster available)
            bzs = mod.forcing['bzs']
            index_to_remove = bzs['index'][bzs.argmax().values.item()].values.item()
            cleaned_bcs = bzs.drop_sel(index=index_to_remove)

            counter = 0
            while counter < n_slr:
                slr = ss.uniform.rvs(loc=a, scale=(b - a), size=1)
                event_id = f'{storm}_{climate}Scaled_{run}_SLR{counter+1}'
                slr_scenarios.append(event_id)

                slr_values.append(slr)
                slr_bcs = cleaned_bcs + slr
                mod.setup_waterlevel_forcing(geodataset=slr_bcs, timeseries=None, locations=None, buffer=500,
                                             merge=False)
                mod.write_forcing()  # this creates the sfincs.bnd and sfincs.bzs input files
    
                dir_out = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models', 'waterlevel_slr')
    
                if os.path.exists(dir_out) is False:
                    os.makedirs(dir_out)
                shutil.move(src=os.path.join(mod.root, 'sfincs.bnd'),
                            dst=os.path.join(dir_out, f'{event_id}_waterlevel.bnd'))
                shutil.move(src=os.path.join(mod.root, 'sfincs.bzs'),
                            dst=os.path.join(dir_out, f'{event_id}_waterlevel.bzs'))

                counter += 1

                # # Plot the output for the event to make sure it makes sense
                # dir_out = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\forcing_figs_slr')
                # if os.path.exists(dir_out) is False:
                #     os.makedirs(dir_out)
                # figout = os.path.join(dir_out, f'{event_id}.png')
                # fig, axes = mod.plot_forcing(forcings='bzs')
                # for ax in axes:
                #     ax.legend().remove()
                #     ax.set_title('')
                # plt.subplots_adjust(left=0.12, wspace=0.05, hspace=0.25, top=0.925, bottom=0.15)
                # plt.suptitle(event_id)
                # plt.savefig(figout)
                # plt.close()
    
    slr_events = pd.DataFrame(slr_values)
    slr_events.index = slr_scenarios
    slr_event_tracker.append(slr_events)

event_db = pd.concat(slr_event_tracker)
event_db.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\slr_event_ids.csv', header=False)