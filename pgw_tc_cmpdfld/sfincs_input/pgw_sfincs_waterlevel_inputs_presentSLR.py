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
yml_pgw = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\data_catalog_pgw.yml'
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'

# Working directory and model root
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\present_matthew\ensmean\matt_ensmean_present'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_pgw, yml_base])
mod.update(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\tmp')
mod.write()

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
randsamp = ss.uniform.rvs(loc=a, scale=(b - a), size=10)

dir_out = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\waterlevel')
for filename in os.listdir(dir_out):
    storm, climate, run, variable = filename.split('_')
    if climate == 'pres':
        multiplier = sf_df.loc[f'{storm}_{run}'][0]
        da = mod.data_catalog.get_rasterdataset(os.path.join(dir_out, filename))
        da_scaled = da * multiplier
        fileout = filename.replace('pres', 'presScaled')
        # da_scaled.to_netcdf(os.path.join(dir_out, fileout))
