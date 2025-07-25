#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from hydromt_sfincs import SfincsModel
import time
import sys
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()


# Filepath to data catalog yml
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\sfincs_base_mod'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_sfincs_Carolinas])
cat = mod.data_catalog
studyarea_gdf = mod.region.to_crs(epsg=32617)

newbern = cat.get_geodataframe(r'Z:\Data-Expansion\users\lelise\projects\NBLL\geospatial\city_limits')
newbern = newbern.to_crs(epsg=32617)

# Connect to the working directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure')

# Read in NC and SC buildings data that was previously clipped
buildings_gdf = pd.read_csv('SFINCS_buildings.csv',  index_col=0, low_memory=True)
gdf = gpd.GeoDataFrame(buildings_gdf, geometry=gpd.points_from_xy(x=buildings_gdf['xcoords'],
                                                                  y=buildings_gdf['ycoords'],
                                                                  crs=32617))
gdf = gdf.clip(newbern)
gdf['FFE_m'] = gdf['FFE'] * 0.3048

# Read in SFINCS MODEL OUTPUTS
rp = 100
data = cat.get_rasterdataset(r'..\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\aep\probabilistic_WSE\ncep_AEP_WSE_compound.nc', crs=32617)
vals = data.sel(return_period=rp).sel(x=gdf['geometry'].x.to_xarray(),
                                      y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'hist_1pct_WSE'] = np.round(vals.transpose(), decimals=3)


data = cat.get_rasterdataset(r'..\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\canesm_ssp585\aep\probabilistic_WSE\canesm_AEP_WSE_compound.nc', crs=32617)
vals = data.sel(return_period=rp).sel(x=gdf['geometry'].x.to_xarray(),
                                      y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'fut_1pct_WSE'] = np.round(vals.transpose(), decimals=3)

gdf_subset =

# FUTURE WATER LEVELS
chunks_size = {'rank':100, 'x': 500, 'y': 500}
sorted_data = cat.get_rasterdataset(r'..\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\canesm_ssp585\aep\sorted_data\canesm_sorted_peakWSE_compound.nc',
                                    chunks=chunks_size, crs=32617, geom=newbern)
n_total = sorted_data.sizes['rank']
n_slice = 100
extreme_slice = slice(n_total - n_slice, n_total)
sorted_data_extreme = sorted_data.isel(rank=extreme_slice).compute()


# Loop through each building and extract the extreme water levels in the future TC set,
# keeping the water levels greater than or equal to the 1pct at the building
x_coords = gdf.geometry.x.values
y_coords = gdf.geometry.y.values
hist_aep_1pct_vals = np.round(gdf['hist_1pct_WSE'].values, 2)
fut_aep_1pct_vals = np.round(gdf['fut_1pct_WSE'].values, 2)

# Store the results
exceeding_n_hist = []
exceeding_n_fut = []
exceeding_vals_hist = []
exceeding_vals_fut = []
flag = []
# Loop through each building point
for x, y, aep_hist, aep_fut in zip(x_coords, y_coords, hist_aep_1pct_vals, fut_aep_1pct_vals):
    # Get the extreme water levels for the location
    vals = np.round(sorted_data_extreme.sel(x=x, y=y, method='nearest').values,2)

    # Filter to keep only values greater than the building's 1% AEP
    exceeding_hist = vals[vals >= aep_hist]
    exceeding_fut = vals[vals >= aep_fut]

    if (len(exceeding_hist) == n_slice) or (len(exceeding_fut) == n_slice):
        print('Number of water levels exceeding is limited by the slice used. Increase n_slice')
        flag.append(1)
    else:
        flag.append(0)

    # Save the number of times the WSE exceeds
    exceeding_n_hist.append(len(exceeding_hist))
    exceeding_n_fut.append(len(exceeding_fut))

    # Save the values
    exceeding_vals_hist.append(np.round(exceeding_hist.tolist(), decimals=2))
    exceeding_vals_fut.append(np.round(exceeding_fut.tolist(), decimals=2))

# Store in new column
gdf['exceed_hist_1pct_n'] = exceeding_n_hist
gdf['exceed_fut_1pct_n'] = exceeding_n_fut
gdf['exceed_hist_1pct_vals'] = exceeding_vals_hist
gdf['exceed_fut_1pct_vals'] = exceeding_vals_fut
gdf['flag'] = flag

