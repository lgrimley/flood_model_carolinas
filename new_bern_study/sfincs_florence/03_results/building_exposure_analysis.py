#!/usr/bin/env python
# coding: utf-8

import os
import re
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import fiona
import matplotlib as mpl
import hydromt
import rioxarray
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel
from matplotlib.ticker import MultipleLocator

# Load the model the results
cat = hydromt.DataCatalog(r'Z:\users\lelise\data\data_catalog.yml')
model_root = r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2\nbll_40m_sbg3m_v3_eff25'
mod = SfincsModel(root=model_root, mode='r')

# Setup output directory
out_dir = model_root  # os.path.join(model_root, 'scenarios', '00_driver_analysis')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

''' PART 1 - Determine damage status using NFIP claims/policy data at each structure'''
# Read in area of interest shapefile and project
studyarea_gdf = mod.region

# Read in structures information and clip to the study area
nc_buildings = gpd.read_file(r'Z:\users\lelise\geospatial\flood_damage\included_data.gdb',
                             layer='buildings',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
nc_buildings['STATE'] = 'NC'
buildings = nc_buildings.drop(nc_buildings.columns[~nc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of NC Buildings in Study Area:', str(len(buildings)))

''' PART 2 - Calculate flood depth at buildings'''
fld_map_key = ['P10yr', 'P100yr', 'florence']
gdf = buildings
sbg_res = 3
hmin = 0.15
fld_area = []
stats_hmax = []
dir = r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2\model_outputs'
for key in fld_map_key:
    print('Calculating info for flood scenario:', key)
    # Load depth raster for model scenario
    filepath = os.path.join(dir, f'nbll_{key}_max_depth_gridRes40m_sbg3m.tif')
    fldpth_da = cat.get_rasterdataset(data_like=filepath)

    # Extract depth at building centroids
    hmax = fldpth_da.sel(x=gdf['geometry'].x.to_xarray(),
                         y=gdf['geometry'].y.to_xarray(),
                         method='nearest').values
    gdf[key] = hmax.transpose()
    # print(gdf[key].describe())
    # stats_hmax.append(gdf[key].describe())

    # Calculate flood extent using depth raster
    flooded_cells_mask = (fldpth_da >= hmin)
    flooded_cells_count = np.count_nonzero(flooded_cells_mask)
    flooded_area = flooded_cells_count * (sbg_res * sbg_res) / (1000 ** 2)
    print(key, ' flooded sbg area (sq.km):', round(flooded_area, 2))
    fld_area.append(flooded_area)

# Clip to city limits and get time of inundation
nb_citylimits = cat.get_geodataframe(r'Z:\users\lelise\projects\NBLL\geospatial\city_limits\city_limits.shp').to_crs(
    mod.crs)
gdf = gdf.clip(nb_citylimits)
# drop buildings that don't have any flooding across the scenarios
gdf1 = gdf.dropna(subset=fld_map_key, how='all')

stats_tmax = []
for key in fld_map_key:
    gdf1[key] = np.where((gdf1[key] <= hmin), np.nan, gdf1[key])

    # Load time inundation raster for model scenario
    filepath = os.path.join(dir, f'nbll_{key}_tmax_hours_gridRes40m.tif')
    fldtime_da = cat.get_rasterdataset(data_like=filepath)

    # Extract depth at building centroids
    tmax = fldtime_da.sel(x=gdf1['geometry'].x.to_xarray(),
                          y=gdf1['geometry'].y.to_xarray(),
                          method='nearest').values
    out_key = (key + '_tmax')
    gdf1[out_key] = tmax.transpose()
    gdf1[out_key] = np.where(gdf1[key].isna(), np.nan, gdf1[out_key])
    print(gdf1[out_key].describe())
    stats_tmax.append(gdf1[out_key].describe())

gdf2 = gdf1.dropna(subset=fld_map_key, how='all')
gdf2.to_file(os.path.join(dir, 'flooded_buildings.shp'))
des = pd.DataFrame(gdf2.describe())

# Plotting
font = {'family': 'Arial',
        'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

props = dict(boxes="white", whiskers="black", caps="black")
boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
flierprops = dict(marker='+', markerfacecolor='none', markersize=3, markeredgecolor='black')
medianprops = dict(linestyle='-', linewidth=2, color='black')
meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=4)

###########################
plot_depth = True
if plot_depth is True:
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(5, 2.5))

    nc_n = [gdf2['P10yr'].count(), gdf2['P100yr'].count(), gdf2['florence'].count()]
    bp = gdf2[fld_map_key].plot.box(ax=ax,
                                    vert=False,
                                    color=props,
                                    boxprops=boxprops,
                                    flierprops=flierprops,
                                    medianprops=medianprops,
                                    meanprops=meanpointprops,
                                    meanline=False,
                                    showmeans=True,
                                    patch_artist=True)
    ax.set_xlabel('Water Depth >0.15m at Buildings (m)')
    ax.set_yticklabels([f'10yr\n(n={nc_n[0]})', f'100yr\n(n={nc_n[1]})', f'Florence\n(n={nc_n[2]})'])
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.8)
    ax.grid(visible=True, which='major', axis='x', **kwargs)
    kwargs = dict(linestyle='--', linewidth=0.75, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='minor', axis='x', **kwargs)
    # ax.set_xscale("log")
    ax.set_xlim(0, 6)
    pos1 = ax.get_position()  # get the original position

    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(dir, 'building_depths_newberncitylimits_nolog.png'),
                dpi=225,
                bbox_inches="tight")
    plt.close()

plot_tmax = True
if plot_tmax is True:
    # Plotting
    fig, axs = plt.subplots(nrows=2, ncols=1, tight_layout=True, figsize=(5, 3), sharey=False, sharex=False)
    nc = gdf2.dropna(axis='index', how='all')
    nc['florence_tmax_day'] = nc['florence_tmax'] / 24

    ax = axs[0]
    bp = nc['florence_tmax_day'].plot.box(ax=ax,
                                          vert=False,
                                          color=props,
                                          boxprops=boxprops,
                                          flierprops=flierprops,
                                          medianprops=medianprops,
                                          meanprops=meanpointprops,
                                          meanline=False,
                                          showmeans=True,
                                          patch_artist=True)
    ax.set_xlim(0, 25)
    ax.set_xlabel('Duration of Water Depths >0.15m at Buildings (days)')
    ax.set_yticklabels([f'Florence\n(n={nc_n[2]})'])
    kwargs = dict(linestyle='-', linewidth=1, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='major', axis='x', **kwargs)
    kwargs = dict(linestyle='--', linewidth=0.75, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='minor', axis='x', **kwargs)
    pos1 = ax.get_position()  # get the original position

    ax = axs[1]
    subkey = ['P10yr_tmax', 'P100yr_tmax']
    bp = nc[subkey].plot.box(ax=ax,
                             vert=False,
                             color=props,
                             boxprops=boxprops,
                             flierprops=flierprops,
                             medianprops=medianprops,
                             meanprops=meanpointprops,
                             meanline=False,
                             showmeans=True,
                             patch_artist=True)
    ax.set_xlabel('Duration of Water Depths >0.15m at Buildings (hr)')
    ax.set_yticklabels([f'10yr\n(n={nc_n[0]})', f'100yr\n(n={nc_n[1]})'])
    kwargs = dict(linestyle='-', linewidth=1, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='major', axis='x', **kwargs)
    kwargs = dict(linestyle='--', linewidth=0.75, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='minor', axis='x', **kwargs)
    ax.set_xlim(0, 25)
    pos1 = ax.get_position()  # get the original position

    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(dir, 'building_time_of_inundation_newberncitylimits.png'),
                dpi=225,
                bbox_inches="tight")
    plt.close()
