#!/usr/bin/env python3
# coding: utf-8
#
# -------------- Script Overview -------------------
#
# Mapping the difference in water levels at locations
#
#   Author: Lauren Grimley, lauren.grimley@unc.edu
#   Last edited by: LEG 2/6/24
#
# Inputs:
#   Netcdf of water level timeseries (format compatible with hydromt)
#     * to convert ADCIRC fort.61.nc use script: adcirc_fort61_2netcdf.py
#
# Outputs:
#   Figure of water level difference stats at points
#
# Dependencies:
#
# Needed updates:
#   Generalize index selection (e.g., shapefile or SFINCS mask, none)
#   Model columns and naming lines 59-79 

import os
from os.path import join
import glob
import datetime

import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils

import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio.merge

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from string import ascii_lowercase as abcd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs

# Filepath to data catalog yml
cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\waterlevel')

mod_root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\flor_ensmean_present'
mod = SfincsModel(root=mod_root, mode='r', data_libs=yml)
mod.read()
gdf_msk = utils.get_bounds_vector(mod.grid["msk"])
gdf_msk2 = gdf_msk[gdf_msk["value"] == 2]
da1 = cat.get_geodataset('adcirc_wl_wrf_matt_ensmean_pgw_0.13m.nc', geom=gdf_msk2, buffer=100)
da2 = cat.get_geodataset('adcirc_wl_wrf_matt_ensmean_pgw_1.10m.nc', geom=gdf_msk2, buffer=100)

i = 0
for ind in da1.index.values:
    df1 = da1.sel(index=ind).to_dataframe()
    df2 = da2.sel(index=ind).to_dataframe()
    coords = [df1['x'].unique().item(), df1['y'].unique().item()]
    df_comb = pd.concat([df1['waterlevel'], df2['waterlevel']], axis=1)
    df_comb.columns = ['flor_13cm', 'flor_45cm']
    df_comb['diff'] = df_comb['flor_45cm'] - df_comb['flor_13cm']
    stat = df_comb['diff'].describe().to_frame().T
    stat['index'] = ind
    stat['x'] = coords[0]
    stat['y'] = coords[1]
    if i == 0:
        stat_df = stat
    else:
        stat_df = pd.concat([stat_df, stat], ignore_index=True, axis=0)
    i += 1
    print(i, 'out of', len(da1.index.values))

# Plotting the data on a map with contextual layers
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

domain = mod.geoms['region']
cmap = mpl.cm.binary
bounds = [-5, 0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(gdf_msk2.buffer(10000).total_bounds)[[0, 2, 1, 3]]

col = ['mean', 'std', '50%', 'max']
label = ['Mean', 'STD', '50%', 'Max']
unit = ['(m)', '(m)', '(m)', 'm']
color_map = ['bwr', 'Reds', 'bwr', 'Reds']
n_bins_ranges = [[-0.003, 0.003], [0.01, 0.05], [-0.003, 0.003], [0.1, 0.3]]
#[[-0.002, 0.002], [0.01, 0.03], [-0.002, 0.002], [0.05, 0.3]]

figname= 'WRF PGW Matthew MSL 1.10-0.13'

stat_gdf = gpd.GeoDataFrame(stat_df,
                            geometry=gpd.points_from_xy(x=stat_df['x'], y=stat_df['y'], crs=4326)).to_crs(mod.crs)
stat_gdf_cpy = stat_gdf.copy()
msl_diff = 1.10-0.13
for c in ['mean', '50%', 'max']:
    stat_gdf[c] = stat_gdf[c] - msl_diff

plt_bc_map = True
if plt_bc_map is True:
    fig, axs = plt.subplots(
        nrows=2, ncols=2,
        figsize=(6, 4.5),
        subplot_kw={'projection': utm},
        tight_layout=True, sharey=True,
        sharex=True)
    axs = axs.flatten()

    for i in range(len(col)):
        dem = mod.grid['dep'].plot(ax=axs[i], cmap=cmap, norm=norm, add_colorbar=False, zorder=1)

        vmin = n_bins_ranges[i][0]
        vmax = n_bins_ranges[i][1]
        stat_gdf.plot(column=col[i],
                      cmap=color_map[i],
                      legend=False,
                      vmin=vmin, vmax=vmax,
                      ax=axs[i],
                      markersize=15,
                      edgecolor='black',
                      linewidth=0.25,
                      zorder=3
                      )

        minx, miny, maxx, maxy = extent
        axs[i].set_xlim(minx, maxx)
        axs[i].set_ylim(miny, maxy)

        sm = plt.cm.ScalarMappable(cmap=color_map[i],
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm,
                     ax=axs[i],
                     shrink=0.7,
                     label=unit[i],
                     extend='both',
                     spacing='uniform'
                     )

        # Add title and save figure
        axs[i].set_extent(extent, crs=utm)
        axs[i].set_title('')
        axs[i].set_title(label[i], loc='left')
        axs[i].set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
        axs[i].yaxis.set_visible(False)

        axs[i].set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
        axs[i].xaxis.set_visible(False)

        axs[i].ticklabel_format(axis='both', style='sci', useOffset=False)

    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    fig.suptitle(figname, fontsize=10, fontweight="bold")
    plt.savefig(os.path.join(os.getcwd(), (figname+'.png')), bbox_inches='tight', dpi=255)
    plt.close()
