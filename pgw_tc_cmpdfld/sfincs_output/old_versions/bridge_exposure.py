#!/usr/bin/env python
# coding: utf-8

import os
import glob
import hydromt
from hydromt import DataCatalog
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
from hydromt_sfincs import SfincsModel, utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors, patheffects
from string import ascii_lowercase as abcd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

# Load the model the results
cat = hydromt.DataCatalog(r'Z:\users\lelise\data\data_catalog.yml')
os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\00_analysis\floodmaps2')

# Setup output directory
out_dir = os.path.join(os.getcwd(), '00_analysis', 'floodmaps')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

mod_root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\present_florence\ensmean\flor_ensmean_present'
mod = SfincsModel(root=mod_root, mode='r')

bridges = gpd.read_file(r'Z:\users\lelise\geospatial\infrastructure\bridges\NCDOT_Structures_'
                        r'\NCDOT_Structures_utm.shp').to_crs(32617).clip(mod.region)

scenarios = [
    'flor_ensmean_present',
    'future_florence_ensmean',
    'flor_ensmean_future_45cm',
    'matt_ensmean_present',
    'future_matthew_ensmean',
    'matt_ensmean_future_45cm',
]

for scen in scenarios:
    hmax = cat.get_rasterdataset(f'floodmap_grid_0.1_hmin_{scen}.tif')
    hmax_bridges = hmax.sel(x=bridges['geometry'].x.to_xarray(),
                            y=bridges['geometry'].y.to_xarray(),
                            method='nearest').values
    bridges[scen] = hmax_bridges.transpose()

bridges_flooded = bridges.dropna(axis=0, how='all', subset=scenarios)

scenario_n = [bridges_flooded[scenarios[0]].count(),
              bridges_flooded[scenarios[1]].count(),
              bridges_flooded[scenarios[2]].count(),
              bridges_flooded[scenarios[3]].count(),
              bridges_flooded[scenarios[4]].count(),
              bridges_flooded[scenarios[5]].count()]
# scenario_n.columns = scenarios

hmax = cat.get_rasterdataset(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\2018_Florence\mod_v6_paper'
                             r'\flo_hindcast_v6_200m_LPD2m_avgN\scenarios\00_driver_analysis\floodmap_sbg_0'
                             r'.05_hmin_compound.tif')
hmax_bridges = hmax.sel(x=bridges['geometry'].x.to_xarray(),
                        y=bridges['geometry'].y.to_xarray(),
                        method='nearest').values
bridges['hindcast'] = hmax_bridges.transpose()
bridges_flooded = bridges.dropna(axis=0, how='all', subset='hindcast')
bf = bridges_flooded[bridges_flooded['hindcast'] >= 2]
#############################
cmap = mpl.cm.binary
bounds = [0, 15, 30, 45, 60, 120, 200]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]

l_gdf = cat.get_geodataframe(
    r'Z:\users\lelise\geospatial\infrastructure\tl_2019_us_primaryroads\tl_2019_us_primaryroads.shp')
#l_gdf = l_gdf[l_gdf['FULLNAME'].isin(['I- 95'])]
l_gdf.to_crs(epsg=32617, inplace=True)
roads = l_gdf.clip(mod.region.total_bounds)

l_gdf = cat.get_geodataframe(
    r'Z:\users\lelise\geospatial\infrastructure\tl_2019_37_prisecroads\tl_2019_37_prisecroads.shp')
#l_gdf = l_gdf[l_gdf['FULLNAME'].isin(['I- 95'])]
l_gdf.to_crs(epsg=32617, inplace=True)
roads2 = l_gdf.clip(mod.region)

''' Study Area Figure '''
plt_map = True
if plt_map is True:
    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(8, 5),
        subplot_kw={'projection': utm},
        tight_layout=True)
    dem = mod.grid['dep'].plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=0)
    #mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
    roads2.plot(ax=ax, color='black', edgecolor='black', linewidth=0.75, zorder=1, alpha=1)

    bf_vmax = 6
    bf_vmin = 2
    bf.plot(column='hindcast',
            cmap='hot_r',
            legend=False,
            vmin=vmin, vmax=vmax,
            ax=ax,
            markersize=10,
            edgecolor='black',
            linewidth=0.5,
            zorder=3)

    #  Setup colorbar and add to plot
    sm = plt.cm.ScalarMappable(cmap='hot_r', norm=plt.Normalize(vmin=bf_vmin, vmax=bf_vmax))
    fig.colorbar(sm,
                 ax=ax,
                 shrink=0.7,
                 label='Depth (m)',
                 extend='max',
                 spacing='uniform')

    minx, miny, maxx, maxy = roads2.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Add title and save figure
    ax.set_extent(extent, crs=utm)
    ax.set_title('')
    ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
    ax.yaxis.set_visible(True)
    ax.set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
    ax.xaxis.set_visible(True)
    ax.ticklabel_format(style='sci', useOffset=False)
    ax.set_aspect('equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(
        r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\2018_Florence\mod_v6_paper'
        r'\flo_hindcast_v6_200m_LPD2m_avgN\scenarios\00_driver_analysis\test2.png',
        bbox_inches='tight',
        dpi=255)
    plt.close()
##############################
box_plot_depth = True
if box_plot_depth is True:
    # Plotting
    font = {'family': 'Arial',
            'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})

    props = dict(boxes="white", whiskers="black", caps="black")
    boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
    flierprops = dict(marker='+', markerfacecolor='none', markersize=3,
                      markeredgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    meanpointprops = dict(marker='D',
                          markeredgecolor='black',
                          markerfacecolor='lightgrey',
                          markersize=4)
    ##############################
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           tight_layout=True,
                           figsize=(5, 4)
                           )
    dp = bridges_flooded['hindcast']
    _ = dp.plot.box(ax=ax,
                    vert=False,
                    color=props,
                    boxprops=boxprops,
                    flierprops=flierprops,
                    medianprops=medianprops,
                    meanprops=meanpointprops,
                    meanline=False,
                    showmeans=True,
                    patch_artist=True)

    ax.set_xlabel('Depth (m+NAVD88)')
    # ax.set_yticklabels([f'Present Florence\n(n={scenario_n[0]})',
    #                     f'Future Florence\n(n={scenario_n[1]})',
    #                     f'Future Florence\nMSL 45cm\n(n={scenario_n[2]})',
    #                     f'Present Matthew\n(n={scenario_n[3]})',
    #                     f'Future Matthew\n(n={scenario_n[4]})',
    #                     f'Future Matthew\nMSL 45cm\n(n={scenario_n[5]})'
    #                     ])
    ax.set_title(f'Water Depth at Bridge Locations', loc='left')
    kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.9)
    ax.grid(visible=True, which='major', axis='x', **kwargs)
    kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.9)
    ax.grid(visible=True, which='minor', axis='x', **kwargs)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\2018_Florence\mod_v6_paper'
                             r'\flo_hindcast_v6_200m_LPD2m_avgN\scenarios\00_driver_analysis\bridge_exposure.png'),
                dpi=225,
                bbox_inches="tight")
    plt.close()
