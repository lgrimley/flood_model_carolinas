import os
import glob
import hydromt
import rioxarray
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
from shapely.geometry import box

# Filepath to data catalog yml
yml = os.path.join(r'Z:\users\lelise\data\data_catalog.yml')
cat = hydromt.DataCatalog(yml)

os.chdir(r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2\design_storms')
mod = SfincsModel(root=r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2\nbll_40m_sbg3m_v3_eff25',
                  mode='r', data_libs=yml)
mod.read(epsg=32618)

scen = ['P2yr', 'P5yr', 'P10yr', 'P25yr', 'P50yr', 'P100yr', 'P500yr',
        # 'F2yr', 'F5yr', 'F10yr', 'F25yr', 'F50yr', 'F100yr', 'F500yr'
        ]

hmax_list = []
tmax_list = []
for rp in scen:
    filepath = os.path.join(os.getcwd(), f'nbll_40m_sbg3m_v3_eff25_{rp}', 'gis', f'nbll_{rp}_max_depth_gridRes40m.tif')
    hmax = cat.get_rasterdataset(filepath)
    hmax_list.append(hmax)

    filepath = os.path.join(os.getcwd(), f'nbll_40m_sbg3m_v3_eff25_{rp}', 'gis', f'nbll_{rp}_tmax_hours_gridRes40m.tif')
    tmax = cat.get_rasterdataset(filepath)
    tmax_list.append(hmax)


da = xr.concat(hmax_list, dim='run')
da['run'] = xr.IndexVariable('run', scen)

dat = xr.concat(tmax_list, dim='run')
dat['run'] = xr.IndexVariable('run', scen)

# Plotting
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)

cmap = mpl.cm.binary
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(1000).total_bounds)[[0, 2, 1, 3]]
#city_limits = cat.get_geodataframe(r'Z:\users\lelise\projects\NBLL\geospatial\city_limits\city_limits.shp').to_crs(mod.crs)
#extent = np.array(city_limits.buffer(1000).total_bounds)[[0, 2, 1, 3]]
plt_fig_hmax = True
if plt_fig_hmax is True:
    dep = mod.grid['dep']
    fig, axs = plt.subplots(
        nrows=4, ncols=2,
        figsize=(5, 8),
        subplot_kw={'projection': utm},
        tight_layout=True,
        layout='constrained',
        sharex=True, sharey=True)
    axs = axs.flatten()
    axs[7].set_visible(False)
    for i in range(len(scen)):
        ckwargs = dict(cmap='Blues', vmin=0.15, vmax=8)
        hmax = da.sel(run=scen[i])
        cs = hmax.plot(ax=axs[i],
                       add_colorbar=False,
                       zorder=2,
                       alpha=1,
                       **ckwargs)
        # Plot background/geography layers
        mod.region.plot(ax=axs[i], color='grey', edgecolor='none', zorder=1, alpha=1)
        mod.region.plot(ax=axs[i], color='none', edgecolor='black', linewidth=0.5,
                        linestyle='-', zorder=3, alpha=1)

        minx, miny, maxx, maxy = mod.region.total_bounds
        axs[i].set_xlim(minx, maxx)
        axs[i].set_ylim(miny, maxy)
        axs[i].set_extent(extent, crs=utm)
        axs[i].set_title('')
        axs[i].set_title((scen[i]), loc='Center')
        if i in [0, 2, 4, 6]:
            axs[i].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
        if i in [5, 6]:
            axs[i].set_xlabel(f"X Coord UTM {utm_zone} (m)")
            axs[i].xaxis.set_visible(True)

        axs[i].ticklabel_format(style='sci', useOffset=False)
        axs[i].set_aspect('equal')

    # Colorbar
    pos1 = axs[7].get_position()  # get the original position
    cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 1, 0.02, pos1.height * 1.5])
    cb = fig.colorbar(cs,
                      cax=cbar_ax,
                      shrink=0.7,
                      extend='max',
                      spacing='uniform',
                      label='Max Water Depth (m)',
                      pad=0,
                      aspect=30
                      )
    # Save and close plot
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.margins(x=0, y=0)
    plt.savefig('design_maps.png', dpi=225, bbox_inches="tight")
    plt.close()

plt_fig_tmax = True
if plt_fig_tmax is True:
    dep = mod.grid['dep']
    fig, axs = plt.subplots(
        nrows=4, ncols=2,
        figsize=(5, 8),
        subplot_kw={'projection': utm},
        tight_layout=True,
        layout='constrained',
        sharex=True, sharey=True)
    axs = axs.flatten()
    axs[7].set_visible(False)
    for i in range(len(scen)):
        ckwargs = dict(cmap='jet', vmin=0.5, vmax=8.5)
        tmax = dat.sel(run=scen[i])
        cs = tmax.plot(ax=axs[i],
                       add_colorbar=False,
                       zorder=2,
                       alpha=1,
                       **ckwargs)
        # Plot background/geography layers
        mod.region.plot(ax=axs[i], color='lightgrey', edgecolor='none', zorder=1, alpha=0.8)
        mod.region.plot(ax=axs[i], color='none', edgecolor='black', linewidth=0.5,
                        linestyle='-', zorder=3, alpha=1)

        minx, miny, maxx, maxy = mod.region.total_bounds
        axs[i].set_xlim(minx, maxx)
        axs[i].set_ylim(miny, maxy)
        axs[i].set_extent(extent, crs=utm)
        axs[i].set_title('')
        axs[i].set_title((scen[i]), loc='Center')
        if i in [0, 2, 4, 6]:
            axs[i].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
        if i in [5, 6]:
            axs[i].set_xlabel(f"X Coord UTM {utm_zone} (m)")
            axs[i].xaxis.set_visible(True)

        axs[i].ticklabel_format(style='sci', useOffset=False)
        axs[i].set_aspect('equal')

    # Colorbar
    pos1 = axs[7].get_position()  # get the original position
    cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 1, 0.02, pos1.height * 1.5])
    cb = fig.colorbar(cs,
                      cax=cbar_ax,
                      shrink=0.7,
                      extend='max',
                      spacing='uniform',
                      label='Time of Inundation (hours)',
                      pad=0,
                      aspect=30
                      )
    # Save and close plot
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.margins(x=0, y=0)
    plt.savefig('design_storm_tmax.png', dpi=225, bbox_inches="tight")
    plt.close()


# ''' Plotting Florence '''
# plt_floodmap_indvidiual = True
# if plt_floodmap_indvidiual is True:
#     cmap = mpl.cm.binary
#     wkt = mod.grid['dep'].raster.crs.to_wkt()
#     utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
#     utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
#     extent = np.array(mod.region.buffer(1000).total_bounds)[[0, 2, 1, 3]]
#
#     fig, ax = plt.subplots(
#         nrows=1, ncols=1,
#         figsize=(5, 5),
#         subplot_kw={'projection': utm},
#         tight_layout=True)
#
#     ckwargs = dict(cmap='Blues', vmin=hmin, vmax=10)
#     cs = hmax.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=2)
#
#     # Plot background/geography layers
#     mod.region.plot(ax=ax, color='grey', edgecolor='none', zorder=1, alpha=0.7)
#     mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=1.15, zorder=2, alpha=1)
#     dams.plot(ax=ax, color='black', edgecolor='none', linewidth=1.5, linestyle='-', zorder=3, alpha=1)
#     dams.plot(ax=ax, color='orange', edgecolor='none', linewidth=1.25, linestyle='-', zorder=3, alpha=1, label='Dams')
#
#     minx, miny, maxx, maxy = mod.region.total_bounds
#     ax.set_xlim(minx, maxx)
#     ax.set_ylim(miny, maxy)
#
#     cbar = fig.colorbar(cs,
#                         ax=ax,
#                         shrink=0.5,
#                         extend='max',
#                         orientation='vertical',
#                         label='Max Depth (m+NAVD88)',
#                         anchor=(-0.1, 0.15))
#
#     legend_kwargs0 = dict(
#         bbox_to_anchor=(1.0, 0.9),
#         title="Legend",
#         loc="upper left",
#         frameon=True,
#         prop=dict(size=8),
#     )
#     ax.legend(**legend_kwargs0)
#
#     # Add title and save figure
#     ax.set_extent(extent, crs=utm)
#     ax.set_title('')
#     ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
#     ax.yaxis.set_visible(True)
#     ax.set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
#     ax.xaxis.set_visible(True)
#     ax.ticklabel_format(style='sci', useOffset=False)
#     ax.set_aspect('equal')
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.margins(x=0, y=0)
#
#     plt.savefig(
#         r'Z:\users\lelise\projects\HCFCD\sfincs_models\03_for_TAMU\Harvey\hcfcd_100m_sbg5m_v2_ksat75\harvey.png',
#         dpi=225, bbox_inches="tight")
#     plt.close()
