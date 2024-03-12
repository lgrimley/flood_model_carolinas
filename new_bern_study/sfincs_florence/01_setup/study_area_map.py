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

yml = r'Z:\users\lelise\data\data_catalog_LEG.yml'
cat = hydromt.DataCatalog(yml)
os.chdir(r'Z:\users\lelise\projects\NBLL\sfincs\nbll_model_v2')
mod = SfincsModel(root='nbll_40m_sbg3m_v3_eff25', mode='r', data_libs=yml)
mod.read()

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

# Plotting the data on a map with contextual layers
domain = mod.geoms['region']
area_sqkm = domain['geometry'].area / 10 ** 6

major_rivers = cat.get_geodataframe('carolinas_major_rivers')
major_rivers.to_crs(mod.crs, inplace=True)
major_rivers = major_rivers.clip(domain)

# Define the colors you want
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(domain.buffer(2000).total_bounds)[[0, 2, 1, 3]]

''' Model Boundary Conditions Figure '''
bzs = mod.forcing['bzs'].vector.to_gdf()
gdf_msk = utils.get_bounds_vector(mod.grid["msk"])
gdf_msk2 = gdf_msk[gdf_msk["value"] == 2]
gdf_msk3 = gdf_msk[gdf_msk["value"] == 3]

plt_bc_map = True
if plt_bc_map is True:
    fig, axs = plt.subplots(
        nrows=2, ncols=1,
        figsize=(6, 5),
        subplot_kw={'projection': utm},
        tight_layout=True,
        sharex=False, sharey=False,
        gridspec_kw={'height_ratios': [1.5, 1]}
    )

    ax = axs[0]
    # basins.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, hatch='+', zorder=0, alpha=1)
    cmap = mpl.cm.binary
    bounds = np.arange(-5, 25, 5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    dem = mod.grid['dep'].plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=1)

    # Plot background/geography layers
    domain.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1, alpha=0.8, label='Domain')

    gdf_msk3.plot(ax=ax, zorder=4, color='magenta', linewidth=1.25, label='Outflow BC')
    gdf_msk2.plot(ax=ax, zorder=4, color='blue', linewidth=1.5, label='Water Level BC')
    bzs.plot(ax=ax, color='blue', zorder=5, marker='o', markersize=35,
             edgecolor='black', linewidth=0.75, label='WL BC Gage')

    for label, grow in bzs.iterrows():
        x, y = grow.geometry.x, grow.geometry.y
        if label == 2:
            ann_kwargs = dict(
                xytext=(-8, -12),
                textcoords="offset points",
                zorder=4,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="skyblue", alpha=1),
                    patheffects.Normal(),
                ],
            )
        else:
            ann_kwargs = dict(
                xytext=(-10, 0),
                textcoords="offset points",
                zorder=4,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="skyblue", alpha=1),
                    patheffects.Normal(),
                ],
            )
        ax.annotate(f'{label}', xy=(x, y), **ann_kwargs)

    minx, miny, maxx, maxy = domain.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    cbar = fig.colorbar(dem,
                        ax=ax,
                        shrink=0.6,
                        extend='both',
                        spacing='uniform',
                        label='Elevation\n(m+NAVD88)',
                        location='right',
                        anchor=(-2, 0)
                        )
    legend_kwargs0 = dict(
        bbox_to_anchor=(1.01, 1.05),
        title=None,
        loc="upper left",
        frameon=True,
        prop=dict(size=10),
    )
    ax.legend(**legend_kwargs0)

    # Add title and save figure
    ax.set_extent(extent, crs=utm)
    ax.set_title('')
    ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
    ax.yaxis.set_visible(True)

    ax.set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
    ax.xaxis.set_visible(True)
    ax.ticklabel_format(axis='both', style='sci', useOffset=False)
    # plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    ax = axs[1]

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    bnd_type = 'bzs'
    da = mod.forcing[bnd_type]
    df = da.to_pandas().transpose()
    df.index = mdates.date2num(df.index)
    df.plot(ax=ax,
            label=False,
            legend=False)
    ax.set_ylim((-0.25, 14.25))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylabel('m +NAVD88')
    ax.xaxis.set_visible(True)
    ax.set_title('')
    ax.set_title('Florence Water Level BC Inputs', loc='left', fontsize=10)
    legend_kwargs0 = dict(
        ncol=1,
        title="Legend",
        bbox_to_anchor=(1.01, 1.1),
        loc="upper left",
        frameon=True,
        prop=dict(size=10)
    )
    ax.legend(**legend_kwargs0)

    for ax in axs:
        ax.set_anchor('W')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(mod.root, 'figs', 'florence_modelsetup.png'), bbox_inches='tight',
                dpi=255)
    plt.close()

''' Model Meteo Boundary Conditions Figure '''
# Load and calculate total precipitation
precip = mod.forcing['precip_2d']
precip_total = precip.sum(dim='time')

plt_bc_map = True
if plt_bc_map is True:
    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(5, 5),
        subplot_kw={'projection': utm},
        tight_layout=True)

    # Precipitation!!!
    cmap = mpl.cm.jet
    bounds = np.arange(start=200, stop=900, step=100)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    precip_total.plot(ax=ax,
                      cmap=cmap,
                      norm=norm,
                      add_colorbar=False,
                      zorder=0,
                      alpha=0.8,
                      )
    # Plot background/geography layers
    major_rivers.plot(ax=ax, color='darkblue', edgecolor='none', linewidth=1,
                      linestyle='-', zorder=1, alpha=1)
    domain.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5,
                linestyle='-', zorder=2, alpha=1, label='HUC6 Basins')

    minx, miny, maxx, maxy = domain.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm,
                 ax=ax,
                 shrink=0.5,
                 extend='max',
                 spacing='uniform',
                 label='(mm)')

    ax.set_extent(extent, crs=utm)
    ax.set_title('')
    # ax.set_title('Cumulative Precipitation', loc='left')
    ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
    ax.yaxis.set_visible(True)
    ax.set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
    ax.xaxis.set_visible(True)
    ax.ticklabel_format(style='sci', useOffset=False)
    ax.set_aspect('equal')

    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(mod.root, 'figs', 'meteo.png'), bbox_inches='tight',
                dpi=255)
    plt.close()
