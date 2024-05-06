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
from matplotlib.ticker import FormatStrFormatter


# Filepath to data catalogs yml
yml_pgw = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\data_catalog_pgw.yml'
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_pgw, yml_base])

# Directory to output stuff
out_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis'

# Get peak water levels across all model runs
da_zsmax, event_ids = get_zsmax_da(mod_results_dir=results_dir)


# Plotting compound flood locations - frequency
# Map setup
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
# extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]


storm = ['flor', 'floy', 'matt']
climate = 'fut'
extent = [0.8075 * 10 ** 6, 0.905 * 10 ** 6, 3.825 * 10 ** 6, 3.92 * 10 ** 6]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6, 9),
                         subplot_kw={'projection': utm},
                         sharex=True, sharey=True,
                         tight_layout=True)
axes = axes.flatten()
counter = 0
for storm in storms:
    nruns = 8  # Florence and Floyd have 7 ensemble members
    if storm == 'matt':
        nruns = 7  # Matthew only has 6 ensemble members
    sel_runs = [f'{storm}_{climate}_ens{i}' for i in np.arange(1, nruns, 1)]
    ensmean_runs = [1]
    if climate == 'presScaled':
        slr_runs = []
        for run in sel_runs:
            # Loop through 5 different SLR scenarios for each present Scaled run
            slr_runs = slr_runs + [f'{run}_SLR{i}' for i in np.arange(1, 6, 1)]
        sel_runs = slr_runs
        ds = fld_da_compound.sel(run=sel_runs).sum(dim='run')
        ds_plot = ds.where(ds > 0)
        ensmean_runs = [f'{storm}_{climate}_ensmean_SLR{i}' for i in np.arange(1, 6, 1)]
        ds2 = fld_da_compound.sel(run=ensmean_runs).sum(dim='run')
        ds_plot2 = ds2.where(ds2 > 0)
    else:
        ds = fld_da_compound.sel(run=sel_runs).sum(dim='run')
        ds_plot = ds.where(ds > 0)

        ds2 = fld_da_compound.sel(run=f'{storm}_{climate}_ensmean')
        ds_plot2 = ds2.where(ds2 > 0)
        
    ax = axes[counter]
    ckwargs = dict(cmap='PuRd', vmin=0, vmax=len(sel_runs))
    cs = ds_plot.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=0)
    label = 'Freq of Compound'
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.15, pos0.y0 + pos0.height * 0.1, 0.025, pos0.height * 0.8])
    cbar = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='neither')
    ax.set_title(f'{storm} {climate}\n(n={len(sel_runs)})')

    # Plot ensmean
    ax = axes[counter + 1]
    ckwargs = dict(cmap='PuRd', vmin=0, vmax=len(ensmean_runs))
    cs2 = ds_plot2.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=0)
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.15, pos0.y0 + pos0.height * 0.1, 0.025, pos0.height * 0.8])
    cbar2 = fig.colorbar(cs2, cax=cax, orientation='vertical', label=label, extend='neither')
    ax.set_title(f'{storm} {climate} ensemble mean\n(n={len(ensmean_runs)}) ')

    counter += 2

ii = 0
for ax in axes:
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
    if ii in [0, 2, 4]:
        ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
        ax.yaxis.set_visible(True)
        ax.xaxis.set_visible(False)
        ax.ticklabel_format(style='sci', useOffset=False)
    if ii in [4, 5]:
        ax.set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
        ax.xaxis.set_visible(True)
        ax.ticklabel_format(style='sci', useOffset=False)

    ax.set_aspect('equal')
    # ax.set_axis_off()
    ax.set_extent(extent, crs=utm)
    ii += 1
#plt.subplots_adjust(wspace=0, hspace=0)
#plt.margins(x=0, y=0)
plt.savefig(
    fr'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\{climate}_compound_flood_freq_zoom.png',
    dpi=225, bbox_inches="tight")
plt.close()

# Plotting extent difference

storm = ['flor', 'floy', 'matt']
extent = [0.8075 * 10 ** 6, 0.905 * 10 ** 6, 3.825 * 10 ** 6, 3.92 * 10 ** 6]
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 9),
                         subplot_kw={'projection': utm},
                         sharex=True, sharey=True,
                         tight_layout=True)
axes = axes.flatten()
counter = 0
for storm in storms:
    nruns = 8  # Florence and Floyd have 7 ensemble members
    if storm == 'matt':
        nruns = 7  # Matthew only has 6 ensemble members
    # Event IDs for Present
    pres_runs = [f'{storm}_{climate}_ens{i}' for i in np.arange(1, nruns, 1)] + [f'{storm}_{climate}_ensmean']
    # Event IDs for Future
    fut_runs = []
    for run in pres_runs:
        # Loop through 5 different SLR scenarios for each present Scaled run
        fut_runs = fut_runs + [f'{run}_SLR{i}' for i in np.arange(1, 6, 1)]

    ds_pres = fld_da_compound.sel(run=pres_runs).sum(dim='run')
    ds_pres_mask = xr.where(ds_pres > 0, 1, 0)
    ds_fut = fld_da_compound.sel(run=fut_runs).sum(dim='run')
    ds_fut_mask = xr.where(ds_fut > 0, 2, 0)
    ds_diff = ds_fut_mask - ds_pres_mask

    ax = axes[counter]
    ckwargs = dict(cmap='PuRd', vmin=0, vmax=2)
    cs = ds_diff.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=0)
    label = 'Freq of Compound'
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.15, pos0.y0 + pos0.height * 0.1, 0.025, pos0.height * 0.8])
    cbar = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='neither')
    ax.set_title(f'{storm} {climate}')
    counter += 1

ii = 0
for ax in axes:
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
    if ii in [0, 2, 4]:
        ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
        ax.yaxis.set_visible(True)
        ax.xaxis.set_visible(False)
        ax.ticklabel_format(style='sci', useOffset=False)
    if ii in [4, 5]:
        ax.set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
        ax.xaxis.set_visible(True)
        ax.ticklabel_format(style='sci', useOffset=False)

    ax.set_aspect('equal')
    # ax.set_axis_off()
    ax.set_extent(extent, crs=utm)
    ii += 1
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.margins(x=0, y=0)
plt.savefig(
    fr'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\{climate}_test.png',
    dpi=225, bbox_inches="tight")
plt.close()