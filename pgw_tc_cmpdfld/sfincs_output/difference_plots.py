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

# This script reads in the netCDF of zsmax and plots the difference between the future and present
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

# Filepath to data catalogs yml
yml_pgw = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\data_catalog_pgw.yml'
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_pgw, yml_base])
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]

# Directory to output stuff
out_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis'
results_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\arch'
da_zsmax = mod.data_catalog.get_rasterdataset(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis'
                                              r'\pgw_zsmax.nc')

storms = ['flor', 'floy', 'matt']
storm_titles = ['Florence', 'Floyd', 'Matthew']
# Calculate the difference in peak water level for the RUNOFF scenario
# scenario = 'runoff'
# nrow = 4
# ncol = 2
# n_subplots = nrow * ncol
# first_in_row = np.arange(0, n_subplots, ncol)
# last_in_row = np.arange(ncol - 1, n_subplots, ncol)
# first_row = np.arange(0, ncol)
# last_row = np.arange(first_in_row[-1], n_subplots, 1)
# for storm in storms:
#     fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5.5, 8),
#                              subplot_kw={'projection': utm}, tight_layout=True, layout='constrained')
#     axes = axes.flatten()
#     nruns = 8
#     if storm == 'matt':
#         axes[-1].set_axis_off()
#         nruns = 7
#     runs = [f'ens{i}' for i in np.arange(1, nruns, 1)] + ['ensmean']
#     counter = 0
#     for run in runs:
#         # Calculate water level diff
#         fut_id = f'{storm}_presScaled_{run}_SLR1_{scenario}'
#         pres_id = f'{storm}_pres_{run}_{scenario}'
#         max_pres = da_zsmax.sel(run=pres_id)
#         max_fut = da_zsmax.sel(run=fut_id)
#         diff = (max_fut - max_pres).compute()
# 
#         ax = axes[counter]
#         ckwargs = dict(cmap='Reds', vmin=0, vmax=1.5)
#         cs = diff.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=0)
#         ax.set_title('')
#         ax.set_title(run, loc='center')
# 
#         minx, miny, maxx, maxy = extent
#         ax.set_xlim(minx, maxx)
#         ax.set_ylim(miny, maxy)
#         ax.set_extent(extent, crs=utm)
#         ax.set_axis_off()
#         mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
# 
#         print(counter)
#         counter += 1
#     label = 'Water Level Difference (m)\nFuture (Scaled) minus Present'
#     pos1 = axes[last_in_row[2]].get_position()  # get the original position
#     cbar_ax = fig.add_axes([pos1.x1 + 0.015, pos1.y0 + pos1.height * 0.25, 0.03, pos1.height * 1.5])
#     cbar = fig.colorbar(cs, cax=cbar_ax,
#                         orientation='vertical',
#                         label=label,
#                         extend='max')
# 
#     plt.subplots_adjust(wspace=0.0, hspace=0.08, top=0.92)
#     plt.margins(x=0, y=0)
#     plt.suptitle(f'{storm} {scenario}')
#     plt.savefig(os.path.join(out_dir, f'{storm}_MaxWL_diff_{scenario}.png'), bbox_inches='tight', dpi=255)
#     plt.close()

# Plotting the difference in peak water level for compound ensmean for paper
nrow = 3
ncol = 1
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)
fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4, 6.5),
                         subplot_kw={'projection': utm}, tight_layout=True, layout='constrained')
axes = axes.flatten()
counter = 0
for storm in storms:
    fut_ids = [f'{storm}_presScaled_ensmean_SLR{i}_compound' for i in np.arange(1, 6, 1)]
    pres_ids = f'{storm}_pres_ensmean_compound'

    ds_fut = da_zsmax.sel(run=fut_ids).mean(dim='run')
    ds_pres = da_zsmax.sel(run=pres_ids)
    diff = (ds_fut - ds_pres).compute()
    diff = diff.where(diff > 0.1)

    ax = axes[counter]
    ckwargs = dict(cmap='gist_heat_r', vmin=0.1, vmax=2.1)
    cs = diff.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=0)
    ax.set_title('')
    ax.set_title(storm_titles[counter], loc='center', fontsize=10)

    minx, miny, maxx, maxy = extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_extent(extent, crs=utm)
    ax.set_axis_off()
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)

    print(counter)
    counter += 1
label = 'Peak Water Level Difference (m)\nFuture minus Present'
pos1 = axes[1].get_position()  # get the original position
cbar_ax = fig.add_axes([pos1.x1 + 0.05, pos1.y0 + pos1.height * 0.1, 0.025, pos1.height * 0.9])
cbar = fig.colorbar(cs, cax=cbar_ax,
                    orientation='vertical',
                    label=label,
                    extend='max')
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(out_dir, f'MaxWL_diff_compound_ensmean.png'), bbox_inches='tight', dpi=255)
plt.close()
