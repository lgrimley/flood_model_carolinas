import os
import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydromt
import os
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns
import cartopy.crs as ccrs
from src.utils import  classify_zsmax_by_process, calc_diff_in_zsmax_compound_minus_max_individual

from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()


'''' Load in the data '''
# Load the water level data for the historical return periods
histdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep'
file_paths = [os.path.join(histdir, file) for file in os.listdir(histdir) if ('attribution' in file) & (file.endswith('.nc'))]
ds_list = [xr.open_dataset(file, chunks={'x': 300, 'y': 300, 'tc_id': 25},
                             engine='netcdf4')['zsmax_diff'] for file in file_paths]
ds_hist = xr.concat(ds_list, dim='tc_id')

# We are only interested in when the compounding changes the water level by more than 0.05 m
mask = np.abs(ds_hist) > 0.05
ds_hist_compound = ds_hist.where(mask)
# Calculate the mean change in depth at all locations where compounding ever occurred
ds_hist_compound_mean = ds_hist_compound.mean(dim='tc_id', skipna=True).compute()
ds_hist_compound_mean.plot(add_colorbar=True, vmin=-.1, vmax=0.1)
plt.imshow

ds_hist_compound_max = ds_hist_compound.max(dim='tc_id', skipna=True).compute()
ds_hist_compound_max.plot(add_colorbar=True)
plt.imshow

# Now get the max extent of compound flooding (exacerbates)
ds_hist_comp_extent = ds_hist_compound.where(ds_hist_compound > 0).sum(dim='tc_id')
# Now get the max extent of compound flooding (lessens)
ds_hist_comp_extent_neg = ds_hist_compound.where(ds_hist_compound < 0).sum(dim='tc_id')


# # Load the water level data for the projected return periods
# projdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\canesm_ssp585'
# file_paths = [os.path.join(histdir, file) for file in os.listdir(histdir) if ('attribution' in file) & (file.endswith('.nc'))]
# ds_list = [xr.open_dataset(file, chunks={'x': 300, 'y': 300, 'tc_id': 25},
#                              engine='netcdf4')['zsmax_diff'] for file in file_paths]
# ds_proj = xr.concat(ds_list, dim='tc_id')
#
# # Load model data/DEM
# yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
# base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
# mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
# dem = mod.grid['dep']
#
#
# compound_diff_da = xr.open_dataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\compound_wl_diff.nc')
# d_plot = ['historical_max_diff', 'projected_max_diff']
# ''' Plot max difference between Compound vs Runoff/Coastal'''
# plot_max_compound_individ_WLdiff = True
# if plot_max_compound_individ_WLdiff is True:
#     os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\figures')
#     scenarios = ['Historic (1980-2005)', 'Projected (2070-2100)']
#     nrow = 1
#     ncol = len(scenarios)
#     n_subplots = nrow * ncol
#     first_in_row = np.arange(0, n_subplots, ncol)
#     last_in_row = np.arange(ncol - 1, n_subplots, ncol)
#     first_row = np.arange(0, ncol)
#     last_row = np.arange(first_in_row[-1], n_subplots, 1)
#
#     # Plot difference in water level raster
#     colors = ['#0000ff', '#0000ff99', 'white', 'white' ,'#ff000099', '#ff0000'] #'#0000ff26',,'#ff000026'
#     levels = [-1, -0.5, -0.15, 0, 0.15, 0.5, 1]
#     cmap, norm = mpl.colors.from_levels_and_colors(levels, colors)
#     fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5, 4),
#                              subplot_kw={'projection': utm},tight_layout=True, layout='constrained')
#     axes = axes.flatten()
#     for i in range(len(axes)):
#         ax = axes[i]
#         data = compound_diff_da[d_plot[i]]
#
#         #ckwargs = dict(cmap=cmap, norm=norm)
#         ckwargs = dict(cmap='Reds', vmin=0.25, vmax=1)
#         mask = data > 0.25
#         cs2 = data.where(mask).plot(ax=ax, add_colorbar=False, **ckwargs, zorder=1)
#
#         ax.set_title('')
#         ax.set_aspect('equal')
#         ax.set_axis_off()
#         mod.region.plot(ax=ax, color='white', edgecolor='none', zorder=0, alpha=1)
#         major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=1)
#         nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=1)
#         coastal_wb_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
#         #ax.plot(*polygon.exterior.xy, color='black', linewidth=1.5, zorder=3, alpha=1)
#         mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.35, zorder=2, alpha=1)
#
#         if i in first_row:
#             ax.set_title(scenarios[i], loc='center', fontsize=10)
#         # for i in range(len(first_in_row)):
#         #     label = f'{rp[i]}%\n({sel_rp[i]}-yr)'
#         #     axes[first_in_row[i]].text(-0.05, 0.5, label,
#         #                                horizontalalignment='right',
#         #                                verticalalignment='center',
#         #                                rotation='horizontal',
#         #                                transform=axes[first_in_row[i]].transAxes)
#
#     label = 'Water Level Difference (m)\ncompound - max. individual'
#     ax = axes[1]
#     pos0 = ax.get_position()  # get the original position
#     cax = fig.add_axes([pos0.x1 + 0.12, pos0.y0 +0.1, 0.025, pos0.height * 0.8])
#     cbar = fig.colorbar(cs2,
#                         cax=cax,
#                         orientation='vertical',
#                         label=label,
#                         extend='max')
#
#     plt.subplots_adjust(wspace=-0.10, hspace=0)
#     plt.margins(x=0, y=0)
#     plt.savefig(r'wl_compound_individ_diff_discrete.png', bbox_inches='tight', dpi=300)
#     plt.close()