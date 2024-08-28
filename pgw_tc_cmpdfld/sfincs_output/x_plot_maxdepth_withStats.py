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

work_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\analysis'
out_dir = os.path.join(work_dir, 'ensemble_mean')
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)
os.chdir(out_dir)

# Load a SFINCS model
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_wrf\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_base])
dep = mod.grid['dep']

coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
coastal_wb = coastal_wb.to_crs(mod.crs)
coastal_wb_clip = coastal_wb.clip(mod.region)

pres_zsmax_all = xr.open_dataarray(os.path.join(work_dir, 'zsmax', f'pgw_zsmax.nc'))
type = 'mean'
fut_ensmean_zsmax = xr.open_dataarray(os.path.join(work_dir, 'zsmax', f'fut_ensemble_zsmax_{type}.nc'))

storms = ['flor', 'floy', 'matt']

ds_plot = []
for storm in storms:
    pres_zsmax = pres_zsmax_all.sel(run=f'{storm}_pres_compound')
    pres_hmax = (pres_zsmax - dep).compute()
    pres_hmax = pres_hmax.where(pres_hmax > 0.05)
    pres_hmax.name = f'{storm}_pres_compound_hmax'

    fut_zsmax = fut_ensmean_zsmax.sel(run=f'{storm}_fut_compound_{type}')
    fut_hmax = (fut_zsmax - dep).compute()
    fut_hmax = fut_hmax.where(fut_hmax > 0.05)
    fut_hmax.name = f'{storm}_fut_compound_hmax'

    dep_diff = (fut_hmax.fillna(0) - pres_hmax.fillna(0)).compute()
    dep_diff.name = f'{storm}_fut_minus_pres_compound_hmax'
    ds_plot = ds_plot + [pres_hmax, fut_hmax, dep_diff]

col_title = ['Present', 'Future', 'Depth Difference']
row_titles = ['Florence', 'Floyd', 'Matthew']
plot_floodmap = False
if plot_floodmap is True:
    # Plotting info
    wkt = mod.grid['dep'].raster.crs.to_wkt()
    utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
    utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)

    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True

    nrow = 3
    ncol = 3
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)

    fig, axes = plt.subplots(
        nrows=nrow, ncols=ncol,
        figsize=(6, 5),
        subplot_kw={'projection': utm},
        tight_layout=True,
        layout='constrained')
    axes = axes.flatten()
    counter = 0
    for ax in axes:
        if counter in last_in_row:
            ckwargs = dict(cmap='Reds', vmin=0, vmax=1.5)
            cs2 = ds_plot[counter].where(ds_plot[counter] > 0).plot(ax=ax,
                                                                    add_colorbar=False,
                                                                    **ckwargs,
                                                                    zorder=1)
            coastal_wb_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=1, alpha=0.5)
        else:
            ckwargs = dict(cmap='Blues', vmin=0.05, vmax=8)
            cs = ds_plot[counter].plot(ax=ax,
                                       add_colorbar=False,
                                       **ckwargs,
                                       zorder=1)
            mod.region.plot(ax=ax, color='grey', edgecolor='none', zorder=0, alpha=0.8)

        mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=2, alpha=1)

        ax.set_title('')
        ax.set_axis_off()
        if counter in first_row:
            ax.set_title(col_title[counter], loc='center', fontsize=10)
        for i in range(len(first_in_row)):
            axes[first_in_row[i]].text(-0.05, 0.5, row_titles[i],
                                       horizontalalignment='right',
                                       verticalalignment='center',
                                       rotation='vertical',
                                       transform=axes[first_in_row[i]].transAxes)
        counter += 1

    label = 'Max Depth (m)'
    ax = axes[2]
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.02, pos0.y0 + pos0.height * -0.4, 0.025, pos0.height * 1.2])
    cbar2 = fig.colorbar(cs,
                         cax=cax,
                         orientation='vertical',
                         label=label,
                         ticks=[0.05, 2, 4, 6, 8],
                         extend='max'
                         )

    label = 'Max Depth\nDifference (m)'
    ax = axes[5]
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.02, pos0.y0 + pos0.height * -0.8, 0.025, pos0.height * 1.2])
    cbar2 = fig.colorbar(cs2,
                         cax=cax,
                         orientation='vertical',
                         label=label,
                         ticks=[0, 0.5, 1, 1.5],
                         extend='max')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.margins(x=0, y=0)
    plt.savefig(f'floodmap_pres_fut_diff_{type}.png',
                tight_layout=True, constrained_layout=True,
                bbox_inches='tight', dpi=255)
    plt.close()

df_dep_stats = pd.DataFrame()
for ds in ds_plot:
    ds = xr.where(ds == 0, np.nan, ds)
    df = ds.to_dataframe().dropna(how='any', axis=0)
    df = df.drop(columns='spatial_ref')
    df_stats = df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    df_dep_stats = pd.concat([df_dep_stats, df_stats], ignore_index=False, axis=1)
    break
df_dep_stats.to_csv(f'max_depth_stats_future_{type}.csv')
