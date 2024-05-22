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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from matplotlib.ticker import FormatStrFormatter

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

# This script classifies the driver of peak water levels for each PGW run as coastal, runoff, compound.
# It outputs a netcdf with the driver classification, compound areas, and a CSV with peak flood extent total areas
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)

# Load data
os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis')
da_zsmax = xr.open_dataarray('pgw_zsmax.nc')
fld_da_classified = xr.open_dataarray('pgw_drivers_classified.nc')
da_zsmax_diff = xr.open_dataarray('pgw_wl_diff.nc')
cc_run_ids = pd.read_csv('cc_run_ids.csv', index_col=0)

storms = ['flor', 'floy', 'matt']
runs = ['ensmean', 'ens1', 'ens2']
diff_stats_df = pd.DataFrame()
for run in runs:
    da_plot = []
    for storm in storms:
        climate = 'pres'
        mod_name = f'{storm}_{climate}_{run}'
        da = da_zsmax_diff.sel(run=mod_name)
        da.name = mod_name
        da_plot.append(da)

        climate = 'presScaled'
        mod_name = f'{storm}_{climate}_{run}'
        slr_runs = [f'{mod_name}_SLR{i}' for i in np.arange(1, 6, 1)]
        da2 = da_zsmax_diff.sel(run=slr_runs).mean(dim='run')
        da2.name = mod_name
        da_plot.append(da2)

    nrow = 3
    ncol = 2
    plot_difference_map = True
    if plot_difference_map is True:
        n_subplots = nrow * ncol
        first_in_row = np.arange(0, n_subplots, ncol)
        last_in_row = np.arange(ncol - 1, n_subplots, ncol)
        first_row = np.arange(0, ncol)
        last_row = np.arange(first_in_row[-1], n_subplots, 1)

        fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                 figsize=(4.75, 6), subplot_kw={'projection': utm},
                                 sharex=True, sharey=True, tight_layout=True)
        axes = axes.flatten()
        for ii in range(len(axes)):
            cmap = 'seismic'
            norm = mpl.colors.Normalize(vmin=-0.25, vmax=0.25)
            cs = da_plot[ii].plot(ax=axes[ii],
                                  cmap=cmap, norm=norm,
                                  extend='neither', shading='auto',
                                  add_colorbar=False, zorder=2, alpha=1)
            axes[ii].set_title(da_plot[ii].name)

        for ax in axes:
            mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=3, alpha=1)
            ax.set_aspect('equal')
            ax.set_axis_off()

        pos0 = axes[3].get_position()
        cax1 = fig.add_axes([pos0.x1 + 0.15, pos0.y0, 0.05, pos0.height * 0.9])
        cbar1 = fig.colorbar(cs,
                             cax=cax1,
                             orientation='vertical',
                             # ticks=[-1, -0.5, 0, 0.5, 1],
                             label='Peak WL Difference (m)')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(x=0, y=0)
        plt.savefig(fr'peak_wl_diff_cmpd_minus_maxIndiv_{run}.png', dpi=225, bbox_inches="tight")
        plt.close()

    for ds in da_plot:
        ds = ds.where(ds > 0.05)
        print(ds.name)
        qs = ds.quantile(q=[0.50, 0.95])
        stats = [round(ds.mean().values.item(), 2)] + list(qs.to_numpy().round(2)) + [round(ds.max().values.item(), 2)]
        diff_stats_df[ds.name] = stats

diff_stats_df = diff_stats_df.T
diff_stats_df.columns = ['mean', 'median', 'PCT95', 'max']
