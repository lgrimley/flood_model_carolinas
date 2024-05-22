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


work_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis'
out_dir = os.path.join(work_dir, 'ensemble_mean')
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)
os.chdir(out_dir)

# Load a SFINCS model
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')
dep = mod.grid['dep']

for type in ['mean', 'max']:
    fld_da_classified = xr.open_dataarray(os.path.join(work_dir,
                                                       'driver_analysis',
                                                       f'pgw_drivers_classified_ensmean_{type}.nc'))
    da_ensmean = xr.open_dataarray(os.path.join(work_dir,
                                                'zsmax',
                                                f'pgw_ensmean_zsmax_{type}.nc'))

    # No Flood = 0, Coastal = 1, Compound-coastal = 2, Runoff = 3, Compound-runoff = 4
    ds_plot = []
    for storm in ['flor', 'floy', 'matt']:
        pres_wl = da_ensmean.sel(run=f'{storm}_pres_compound_{type}')
        pres_drivers = fld_da_classified.sel(run=f'{storm}_pres_ensmean')

        fut_wl = da_ensmean.sel(run=f'{storm}_presScaled_compound_{type}')
        fut_drivers = fld_da_classified.sel(run=f'{storm}_presScaled_ensmean')

        for scenario in ['coastal', 'runoff', 'compound']:
            if scenario == 'coastal':
                mask_pres = xr.where((pres_drivers == 1), True, False)
                mask_fut = xr.where((fut_drivers == 1), True, False)
            elif scenario == 'runoff':
                mask_pres = xr.where((pres_drivers == 3), True, False)
                mask_fut = xr.where((fut_drivers == 3), True, False)
            else:
                mask_pres = xr.where((pres_drivers == 2) | (pres_drivers == 4), True, False)
                mask_fut = xr.where((fut_drivers == 2) | (fut_drivers == 4), True, False)

            depth_pres = (pres_wl.where(mask_pres) - dep).compute()
            depth_pres = depth_pres.where(depth_pres > 0.05)
            depth_pres.name = f'{storm}_pres_{scenario}'

            depth_fut = (fut_wl.where(mask_fut) - dep).compute()
            depth_fut = depth_fut.where(depth_fut > 0.05)
            depth_fut.name = f'{storm}_fut_{scenario}'

            # ds_plot.append(depth_pres)
            # ds_plot.append(depth_fut)

            diff = (depth_fut.fillna(0) - depth_pres.fillna(0)).compute()
            diff.name = f'{storm}_{scenario}'
            ds_plot.append(diff)

    plot_ensmean_diff_by_driver = True
    if plot_ensmean_diff_by_driver is True:
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
            figsize=(6, 6),
            subplot_kw={'projection': utm},
            tight_layout=True,
            layout='constrained')
        axes = axes.flatten()
        counter = 0
        for ax in axes:
            ckwargs = dict(cmap='seismic', vmin=-3, vmax=3)
            cs = ds_plot[counter].plot(ax=ax,
                                       add_colorbar=False,
                                       **ckwargs,
                                       zorder=0)
            ax.set_title('')
            ax.set_title(ds_plot[counter].name)
            ax.set_axis_off()
            mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
            counter += 1

        label = 'Depth Difference (m)\nFuture minus Present'
        ax = axes[5]
        pos0 = ax.get_position()  # get the original position
        cax = fig.add_axes([pos0.x1 + 0.02, pos0.y0 + pos0.height * -0.0, 0.025, pos0.height * 1.2])
        cbar2 = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='both')

        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.margins(x=0, y=0)
        plt.savefig(f'WL_diff_by_driver_ensemble_mean_{type}.png',
                    tight_layout=True, constrained_layout=True,
                    bbox_inches='tight', dpi=255)
        plt.close()
