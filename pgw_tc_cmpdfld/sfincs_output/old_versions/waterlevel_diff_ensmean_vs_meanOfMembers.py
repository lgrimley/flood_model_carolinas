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


# This script plots the difference between the ensmean water level and the mean water level across the ensemble

def get_zsmax_da(mod_results_dir):
    da_zsmax_list = []
    event_ids = []
    for dir in os.listdir(mod_results_dir):
        mod.read_results(fn_map=os.path.join(results_dir, dir, 'sfincs_map.nc'))
        zsmax = mod.results["zsmax"].max(dim='timemax')
        zsmax.raster.to_raster(os.path.join(results_dir, f'{dir}_zsmax.tif'), nodata=-9999.0)
        da_zsmax_list.append(zsmax)
        event_ids.append(dir)
    da_zsmax = xr.concat(da_zsmax_list, dim='run')
    da_zsmax['run'] = xr.IndexVariable('run', event_ids)
    return da_zsmax, event_ids


# Filepath to data catalogs yml
yml_pgw = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\data_catalog_pgw.yml'
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_pgw, yml_base])

# Directory to output stuff
out_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis'
results_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\arch'

# Get zsmax across model runs
# da_zsmax, event_ids = get_zsmax_da(mod_results_dir=results_dir)
da_zsmax = mod.data_catalog.get_rasterdataset(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis'
                                              r'\pgw_zsmax.nc')
cc_run_ids = pd.read_csv(os.path.join(out_dir, 'cc_run_ids.csv'))

da_diff_list = []
run_ids = []
scenarios = ['coastal', 'runoff', 'compound']
storms = ['flor', 'floy', 'matt']
climates = ['pres', ]
for storm in storms:
    for climate in climates:
        if climate == 'presScaled':
            for scenario in scenarios:
                member_ids = []
                ensmean_ids = [f'{storm}_{climate}_ensmean_SLR{i}_{scenario}' for i in np.arange(1, 6, 1)]
                nruns = 8
                if storm == 'matt':
                    nruns = 7  # only 6 ensemble members
                for ii in range(1, nruns):
                    ids = [f'{storm}_{climate}_ens{ii}_SLR{i}_{scenario}' for i in np.arange(1, 6, 1)]
                    member_ids = member_ids + ids
                print(len(member_ids))
                # Calculate water level diff
                meanOfMembers = da_zsmax.sel(run=member_ids).mean('run')
                ensmean = da_zsmax.sel(run=ensmean_ids).mean('run')
                ensmean_minus_meanOfMembers = (ensmean - meanOfMembers).compute()
                ensmean_minus_meanOfMembers.name = f'{storm}_{climate}_{scenario}'
                ensmean_minus_meanOfMembers['run'] = f'{storm}_{climate}_{scenario}'

                da_diff_list.append(ensmean_minus_meanOfMembers)
                run_ids.append(f'{storm}_{climate}_{scenario}')
        else:
            for scenario in scenarios:
                nruns = 8
                if storm == 'matt':
                    nruns = 7  # only 6 ensemble members
                ensmean_id = f'{storm}_{climate}_ensmean_{scenario}'
                member_ids = [f'{storm}_{climate}_ens{i}_{scenario}' for i in np.arange(1, nruns, 1)]
                print(member_ids)

                # Calculate water level diff
                meanOfMembers = da_zsmax.sel(run=member_ids).mean('run')
                ensmean = da_zsmax.sel(run=ensmean_id)
                ensmean_minus_meanOfMembers = (ensmean - meanOfMembers).compute()

                ensmean_minus_meanOfMembers.name = f'{storm}_{climate}_{scenario}'
                ensmean_minus_meanOfMembers['run'] = f'{storm}_{climate}_{scenario}'
                da_diff_list.append(ensmean_minus_meanOfMembers)

                run_ids.append(f'{storm}_{climate}_{scenario}')

da_diff = xr.concat(da_diff_list, dim='run')
da_diff['run'] = xr.IndexVariable('run', run_ids)
#da_diff = da_diff.sel(run=['flor_pres_compound', 'floy_pres_compound', 'matt_pres_compound'])

# Plotting info
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10).total_bounds)[[0, 2, 1, 3]]

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
nrow = 1
ncol = 3
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)
scenarios = [
    #'coastal',
    #'runoff',
    'combined']
fig, axes = plt.subplots(
    nrows=nrow, ncols=ncol,
    figsize=(6, 3),
    subplot_kw={'projection': utm},
    tight_layout=True,
    layout='constrained')
axes = axes.flatten()
ckwargs = dict(cmap='seismic', vmin=-0.5, vmax=0.5)
counter = 0
for ax in axes:
    run_id = da_diff[counter].run.values.item()
    cs = da_diff[counter].plot(ax=ax, add_colorbar=False, **ckwargs, zorder=0)
    ax.set_title('')
    # ax.set_title(run_id)

    minx, miny, maxx, maxy = extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_extent(extent, crs=utm)
    ax.set_axis_off()
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
    counter += 1
for i in range(len(axes)):
    if i in first_row:
        axes[i].set_title(f'{storms[i]}')
    # if i in first_in_row:
    #     axes[i].text(-0.05, 0.5, f'{storms[int(i / 3)]}',
    #                  horizontalalignment='right',
    #                  verticalalignment='center',
    #                  rotation='vertical',
    #                  transform=axes[i].transAxes)
# Colorbar - Precip
label = 'Water Level\nDifference (m)'
ax = axes[-1]
pos0 = ax.get_position()  # get the original position
cax = fig.add_axes([pos0.x1 + 0.02, pos0.y0 + pos0.height * -0.0, 0.025, pos0.height * 1.2])
cbar = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='both')

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
#plt.suptitle('Present Ensemble Mean minus Mean of the Ensemble Members')
plt.savefig(os.path.join(out_dir, f'MeanMinusMeanOfMembers_pres_combined.png'),
            tight_layout=True, constrained_layout=True,
            bbox_inches='tight', dpi=255)
plt.close()
