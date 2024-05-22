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


def calculate_ensmean_zsmax(da_zsmax, storm, climate, scenario, type='mean'):
    nruns = 8
    if storm == 'matt':
        nruns = 7

    ids = [f'{storm}_{climate}_ens{ii}_{scenario}' for ii in np.arange(1, nruns, 1)]
    if climate == 'presScaled':
        if scenario in ['compound', 'coastal']:
            for ii in range(1, nruns):
                ids = [f'{storm}_{climate}_ens{ii}_SLR{i}_{scenario}' for i in np.arange(1, 6, 1)]
    print(ids)
    if type == 'mean':
        meanOfMembers = da_zsmax.sel(run=ids).mean('run')
    elif type == 'max':
        meanOfMembers = da_zsmax.sel(run=ids).max('run')
    meanOfMembers.name = f'{storm}_{climate}_{scenario}_{type}'

    return meanOfMembers


def classify_zsmax_by_driver(da, compound_key, runoff_key, coastal_key, name_out, hmin):
    # Calculate the max water level at each cell across the coastal and runoff drivers
    da_single_max = da.sel(run=[runoff_key, coastal_key]).max('run')
    # Calculate the difference between the max water level of the compound and the max of the individual drivers
    da_diff = (da.sel(run=compound_key) - da_single_max).compute()
    da_diff.name = 'diff in waterlevel compound minus max. single driver'
    da_diff.attrs.update(unit='m')

    # Create masks based on the driver that caused the max water level given a depth threshold hmin
    compound_mask = da_diff > hmin
    coastal_mask = da.sel(run=coastal_key).fillna(0) > da.sel(run=[runoff_key]).fillna(0).max('run')
    runoff_mask = da.sel(run=runoff_key).fillna(0) > da.sel(run=[coastal_key]).fillna(0).max('run')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    # No Flood = 0, Coastal = 1, Compound-coastal = 2, Runoff = 3, Compound-runoff = 4
    da_classified = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
                     + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()
    da_classified.name = name_out

    # Calculate the number of cells that are attributed to the different drivers
    unique_codes, fld_area_by_driver = np.unique(da_classified.data, return_counts=True)

    # Return compound only locations
    da_compound = xr.where(compound_mask, x=1, y=0)
    da_compound.name = name_out

    return da_classified, fld_area_by_driver, da_compound, da_diff


os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis')
da_zsmax = xr.open_dataarray(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\pgw_zsmax.nc')
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')
dep = mod.grid['dep']

type = 'max'
if os.path.exists(f'pgw_ensmean_zsmax_{type}.nc') is False:
    ensmean_list = []
    run_ids = []
    storms = ['flor', 'floy', 'matt']
    climates = ['pres', 'presScaled']
    scenarios = ['coastal', 'runoff', 'compound']
    for storm in storms:
        for climate in climates:
            for scenario in scenarios:
                ensmean = calculate_ensmean_zsmax(da_zsmax=da_zsmax, storm=storm, climate=climate,
                                                  scenario=scenario, type=type)
                ensmean_list.append(ensmean)
                run_ids.append(ensmean.name)
                print(ensmean.name)

    da_ensmean = xr.concat(ensmean_list, dim='run')
    da_ensmean['run'] = xr.IndexVariable('run', run_ids)
    da_ensmean.to_netcdf(f'pgw_ensmean_zsmax_{type}.nc')
else:
    da_ensmean = xr.open_dataarray(f'pgw_ensmean_zsmax_{type}.nc')

if os.path.exists(f'pgw_ensmean_drivers_classified_{type}.nc') is False:
    fld_cells = pd.DataFrame()  # dataframe populated with total flooded area
    fld_da_compound = []  # populated with data arrays of the compound areas for each run
    fld_da_classified = []
    fld_da_diff = []
    run_ids = []
    for storm in ['flor', 'floy', 'matt']:
        for climate in ['pres', 'presScaled']:
            compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_compound_{type}',
                                                     f'{storm}_{climate}_runoff_{type}',
                                                     f'{storm}_{climate}_coastal_{type}']
            out = classify_zsmax_by_driver(da=da_ensmean,
                                           compound_key=compound_key, runoff_key=runoff_key,
                                           coastal_key=coastal_key, name_out=f'{storm}_{climate}_ensmean',
                                           hmin=0.05)
            da_classified, fld_cells_by_driver, da_compound, da_diff = out
            fld_da_classified.append(da_classified)
            fld_cells[f'{da_compound.name}'] = fld_cells_by_driver
            fld_da_compound.append(da_compound)
            fld_da_diff.append(da_diff)
            run_ids.append(f'{da_compound.name}')
            print(da_compound.name)

    # Concatenate the data arrays
    fld_da_compound = xr.concat(fld_da_compound, dim='run')
    fld_da_compound['run'] = xr.IndexVariable('run', run_ids)
    fld_da_compound.to_netcdf(f'pgw_ensmean_compound_extent_{type}.nc')

    fld_da_classified = xr.concat(fld_da_classified, dim='run')
    fld_da_classified['run'] = xr.IndexVariable('run', run_ids)
    fld_da_classified.to_netcdf(f'pgw_ensmean_drivers_classified_{type}.nc')

    fld_da_diff = xr.concat(fld_da_diff, dim='run')
    fld_da_diff['run'] = xr.IndexVariable('run', run_ids)
    fld_da_diff.to_netcdf(f'pgw_ensmean_wl_diff_{type}.nc')

    # Cleanup flood area dataframe
    # No Flood = 0, Coastal = 1, Compound-coastal = 2, Runoff = 3, Compound-runoff = 4
    fld_cells.index = ['no_flood', 'coastal', 'compound_coastal', 'runoff', 'compound_runoff']
    fld_cells = pd.DataFrame(fld_cells)
    fld_cells.to_csv(f'pgw_ensmean_compound_extent_{type}.csv')
else:
    fld_da_compound = xr.open_dataarray(f'pgw_ensmean_compound_extent_{type}.nc')
    fld_da_classified = xr.open_dataarray(f'pgw_ensmean_drivers_classified_{type}.nc')
    fld_da_diff = xr.open_dataarray(f'pgw_ensmean_wl_diff_{type}.nc')
    fld_cells = pd.read_csv(f'pgw_ensmean_compound_extent_{type}.csv')

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
# label = 'Peak Depth (m)'
# ax = axes[first_row[-1]]
# pos0 = ax.get_position()  # get the original position
# cax = fig.add_axes([pos0.x1 + 0.02, pos0.y0 + pos0.height * -0.0, 0.025, pos0.height * 1.2])
# cbar = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='both')

label = 'Depth Difference (m)'
ax = axes[5]
pos0 = ax.get_position()  # get the original position
cax = fig.add_axes([pos0.x1 + 0.02, pos0.y0 + pos0.height * -0.0, 0.025, pos0.height * 1.2])
cbar2 = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='both')

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
plt.savefig(f'test.png',
            tight_layout=True, constrained_layout=True,
            bbox_inches='tight', dpi=255)
plt.close()
