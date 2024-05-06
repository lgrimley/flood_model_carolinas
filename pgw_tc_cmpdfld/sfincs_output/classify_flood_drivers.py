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
from shapely import geometry
import pyproj
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



def classify_zsmax_by_driver(da, storm, climate, run, hmin):
    compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_{run}_compound',
                                             f'{storm}_{climate}_{run}_runoff',
                                             f'{storm}_{climate}_{run}_coastal']

    # Calculate the max water level at each cell across the coastal and runoff drivers
    da_single_max = da.sel(run=[runoff_key, coastal_key]).max('run')
    # Calculate the difference between the max water level of the compound and the max of the individual drivers
    da_diff = (da.sel(run=compound_key) - da_single_max).compute()
    da_diff.name = 'diff. in waterlevel\ncompound - max. single driver'
    da_diff.attrs.update(unit='m')

    # Create masks based on the driver that caused the max water level given a depth threshold hmin
    compound_mask = da_diff > hmin
    coastal_mask = da.sel(run=coastal_key).fillna(0) > da.sel(run=[runoff_key]).fillna(0).max('run')
    runoff_mask = da.sel(run=runoff_key).fillna(0) > da.sel(run=[coastal_key]).fillna(0).max('run')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    da_c = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
            + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()
    da_c.name = f'{storm}_{climate}_{run}'

    # Calculate the number of cells that are attributed to the different drivers
    unique_codes, fld_area_by_driver = np.unique(da_c.data, return_counts=True)

    # Return compound only locations
    da_compound = xr.where(compound_mask, x=1, y=0)
    da_compound.name = f'{storm}_{climate}_{run}'

    return fld_area_by_driver, da_compound


zsmax_file = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\pgw_zsmax.nc'
da_zsmax = xr.open_dataarray(zsmax_file)


storms = ['flor', 'matt', 'floy']
climates = ['pres', 'presScaled']
driver_file = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\pgw_drivers.nc'

if os.path.exists(driver_file) is False:
    hmin = 0.05  # minimum difference between the individual and compound drivers
    fld_cells = pd.DataFrame()  # dataframe populated with total flooded area
    fld_da_compound = []  # populated with data arrays of the compound areas for each run
    run_ids = []  # keep track of the run IDs and their order
    # Loop through and classify flooding by driver
    for storm in storms:
        for climate in climates:
            # Create a list of runs to loop through
            nruns = 8  # Florence and Floyd have 7 ensemble members
            if storm == 'matt':
                nruns = 7  # Matthew only has 6 ensemble members
            # Add the ensemble mean to the list of runs
            runs = [f'ens{i}' for i in np.arange(1, nruns, 1)] + ['ensmean']

            if climate == 'presScaled':
                for run in runs:
                    # Loop through 5 different SLR scenarios for each present Scaled run
                    slr_runs = [f'{run}_SLR{i}' for i in np.arange(1, 6, 1)]
                    for slr_run in slr_runs:
                        print(slr_run)
                        # Calculate the number of grid cells attributed to each driver for the peak flood extent
                        # and return data array of compound
                        fld_cells_by_driver, da_compound = classify_zsmax_by_driver(da=da_zsmax,
                                                                                    storm=storm, climate=climate,
                                                                                    run=slr_run, hmin=hmin)
                        fld_cells[f'{da_compound.name}'] = fld_cells_by_driver
                        fld_da_compound.append(da_compound)
                        run_ids.append(f'{da_compound.name}')
            else:
                # If the climate runs are WRF present and future, loop through the members and ensmean
                for run in runs:
                    fld_cells_by_driver, da_compound = classify_zsmax_by_driver(da=da_zsmax,
                                                                                storm=storm, climate=climate,
                                                                                run=run, hmin=hmin)
                    fld_cells[f'{da_compound.name}'] = fld_cells_by_driver
                    fld_da_compound.append(da_compound)
                    run_ids.append(f'{da_compound.name}')
    # Concatenate the data arrays
    fld_da_compound = xr.concat(fld_da_compound, dim='run')
    fld_da_compound['run'] = xr.IndexVariable('run', run_ids)
    fld_da_compound.to_netcdf(driver_file)

    # Cleanup flood area dataframe
    fld_cells.index = ['no_flood', 'coastal', 'compound_coastal', 'runoff', 'compound_runoff']
    fld_cells = pd.DataFrame(fld_cells)
    fld_cells.to_csv(driver_file.replace('.nc', '.csv'))
else:
    da_compound = xr.open_dataarray(driver_file)
    fld_cells = pd.read_csv(driver_file.replace('.nc', '.csv'), index_col=0)

# Calculate the area of peak flooding per driver using the area of the grid cells
fld_area = fld_cells.copy()
res = 200  # meters
fld_area = fld_area * (res * res) / (1000 ** 2)  # square km
fld_area = fld_area.T

# Combine into single compound flood driver, remove columns, calculate total cells flooded
fld_area['compound'] = fld_area['compound_coastal'] + fld_area['compound_runoff']
fld_area.drop(['no_flood', 'compound_coastal', 'compound_runoff'], axis=1, inplace=True)
fld_area['All Drivers'] = fld_area.sum(axis=1)

# Rename Columns for plotting
scenarios = ['Coastal', 'Runoff', 'Compound', 'All Drivers']
fld_area.columns = scenarios
fld_area['storm'] = [i.split("_")[0] for i in fld_area.index]
fld_area['climate'] = [i.split("_")[1] for i in fld_area.index]
fld_area['run'] = [i.split("_")[2] for i in fld_area.index]
fld_area['group'] = fld_area['storm'] + ' ' + fld_area['climate']

# Drop ensemble means from dataframe
da_plot = fld_area.drop(fld_area[fld_area['run'] == 'ensmean'].index)
da_plot.sort_values(by='group', ascending=True, inplace=True)

# Organize dataframe of ensemble means
da_ensmean_plot = fld_area.drop(fld_area[fld_area['run'] != 'ensmean'].index)
da_ensmean_plot.drop(['storm', 'climate', 'run'], axis=1, inplace=True)
da_ensmean_plot = da_ensmean_plot.groupby('group').mean()
da_ensmean_plot.sort_values(by='group', ascending=True, inplace=True)

# PLOTTING Boxplot of flooded area
font = {'family': 'Arial', 'size': 12}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 12})
mpl.rcParams["figure.autolayout"] = True
props = dict(boxes="white", whiskers="black", caps="black")
boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
flierprops = dict(marker='o', markerfacecolor='none', markersize=6, markeredgecolor='black')
medianprops = dict(linestyle='-', linewidth=2, color='black')
meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=6)

fig, axes = plt.subplots(nrows=4, ncols=1, tight_layout=True, figsize=(8, 8))
counter = 0
for ax in axes:
    print(counter)
    bp = da_plot.boxplot(ax=ax,
                         by='group',
                         column=scenarios[counter],
                         vert=True,
                         color=props,
                         boxprops=boxprops,
                         flierprops=flierprops,
                         medianprops=medianprops,
                         meanprops=meanpointprops,
                         meanline=False,
                         showmeans=True,
                         patch_artist=True,
                         layout=(3, 1),
                         fontsize=12,
                         zorder=1
                         )
    ax.scatter(x=ax.get_xticks(), y=da_ensmean_plot[scenarios[counter]].values,
               s=30, color='red', marker='s', zorder=2, edgecolor='black', alpha=0.8)
    # ax.set_xticklabels(['Flor\nFut', 'Flor\nPres',
    #                     'Floy\nFut', 'Floy\nPres',
    #                     'Matt\nFut', 'Matt\nPres'])
    ax.set_ylim(np.floor(da_plot[scenarios[counter]].min()) - 10 ** 3,
                np.ceil(da_plot[scenarios[counter]].max()) + 10 ** 3
                )
    ax.set_xlabel(None)
    ax.set_ylabel('Flooded Area\n(sq.km.)')
    kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.8)
    ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
    kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
    ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
    ax.set_axisbelow(True)
    counter += 1
plt.suptitle(None)
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\flooded_area_boxplot_v2.png')
plt.close()