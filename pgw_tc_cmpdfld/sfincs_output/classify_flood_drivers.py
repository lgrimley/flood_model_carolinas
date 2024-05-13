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


# This script classifies the driver of peak water levels for each PGW run
# This script outputs a netcdf with the driver classification and a CSV with peak flood extent total areas


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
os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis')

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
# fld_area.to_csv(driver_file.replace('.nc', '_area.csv'))
fld_area.drop(['no_flood', 'compound_coastal', 'compound_runoff'], axis=1, inplace=True)

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

plot_boxplot = True
if plot_boxplot is True:
    fld_area['Total'] = fld_area.sum(axis=1)

    # Rename Columns for plotting
    scenarios = ['Coastal', 'Runoff', 'Compound', 'Total']
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
    props = dict(boxes="white", whiskers="black", caps="black")
    boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='none', markersize=6, markeredgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=6)

    fig, axes = plt.subplots(nrows=4, ncols=1, tight_layout=True, figsize=(5, 6))
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
                             zorder=1
                             )
        ax.scatter(x=ax.get_xticks(), y=da_ensmean_plot[scenarios[counter]].values,
                   s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)
        ax.set_xticklabels(['Flor-Pres\n(n=7)', 'Flor-Fut\n(n=35)',
                            'Floy-Pres\n(n=7)', 'Floy-Fut\n(n=35)',
                            'Matt-Pres\n(n=6)', 'Matt-Fut\n(n=30)'])
        ax.set_ylim(np.floor(da_plot[scenarios[counter]].min()) - 10 ** 3,
                    np.ceil(da_plot[scenarios[counter]].max()) + 10 ** 3
                    )
        ax.set_xlabel(None)
        ax.set_ylabel('Peak Flooded\nArea (sq.km.)')
        kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.8)
        ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
        ax.set_axisbelow(True)
        counter += 1
    plt.suptitle(None)
    plt.subplots_adjust(wspace=0, hspace=0.15)
    plt.margins(x=0, y=0)
    plt.savefig(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\flooded_area_boxplot.png',
                bbox_inches='tight', dpi=255)
    plt.close()

plot_col_chart = True
if plot_col_chart is True:
    nrow = 2
    ncol = 4
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)
    for storm in storms:
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6.5, 4.5),
                                 constrained_layout=True,
                                 sharex=True, sharey=True)
        axes = axes.flatten()
        nruns = 8
        if storm == 'matt':
            nruns = 7
            axes[-1].set_axis_off()
        runs = [f'ens{i}' for i in np.arange(1, nruns, 1)] + ['ensmean']
        for i in range(len(runs)):
            pres_id = [f'{storm}_pres_{runs[i]}']
            print(pres_id)
            fut_ids = [f'{storm}_presScaled_{runs[i]}_SLR{ii}' for ii in np.arange(1, 6, 1)]
            sel_runs = pres_id + fut_ids
            fld_area[fld_area.index.isin(sel_runs)].plot.bar(ax=axes[i],
                                                             stacked=True,
                                                             legend=False,
                                                             title=f'{runs[i]}')
            axes[i].set_ylabel('Peak Flooded\nArea (sq.km.)')
            if i in last_row:
                axes[i].set_xticklabels(['Pres', 'Fut-SLR1', 'Fut-SLR2', 'Fut-SLR3', 'Fut-SLR4', 'Fut-SLR5'],
                                        rotation=90,
                                        #ha='center',
                                        va='top')

        axes[last_in_row[0]].legend(bbox_to_anchor=(1.05, 0.75), loc='upper left', borderaxespad=0.)
        plt.subplots_adjust(wspace=0, hspace=0)#, top=0.95)
        plt.suptitle(f'{storm}')
        plt.margins(x=0, y=0)
        plt.savefig(os.path.join(f'flooded_area_by_driver_barchart_{storm}.png'),
                    bbox_inches='tight',
                    dpi=225)
        plt.close()

# Fractions
fld_area = fld_cells.copy()
res = 200  # meters
fld_area = fld_area * (res * res) / (1000 ** 2)  # square km
fld_area = fld_area.T
fld_area.drop(['no_flood'], axis=1, inplace=True)
fld_area = fld_area.div(fld_area.sum(axis=1), axis=0)
fld_area = round(fld_area * 100, 1)

plot_boxplot_fraction = True
if plot_boxplot_fraction is True:
    # Rename Columns for plotting
    scenarios = ['Coastal', 'Compound-Coastal', 'Runoff', 'Compound-Runoff']
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
    props = dict(boxes="white", whiskers="black", caps="black")
    boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='none', markersize=6, markeredgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=6)

    fig, axes = plt.subplots(nrows=4, ncols=1, tight_layout=True, figsize=(5, 6))
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
                             zorder=1
                             )
        ax.scatter(x=ax.get_xticks(), y=da_ensmean_plot[scenarios[counter]].values,
                   s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)
        ax.set_xticklabels(['Flor-Pres\n(n=7)', 'Flor-Fut\n(n=35)',
                            'Floy-Pres\n(n=7)', 'Floy-Fut\n(n=35)',
                            'Matt-Pres\n(n=6)', 'Matt-Fut\n(n=30)'])
        ax.set_ylim(np.floor(da_plot[scenarios[counter]].min()),
                    np.ceil(da_plot[scenarios[counter]].max())
                    )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.8)
        ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
        ax.set_axisbelow(True)
        counter += 1
    plt.suptitle('Percent Contribution to Peak Flooded Extent')
    plt.subplots_adjust(wspace=0, hspace=0.15)
    plt.margins(x=0, y=0)
    plt.savefig(
        r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\flooded_area_boxplot_fraction.png',
        bbox_inches='tight', dpi=255)
    plt.close()
