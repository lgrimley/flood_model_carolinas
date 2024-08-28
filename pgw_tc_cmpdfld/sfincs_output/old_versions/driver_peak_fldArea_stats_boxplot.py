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
work_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis'
out_dir = os.path.join(work_dir, 'driver_fldArea_stats')
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)
os.chdir(out_dir)

fld_cells = pd.read_csv(os.path.join(work_dir,
                                     'driver_analysis',
                                     'pgw_drivers_classified_all_cellCount.csv'),
                        index_col=0)

fld_area_file = os.path.join(work_dir, 'driver_analysis', 'pgw_drivers_classified_all_area.csv')
if os.path.exists(fld_area_file) is False:
    # Calculate the area of peak flooding per driver using the area of the grid cells
    fld_area = fld_cells.copy()
    res = 200  # meters
    fld_area = fld_area * (res * res) / (1000 ** 2)  # square km
    fld_area = fld_area.T

    # Combine into single compound flood driver, remove columns, calculate total cells flooded
    fld_area['compound'] = fld_area['compound_coastal'] + fld_area['compound_runoff']
    fld_area.to_csv(fld_area_file)
else:
    fld_area = pd.read_csv(fld_area_file, index_col=0)
    # remove climate change runs
    cc_run_ids = pd.read_csv(os.path.join(work_dir, 'zsmax', 'cc_run_ids.csv'), index_col=0)
    cc_run_ids.columns = ['event_id']
    fld_area = fld_area[~fld_area.index.isin(cc_run_ids['event_id'].values.tolist())]

# Data organization for boxplots
fld_area.drop(['no_flood', 'compound_coastal', 'compound_runoff'], axis=1, inplace=True)
fld_area['Total'] = fld_area.sum(axis=1)
scenarios = ['Coastal', 'Runoff', 'Compound', 'Total']
fld_area.columns = scenarios
fld_area['storm'] = [i.split("_")[0] for i in fld_area.index]
fld_area['climate'] = [i.split("_")[1] for i in fld_area.index]
fld_area['run'] = [i.split("_")[2] for i in fld_area.index]
fld_area['group'] = fld_area['storm'] + ' ' + fld_area['climate']


# Plotting info
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

boxplot = True
if boxplot is True:
    # Drop ensemble means from dataframe
    da_plot = fld_area.drop(fld_area[fld_area['run'] == 'ensmean'].index)
    da_plot.sort_values(by='group', ascending=True, inplace=True)

    for storm in fld_area.storm.unique():
        for climate in fld_area.climate.unique():
            sub = da_plot[da_plot['storm'] == storm][da_plot['climate'] == climate]
            stats = sub.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            stats.to_csv(f'{storm}_{climate}_driver_fldArea_stats.csv')

    # PLOTTING Boxplot of flooded area
    nrow = 4
    ncol = 1
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)

    props = dict(boxes="white", whiskers="black", caps="black")
    boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='none', markersize=6, markeredgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=6)

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, tight_layout=True, figsize=(5, 6))
    axes = axes.flatten()
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

        ax.set_ylim(np.floor(da_plot[scenarios[counter]].min()) - 10 ** 3,
                    np.ceil(da_plot[scenarios[counter]].max()) + 10 ** 3)

        if counter in last_row:
            xtick = ax.get_xticks()
            ax.set_xticklabels(['Flor-Pres\n(n=7)', 'Flor-Fut\n(n=35)',
                                'Floy-Pres\n(n=7)', 'Floy-Fut\n(n=35)',
                                'Matt-Pres\n(n=6)', 'Matt-Fut\n(n=30)'])
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_title(scenarios[counter])
        ax.set_xlabel(None)
        ax.set_ylabel('Peak Flooded\nArea (sq.km.)')
        kwargs = dict(linestyle='--', linewidth=1, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
        ax.set_axisbelow(True)
        counter += 1

    plt.suptitle(None)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig('driver_fldArea_boxplot_clean.png', bbox_inches='tight', dpi=255)
    plt.close()

boxplot_withEnsmean = False
if boxplot_withEnsmean is True:
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
    plt.savefig('driver_fldArea_boxplot_with_WRFensmean.png', bbox_inches='tight', dpi=255)
    plt.close()

# plot_col_chart = False
# if plot_col_chart is True:
#     fld_area = fld_area.drop('Total', axis=1)
#     nrow = 2
#     ncol = 4
#     n_subplots = nrow * ncol
#     first_in_row = np.arange(0, n_subplots, ncol)
#     last_in_row = np.arange(ncol - 1, n_subplots, ncol)
#     first_row = np.arange(0, ncol)
#     last_row = np.arange(first_in_row[-1], n_subplots, 1)
#     for storm in storms:
#         fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6.5, 4.5),
#                                  constrained_layout=True,
#                                  sharex=True, sharey=True)
#         axes = axes.flatten()
#         nruns = 8
#         if storm == 'matt':
#             nruns = 7
#             axes[-1].set_axis_off()
#         runs = [f'ens{i}' for i in np.arange(1, nruns, 1)] + ['ensmean']
#         for k in range(len(runs)):
#             pres_id = [f'{storm}_pres_{runs[k]}']
#             print(pres_id)
#             fut_ids = [f'{storm}_presScaled_{runs[k]}_SLR{ii}' for ii in np.arange(1, 6, 1)]
#             sel_runs = pres_id + fut_ids
#             fld_area[fld_area.index.isin(sel_runs)].plot.bar(ax=axes[k],
#                                                              stacked=True,
#                                                              legend=False,
#                                                              title=f'{runs[k]}')
#             axes[k].set_ylabel('Peak Flooded\nArea (sq.km.)')
#             if k in last_row:
#                 axes[k].set_xticklabels(['Pres', 'Fut-SLR1', 'Fut-SLR2', 'Fut-SLR3', 'Fut-SLR4', 'Fut-SLR5'],
#                                         rotation=90,
#                                         # ha='center',
#                                         va='top')
#
#         axes[last_in_row[0]].legend(bbox_to_anchor=(1.05, 0.75), loc='upper left', borderaxespad=0.)
#         plt.subplots_adjust(wspace=0, hspace=0)  # , top=0.95)
#         plt.suptitle(f'{storm}')
#         plt.margins(x=0, y=0)
#         plt.savefig(os.path.join(f'flooded_area_by_driver_barchart_{storm}.png'),
#                     bbox_inches='tight',
#                     dpi=225)
#         plt.close()
#
# plot_col_chart_cc = False
# if plot_col_chart_cc is True:
#     storm = 'floy'
#     da_storm_pres = fld_area[(fld_area.storm == storm)][(fld_area.climate == 'pres')]
#     da_storm_fut = fld_area[(fld_area.storm == storm)][(fld_area.climate == 'presScaled')]
#
#     run = 'ensmean'
#     pres_msl = da_storm_pres[(da_storm_pres.run == run)][(da_storm_pres.coast == 'MSL')]
#     fut_msl = da_storm_fut[(da_storm_fut.run == run)][(da_storm_fut.coast == 'MSL')]
#
#     pres_slr = da_storm_pres[(da_storm_pres.run == run)][~(da_storm_pres.coast == 'MSL')]
#     fut_slr = da_storm_fut[(da_storm_fut.run == run)][~(da_storm_fut.coast == 'MSL')]
#
#     org = [[pres_slr.index[i], fut_slr.index[i]] for i in range(5)]
#     org = sum(org, [])
#     interest = list(pres_msl.index) + list(fut_msl.index) + org
#     sub = fld_area[fld_area.index.isin(interest)]
#
#     nrow = 2
#     ncol = 3
#     n_subplots = nrow * ncol
#     first_in_row = np.arange(0, n_subplots, ncol)
#     last_in_row = np.arange(ncol - 1, n_subplots, ncol)
#     first_row = np.arange(0, ncol)
#     last_row = np.arange(first_in_row[-1], n_subplots, 1)
#     fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6, 4), sharex=True, sharey=True, tight_layout=True)
#     axes = axes.flatten()
#     k = 0
#     for c in sub['coast'].unique():
#         ax = axes[k]
#         subsub = sub[sub['coast'] == c]
#         subsub.set_index('group', inplace=True, drop=True)
#         subsub.drop(['Total', 'storm', 'climate', 'run', 'coast'], axis=1, inplace=True)
#         dp = subsub.plot.bar(ax=ax, stacked=True, legend=False)
#         ax.set_title(c)
#         ax.set_axisbelow(True)
#         ax.grid(color='gray', linestyle='dashed')
#         axes[k].set_xticklabels(['Pres\nPrecip', 'Fut\nPrecip'],
#                                 rotation=0,
#                                 # ha='center',
#                                 va='top')
#         axes[k].set_xlabel('')
#         k += 1
#         print(c)
#     for i in first_in_row:
#         axes[i].set_ylabel('Peak Flooded\nArea (sq.km.)')
#     axes[last_in_row[0]].legend(bbox_to_anchor=(1.05, 0.75), loc='upper left', borderaxespad=0.)
#     plt.subplots_adjust(wspace=0, hspace=0)  # , top=0.95)
#     plt.margins(x=0, y=0)
#     plt.suptitle(f'{storm} {run}')
#     plt.savefig(f'{storm}_{run}_flooded_area_barChart.png', dpi=255)
#     plt.close()
