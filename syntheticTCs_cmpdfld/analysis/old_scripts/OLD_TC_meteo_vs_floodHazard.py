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
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep')
fld_extent_df = pd.read_csv(f'overland_flooded_area_table.csv', index_col=0)
fld_extent_df['AOI'] = [s.replace(" ","") for s in fld_extent_df['AOI']]
fld_extent_df = fld_extent_df[['Coastal', 'Compound', 'Runoff', 'Total_Flooded','AOI']]
fld_extent_stats = fld_extent_df.groupby('AOI').describe(percentiles=[0.99, 0.90, 0.75, 0.25]).T

# Load meteo information
ws_df = pd.read_csv(r'..\..\02_DATA\NCEP_Reanalysis\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df = pd.read_csv(r'..\..\02_DATA\NCEP_Reanalysis\rain\basin_tc_precipitation_stats.csv', index_col=0)
meteo_df = pd.concat(objs=[ws_df, rain_df], ignore_index=False, axis=1)
df_list = [meteo_df[meteo_df.index == i].T  for i in meteo_df.index]
meteo_df = pd.concat(objs=df_list,axis=1)
basins = [x.split('_')[0] for x in meteo_df.index]
vars = [''.join(x.split('_')[1:]) for x in meteo_df.index]
tuples = [(basins[i], vars[i]) for i in range(len(vars))]
meteo_df.index = pd.MultiIndex.from_tuples(tuples, names=['Basin', 'Variable'])
meteo_df = meteo_df.T
meteo_stats = meteo_df.describe(percentiles=[0.99, 0.90, 0.75, 0.25])

pct_thresh = 0
basins = np.unique(fld_extent_df['AOI'].values).tolist()
scenarios = ['Coastal','Runoff', 'Compound', 'Total_Flooded']
data = {}
for i in range(len((basins))):
    basin = basins[i]
    data[basin] = {}
    for k in range(len(scenarios)):
        scenario = scenarios[k]
        storms = fld_extent_df[fld_extent_df['AOI'] == basin][scenario]
        if pct_thresh != 0:
            thresh = fld_extent_stats[basin][scenario][pct_thresh]
            storms = storms[storms > thresh].dropna()
        else:
            storms = storms.dropna()
        meteo_subset = meteo_df[meteo_df.index.isin(storms.index)][basin]
        combined = pd.concat(objs=[storms, meteo_subset], axis=1, ignore_index=False)
        data[basin][scenario] = combined

#### FUTURE #####
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\canesm_ssp585')
fld_extent_df = pd.read_csv(f'overland_flooded_area_table.csv', index_col=0)
fld_extent_df['AOI'] = [s.replace(" ","") for s in fld_extent_df['AOI']]
fld_extent_df = fld_extent_df[['Coastal', 'Compound', 'Runoff', 'Total_Flooded','AOI']]
fld_extent_stats = fld_extent_df.groupby('AOI').describe(percentiles=[0.99, 0.90, 0.75, 0.25]).T

# Load meteo information
ws_df = pd.read_csv(r'..\..\02_DATA\CMIP6_585\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df = pd.read_csv(r'..\..\02_DATA\CMIP6_585\rain\basin_tc_precipitation_stats.csv', index_col=0)
meteo_df = pd.concat(objs=[ws_df, rain_df], ignore_index=False, axis=1)
df_list = [meteo_df[meteo_df.index == i].T  for i in meteo_df.index]
meteo_df = pd.concat(objs=df_list,axis=1)
basins = [x.split('_')[0] for x in meteo_df.index]
vars = [''.join(x.split('_')[1:]) for x in meteo_df.index]
tuples = [(basins[i], vars[i]) for i in range(len(vars))]
meteo_df.index = pd.MultiIndex.from_tuples(tuples, names=['Basin', 'Variable'])
meteo_df = meteo_df.T
meteo_stats = meteo_df.describe(percentiles=[0.99, 0.90, 0.75, 0.25])

basins = np.unique(fld_extent_df['AOI'].values).tolist()
scenarios = ['Coastal','Runoff', 'Compound', 'Total_Flooded']
data_proj = {}
for i in range(len((basins))):
    basin = basins[i]
    data_proj[basin] = {}
    for k in range(len(scenarios)):
        scenario = scenarios[k]
        storms = fld_extent_df[fld_extent_df['AOI'] == basin][scenario]
        if pct_thresh != 0:
            thresh = fld_extent_stats[basin][scenario][pct_thresh]
            storms = storms[storms > thresh].dropna()
        else:
            storms = storms.dropna()
        meteo_subset = meteo_df[meteo_df.index.isin(storms.index)][basin]
        combined = pd.concat(objs=[storms, meteo_subset], axis=1, ignore_index=False)
        data_proj[basin][scenario] = combined


#### PLOTTING METEO dist #####

# variables = ['meanRRthresh', 'meanWSthresh', 'maxRR']
# var = variables[2]
#
# basins = ['CapeFear', 'LowerPeeDee', 'Neuse', 'OnslowBay', 'Pamlico']
# ncols = len(scenarios)
# nrows = len(basins)
#
# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8), sharey=True, sharex=True)
# for i in range(nrows):
#     basin = basins[i]
#     basin_data = data[basin]
#     basin_data_proj = data_proj[basin]
#
#     for k in range(ncols):
#         scen = scenarios[k]
#         ax = axs[i, k]
#         sns.histplot(data=basin_data[scen][var], label='Hist', fill=True, ax=ax, color='red', kde=True)
#         sns.histplot(data=basin_data_proj[scen][var], label='Proj', fill=True, ax=ax, color='grey', kde=True)
#         ax.set_ylabel('Density')
#         ax.set_xlabel('Max Rain Rate (mm/hr)')
#
#     axs[i,0].text(-0.35, 0.5, basins[i], horizontalalignment='right', verticalalignment='center',
#             rotation='vertical', transform=axs[i,0].transAxes)
# for k in range(ncols):
#     ax = axs[0, k]
#     ax.set_title(scenarios[k])
# plt.legend()
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.margins(x=0, y=0)
# plt.tight_layout()
# plt.savefig('top_90PCT_fldarea_dist_maxRR.png')


##### PLOT ONE ########
# variable_x = 'meanWSthresh'
# variable_y = 'meanRRthresh'
# ylab = 'Mean Rain Rate > 5 m/hr'
# xlab = 'Mean Wind Speed > 5 m/s'
# nrow = len(basins)
# ncol = len(scenarios)
# font = {'family': 'Arial', 'size': 10}
# mpl.rc('font', **font)
# mpl.rcParams.update({'axes.titlesize': 10})
# mpl.rcParams["figure.autolayout"] = True
# n_subplots = nrow * ncol
# first_in_row = np.arange(0, n_subplots, ncol)
# last_in_row = np.arange(ncol - 1, n_subplots, ncol)
# first_row = np.arange(0, ncol)
# last_row = np.arange(first_in_row[-1], n_subplots, 1)
#
# figsize = (9, 8)
# fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize,
#                          tight_layout=True, sharex=True, sharey=True)
# for i in range(len((basins))):
#     basin = basins[i]
#     for k in range(len(scenarios)):
#         scenario = scenarios[k]
#         df = data[basin][scenario]
#         df2 = data_proj[basin][scenario]
#         ax = axes[i][k]
#         ckwargs = dict(cmap='Blues', norm=LogNorm())
#         sc = ax.scatter(df2[variable_x], df2[variable_y], c=df2[scenario],  **ckwargs,
#                      marker='^', edgecolor='none', s=80)
#         sc2 = ax.scatter(df[variable_x], df[variable_y], c=df[scenario], **ckwargs,
#                       marker='.', edgecolor='none', s=100, alpha=0.8)
#         ax.grid(True)
#         ax.set_yscale('log')
#         ax.set_xscale('log')
#         ax.set_axisbelow(True)
# axes = axes.flatten()
# for kk in range(nrow):
#     axes[first_in_row[kk]].text(-0.38, 0.5, basins[kk],
#                                 horizontalalignment='right',
#                                 verticalalignment='center',
#                                 rotation='vertical',
#                                 transform=axes[first_in_row[kk]].transAxes)
#     axes[first_in_row[kk]].set_ylabel(ylab)
#     if kk == 2:
#         pos0 = axes[first_in_row[kk]].get_position()  # get the original position
#         cax1 = fig.add_axes([pos0.x1 + 0.7, pos0.y0, 0.05, pos0.height * 2.5])
#         cbar1 = fig.colorbar(sc,
#                              cax=cax1,
#                              orientation='vertical',
#                              label='Area (sq.km)'
#                              )
# for kk in range(ncol):
#     axes[last_row[kk]].set_xlabel(xlab)
#     axes[first_row[kk]].set_title(scenarios[kk])
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.margins(x=0, y=0)
# plt.tight_layout()
# plt.imshow
# plt.savefig(f'flood_area_vs_TC_meteo_both_{pct_thresh}.png', dpi=300, bbox_inches="tight")
# plt.close()


#### COMPOUND ONLY!!!! ###

print(data['Pamlico']['Coastal'].columns)
variable_x = 'meanWSthresh'
variable_y = 'meanRRthresh'
variable_y_label = 'Mean Rain Rate\n > 5 m/hr'
variable_x_label = 'Mean Wind Speed\n > 5 m/s'
plot_please = True
if plot_please is True:
    nrow = 6
    ncol = 2
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)

    figsize = (6, 8)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize, tight_layout=True, sharex=True, sharey=True)
    for i in range(len((basins))):
        basin = basins[i]
        scenario = 'Compound'
        ckwargs = dict(cmap='Blues', norm=LogNorm())

        ax = axes[i][0]
        df = data[basin][scenario]
        sc = ax.scatter(df[variable_x], df[variable_y], c=df[scenario],  **ckwargs,
                     marker='.', edgecolor='none', s=80)
        ax.text(-0.35, 0.5, basins[i],
                                    horizontalalignment='right',
                                    verticalalignment='center',
                                    rotation='vertical',
                                    transform=ax.transAxes)

        ax = axes[i][1]
        df2 = data_proj[basin][scenario]
        sc2 = ax.scatter(df[variable_x], df[variable_y], c=df[scenario], **ckwargs,
                      marker='.', edgecolor='none', s=80)

    axes = axes.flatten()
    for kk in range(len(axes)):
        if kk in first_in_row:
            axes[kk].set_ylabel(variable_y_label)
        if kk in last_row:
            axes[kk].set_xlabel(variable_x_label)
    axes[0].set_title('Historic (1980-2005)')
    axes[1].set_title('Projected (2070-2100)')
    pos0 = axes[5].get_position()  # get the original position
    cax1 = fig.add_axes([pos0.x1 + 0.1, pos0.y0, 0.05, pos0.height * 2.5])
    cbar1 = fig.colorbar(sc, cax=cax1, orientation='vertical', label='Area (sq.km)')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.imshow
    plt.savefig(rf'..\figures\{variable_x}_vs_{variable_y}.png', dpi=300, bbox_inches="tight")
    plt.close()


variable_x = 'maxWS'
variable_y = 'maxRR'
variable_y_label = 'Max Rain\nRate m/hr'
variable_x_label = 'Max Wind\nSpeed m/s'
plot_please = True
if plot_please is True:
    nrow = 6
    ncol = 2
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)

    figsize = (6, 8)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize, tight_layout=True, sharex=True, sharey=True)
    for i in range(len((basins))):
        basin = basins[i]
        scenario = 'Compound'
        ckwargs = dict(cmap='Blues', norm=LogNorm())

        ax = axes[i][0]
        df = data[basin][scenario]
        sc = ax.scatter(df[variable_x], df[variable_y], c=df[scenario],  **ckwargs,
                     marker='.', edgecolor='none', s=80)
        ax.text(-0.35, 0.5, basins[i],
                                    horizontalalignment='right',
                                    verticalalignment='center',
                                    rotation='vertical',
                                    transform=ax.transAxes)

        ax = axes[i][1]
        df2 = data_proj[basin][scenario]
        sc2 = ax.scatter(df[variable_x], df[variable_y], c=df[scenario], **ckwargs,
                      marker='.', edgecolor='none', s=80)

    axes = axes.flatten()
    for kk in range(len(axes)):
        if kk in first_in_row:
            axes[kk].set_ylabel(variable_y_label)
        if kk in last_row:
            axes[kk].set_xlabel(variable_x_label)
    axes[0].set_title('Historic (1980-2005)')
    axes[1].set_title('Projected (2070-2100)')
    pos0 = axes[5].get_position()  # get the original position
    cax1 = fig.add_axes([pos0.x1 + 0.1, pos0.y0, 0.05, pos0.height * 2.5])
    cbar1 = fig.colorbar(sc, cax=cax1, orientation='vertical', label='Area (sq.km)')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.imshow
    plt.savefig(rf'..\figures\{variable_x}_vs_{variable_y}.png', dpi=300, bbox_inches="tight")
    plt.close()

