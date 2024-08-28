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

# THIS SCRIPT COMPILES ALL THE FLOOD DEPTHS FOR EACH SCENARIO (E.G., STORM, CLIMATE) AND CREATES A DISTRIBUTION
# TO EXTRACT DEPTH STATISTICS. THIS CAN BE COMPARED TO TAKING THE MEAN ACROSS THE ENSEMBLE AT EACH CELL TO GET A
# MEAN DEPTH SURFACE... THEN FITTING THE DISTRIBUTION TO THE DEPTHS.

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\analysis')

depfile = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_wrf\subgrid\dep.tif'

root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_wrf\AGU2023\future_florence' \
       r'\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r')

da_zsmax = xr.open_dataarray(os.path.join(os.getcwd(), 'zsmax', f'pgw_zsmax.nc'))
da_zsmax_classified = xr.open_dataarray(os.path.join(os.getcwd(), 'driver_analysis', f'pgw_drivers_classified_all.nc'))

# Loop through the zsmax and calculate the depths for the water levels for each attributing process
runs = list(da_zsmax.run.values)
runs = list(filter(lambda x: 'compound' in x, runs))
da_hmax = []
event_ids = []
for run in runs:
    # remove the runs using ensemble mean scale factor
    if 'matt_fut_SF7' in run:
        continue
    if 'fut_SF8' in run:
        continue

    zsmax = da_zsmax.sel(run=run)
    r = run.replace('_compound', '')
    da_class = da_zsmax_classified.sel(run=r)

    for p in ['coastal', 'runoff', 'compound']:
        if p == 'coastal':
            mask = da_class == 1
        elif p == 'runoff':
            mask = da_class == 3
        elif p == 'compound':
            mask = ((da_class == 2) | (da_class == 4))

        hmax = utils.downscale_floodmap(
            zsmax=zsmax.where(mask),
            dep=mod.data_catalog.get_rasterdataset(depfile),
            hmin=0.05,
            gdf_mask=None,
            reproj_method='bilinear',
            # floodmap_fn='test_compound.tif'
        )
        da_hmax.append(hmax)
        event_ids.append(f'{r}_{p}')

da_hmax = xr.concat(da_hmax, dim='run')
da_hmax['run'] = xr.IndexVariable('run', event_ids)

runs_master = list(da_hmax.run.values)
df_dep_stats = pd.DataFrame()
df_bxplot_data = pd.DataFrame()
for scenario in ['coastal', 'runoff', 'compound']:
    for storm in ['flor', 'floy', 'matt']:
        for climate in ['pres', 'fut']:
            runs = [x for x in runs_master if scenario in x and storm in x and climate in x]
            if len(runs) > 1:
                hmax = da_hmax.sel(run=runs)
            else:
                run = runs[0]
                hmax = da_hmax.sel(run=run)

            df = hmax.to_dataframe().dropna(how='any', axis=0)
            df2 = pd.DataFrame(df['hmax'])
            df2.reset_index(inplace=True)
            df3 = pd.DataFrame(df2['hmax'])
            df3.columns = [f'{storm}_{climate}_{scenario}']
            df_bxplot_data = pd.concat([df_bxplot_data, df3],
                                       ignore_index=False, axis=1)

            df_stats = df3.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            df_dep_stats = pd.concat([df_dep_stats, df_stats], ignore_index=False, axis=1)

df_dep_stats.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\analysis\driver_analysis'
                    r'\depths_by_driver_all_runs.csv')

# Plotting info
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

scenarios = ['coastal', 'runoff', 'compound']
nrow = 3
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

fig, axes = plt.subplots(nrows=nrow, ncols=ncol, tight_layout=True, figsize=(5, 5))
axes = axes.flatten()
for i in range(len(scenarios)):
    scenario = scenarios[i]
    runs = [x for x in df_bxplot_data.columns if scenario in x]
    ax = axes[i]
    df_plot = df_bxplot_data[runs]
    bp = df_plot.boxplot(ax=ax,
                         # by='group',
                         # column=scenarios[counter],
                         vert=True,
                         color=props,
                         boxprops=boxprops,
                         flierprops=flierprops,
                         medianprops=medianprops,
                         meanprops=meanpointprops,
                         meanline=False,
                         showmeans=True,
                         patch_artist=True,
                         # layout=(3, 1),
                         zorder=1
                         )
plt.savefig('test.png', bbox_inches='tight', dpi=255)
plt.close()

boxplot_total_area = True
if boxplot_total_area is True:
    for ax in axes:
        print(counter)
        bp = da_plot.boxplot(ax=ax,
                             by='group',
                             column=scenarios[counter],
                             )
        ax.scatter(x=ax.get_xticks(), y=pres_ensmean[scenarios[counter]].values,
                   s=30, color='red', marker='X', zorder=2, edgecolor='black', alpha=0.9)
        if counter == 3:
            ax.set_ylim(74500, 78500)

        if counter in last_row:
            xtick = ax.get_xticks()
            ax.set_xticklabels(['Flor (n=35)',
                                'Floy (n=35)',
                                'Matt (n=30)'])
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_title(scenarios[counter])
        ax.set_xlabel(None)
        ax.set_ylabel('Flooded Area\n(sq.km.)')
        kwargs = dict(linestyle='--', linewidth=1, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='major', axis='y', zorder=0, **kwargs)
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.8)
        ax.grid(visible=True, which='minor', axis='y', zorder=0, **kwargs)
        ax.set_axisbelow(True)
        counter += 1

    plt.suptitle(None)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig('driver_fldArea_boxplot_ensmean.png', bbox_inches='tight', dpi=255)
    plt.close()
