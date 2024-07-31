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

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\final_model\process_attribution\20m')
buildings = pd.read_csv(r'building_pts_with_depth_floodedOnly.csv', index_col=0)
gdf = buildings.copy()
gdf = gdf[gdf['hmax'] > 0.15]
gdf['summed'] = gdf['hmax_coastal'].fillna(0) + gdf['hmax_runoff'].fillna(0)

gdf_clean = gdf[(gdf['hmax_class'] == 2) | (gdf['hmax_class'] == 4)]
gdf_clean = gdf_clean[['hmax', 'hmax_coastal', 'hmax_runoff', 'summed', 'Name']]
gdf_clean.columns = ['Compound', 'Coastal', 'Runoff', 'Summed', 'Name']
gdf_clean.set_index('Name', inplace=True, drop=True)

# PLOTTING
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
nrow = 6
ncol = 4
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)
nbins = 10
basins = ['Lower Pee Dee', 'Cape Fear', 'Onslow Bay', 'Neuse', 'Pamlico', 'Domain']
process = ['Runoff', 'Coastal', 'Summed', 'Compound']

fig, axs = plt.subplots(figsize=(6.2, 8), ncols=ncol, nrows=nrow,
                        tight_layout=True, layout='constrained',
                        sharey=True, sharex=True)
for i in range(nrow):
    k = first_in_row[i]
    if i == 5:
        dsp = gdf_clean
    else:
        dsp = gdf_clean[gdf_clean.index == basins[i]]

    for ii in range(ncol):
        dsp1 = dsp[process[ii]]
        dsp1.dropna(axis=0, how='any', inplace=True)

        ax = axs[i][ii]
        data, bins, _ = ax.hist(dsp1, bins=nbins,
                                range=[0, 3],
                                density=True,
                                histtype='stepfilled',
                                color='grey',
                                alpha=0.8,
                                )
        ax.set_ylim(0, 2)
        ax.axvline(x=dsp1.mean(), color='black', linestyle='--')
        ax.axvline(x=dsp1.median(), color='black')
        start, end = ax.get_ylim()
        startx, endx = ax.get_xlim()
        ax.text(x=endx - 0.05, y=end - 0.30, s=f'Mean:{np.round(dsp1.mean(), 2)}', ha='right', va='bottom', fontsize=10)
        ax.text(x=endx - 0.05, y=end - 0.55, s=f'Median:{np.round(dsp1.median(), 2)}', ha='right', va='bottom',
                fontsize=10)
        ax.text(x=endx - 0.05, y=end - 0.80, s=f'n={len(dsp1)}', ha='right', va='bottom', fontsize=10)

axs = axs.flatten()
counter = 0
for i in first_row:
    axs[i].set_title(f'{process[counter]}', loc='center')
    counter += 1
for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.15, 0.5, basins[i],
                              horizontalalignment='right',
                              verticalalignment='center',
                              rotation='vertical',
                              transform=axs[first_in_row[i]].transAxes)
for i in last_row:
    axs[i].set_xlabel('Depths (m)')
plt.margins(x=0, y=0)
plt.savefig('histogram_of_depths_at_buildings_within_compoundExtent.png',
            bbox_inches='tight', dpi=255)
plt.close()

summed = np.array([0.74,0.94,0.66,0.85,0.64,0.75])
com = np.array([0.78,0.89,0.73,0.88,0.7,0.8])
dif = np.subtract(com, summed)