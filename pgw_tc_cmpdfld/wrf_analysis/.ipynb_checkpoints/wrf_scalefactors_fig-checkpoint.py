import os
import numpy as np
import xarray as xr
import pandas as pd
import hydromt
from hydromt import DataCatalog
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import colors, patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as ss
from scipy.optimize import curve_fit

# Script for reading WRF model output and writing to combined netcdf
# Author: Lauren Grimley


'''  Load WRF output and calculate wind speed '''
wd = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\wrf_analysis\scale_factors\scale_factors_carolinas'
os.chdir(wd)

var = 'precip'
bin_size = 5
upper_threshold = 100
thresholds = np.arange(0, 35, bin_size)
plt_label = 'Rain Rate\nLower Threshold\n(mm/hr)'
axis_label = 'Rain Rate (mm/hr)'

sf_threshold = []
for lower_threshold in thresholds:
    dir_name = f'{var}_thresh_{lower_threshold}_to_{upper_threshold}'
    file_name = f'{var}_thresh_{lower_threshold}_to_{upper_threshold}_scalefactors.csv'
    df = pd.read_csv(os.path.join(wd, dir_name, file_name), index_col=0)
    sf_threshold.append(df)

runs = ['ens1', 'ens2', 'ens4', 'ens4', 'ens5', 'ens6', 'ens7']
storms = ['florence', 'floyd', 'matthew']
runs_to_plot = []
for r in runs:
    for s in storms:
        runs_to_plot.append(f'{s}_{r}')
runs_to_plot.remove('matthew_ens7')

# Setup Plot Info
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
ax_ylim = [-0.2, 1]
nrow, ncol = 7, 3
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, axs = plt.subplots(nrows=nrow,
                        ncols=ncol,
                        figsize=(5.5,6),
                        tight_layout=True,
                        layout='constrained',
                        sharey=True, sharex=True)
axs = axs.flatten()
axs[-1].set_visible(False)
for i in range(len(runs_to_plot)):
    ax = axs[i]
    run_df = pd.DataFrame(columns=thresholds)
    for ii in range(len(thresholds)):
        d = sf_threshold[ii].loc[runs_to_plot[i]]
        run_df[thresholds[ii]] = d[d.index.isin(['mean', '5%', '50%', '99%'])]

    run_df.T.plot(ax=ax,
                  legend=False,
                  lw=2,
                  color=['black', 'lightcoral', 'deepskyblue', 'crimson'])

    ax.set_xlim((min(thresholds), max(thresholds)))
    ax.set_ylim(ax_ylim)
    ax.set_title('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(which='minor', linewidth=0.5, alpha=0.5, linestyle='--', color='grey', axis='both')
    ax.grid(which='major', linewidth=0.75, alpha=0.8, linestyle='-', color='darkgrey', axis='both')

    if i in first_in_row:
        ax.yaxis.set_visible(True)
        ax.set_ylabel('')
    if i in last_row:
        ax.xaxis.set_visible(True)
        ax.set_xlabel(plt_label)
    if i == 17:
        ax.xaxis.set_visible(True)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel(plt_label)

    legend_kwargs0 = dict(
        bbox_to_anchor=(1.6, -1.175),
        title=None,
        loc="lower center",
        frameon=True,
        prop=dict(size=10),
    )
    axs[-2].legend(**legend_kwargs0)
for i in range(ncol):
    axs[i].set_title(storms[i])
for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.25, 0.5, runs[i],
                              horizontalalignment='right',
                              verticalalignment='center',
                              rotation='vertical',
                              transform=axs[first_in_row[i]].transAxes)
plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.92)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(os.getcwd(),
                         f'scale_factors_all_storms.png'),
            bbox_inches='tight',
            dpi=255)
plt.close()
