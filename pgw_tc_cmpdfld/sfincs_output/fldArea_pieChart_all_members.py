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

work_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs'
fld_area = pd.read_csv(os.path.join(work_dir,
                                    'analysis',
                                    'driver_analysis',
                                    'pgw_drivers_classified_all_area.csv'),
                       index_col=0)
fld_area = fld_area.round(2)
fld_area.drop(columns=['no_flood', 'compound_coastal', 'compound_runoff'], inplace=True)
fld_area.columns = ['Coastal', 'Runoff', 'Compound']

slr = pd.read_csv(os.path.join(work_dir, 'hindcast_slr_event_ids.csv'),
                  index_col=0, header=None)
slr.columns = ['SLR']
slr = slr.round(2)

runs_to_plot = []
storm = 'matt'
for id in fld_area.index:
    if storm in id.split('_'):
        runs_to_plot.append(id)
# runs_to_plot.remove(f'{storm}_pres')
runs = ['Pres', 'SF1', 'SF2', 'SF3', 'SF4', 'SF5', 'SF6', 'SF7']
if storm == 'matt':
    runs.remove('SF7')
pres_tot_area = fld_area[fld_area.index == runs_to_plot[0]].sum(axis=1)
pie_scale = fld_area.sum(axis=1) / pres_tot_area[0]

# Setup Plot Info
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
colors = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897']
nrow, ncol = len(runs), 5
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, axs = plt.subplots(nrows=nrow,
                        ncols=ncol,
                        figsize=(6, 8),
                        tight_layout=True,
                        layout='constrained'
                        )
axs = axs.flatten()
axs[1].set_visible(False)
axs[2].set_visible(False)
axs[3].set_visible(False)
axs[4].set_visible(False)
for i in range(len(runs_to_plot)):
    if i == 0:
        ax = axs[0]
        d = fld_area[fld_area.index == runs_to_plot[i]]
        ax.pie(d.to_numpy()[0],
               radius=pie_scale[i],
               startangle=90,
               colors=colors
               )

        legend_kwargs0 = dict(
            bbox_to_anchor=(3.75, 1.2),
            title=None,
            loc="upper right",
            frameon=True,
            prop=dict(size=10),
        )
        ax.legend(labels=fld_area.columns, **legend_kwargs0)
    else:
        ax = axs[i + 4]
        d = fld_area[fld_area.index == runs_to_plot[i]]
        ax.pie(d.to_numpy()[0],
               colors=colors,
               radius=pie_scale[i],
               startangle=90)
        slr_amount = slr[slr.index == runs_to_plot[i]].values.item()
        ax.set_title(f'SLR:{slr_amount}m')

for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.1, 0.5, runs[i],
                              horizontalalignment='right',
                              verticalalignment='center',
                              rotation='horizontal',
                              transform=axs[first_in_row[i]].transAxes)

plt.subplots_adjust(wspace=0.0, hspace=0.3)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(work_dir, f'{storm}_fld_area_pieChart.png'), bbox_inches='tight', dpi=255, tight_layout=True)
plt.close()
