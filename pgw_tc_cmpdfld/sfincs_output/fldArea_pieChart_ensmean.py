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

# Getting present info
work_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs'
fld_area = pd.read_csv(os.path.join(work_dir,
                                    'analysis',
                                    'driver_analysis',
                                    'pgw_drivers_classified_all_area.csv'),
                       index_col=0)
fld_area_pres = fld_area.round(2)
fld_area_pres.drop(columns=['no_flood', 'compound_coastal', 'compound_runoff'], inplace=True)
fld_area_pres.columns = ['Coastal', 'Runoff', 'Compound']

# Getting ensmean future data
work_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs'
fld_da = xr.open_dataset(
    os.path.join(work_dir, 'analysis', 'driver_analysis', 'pgw_drivers_classified_ensmean_mean.nc'))
fld_area_fut = pd.DataFrame()
for run in fld_da.run.values.tolist():
    ds = fld_da.sel(run=run)
    unique_codes, fld_cells = np.unique(ds[list(ds.keys())[0]].data, return_counts=True)
    fld_area = fld_cells.copy()
    res = 200  # meters
    fld_area = fld_area * (res * res) / (1000 ** 2)  # square km
    fld_area = pd.DataFrame(fld_area).T
    fld_area.columns = ['no_flood', 'coastal', 'compound_coastal', 'runoff', 'compound_runoff']
    fld_area_fut = pd.concat([fld_area_fut, fld_area], axis=0, ignore_index=True)
fld_area_fut.index = ['flor_fut', 'floy_fut', 'matt_fut']
fld_area_fut['compound'] = fld_area_fut['compound_coastal'] + fld_area_fut['compound_runoff']
fld_area_fut = fld_area_fut.round(2)
fld_area_fut.drop(columns=['no_flood', 'compound_coastal', 'compound_runoff'], inplace=True)
fld_area_fut.columns = ['Coastal', 'Runoff', 'Compound']

combined = pd.concat([fld_area_pres, fld_area_fut])
runs_to_plot = ['flor_pres', 'flor_fut',
                'floy_pres', 'floy_fut',
                'matt_pres', 'matt_fut']
combined = combined[combined.index.isin(runs_to_plot)]
pie_scale = combined.sum(axis=1) / 70000
storms = ['Florence', 'Floyd', 'Matthew']

# Setup Plot Info
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
colors = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897']
nrow, ncol = 3, 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, axs = plt.subplots(nrows=nrow,
                        ncols=ncol,
                        figsize=(3, 4),
                        tight_layout=True,
                        layout='constrained'
                        )
axs = axs.flatten()
for i in range(len(runs_to_plot)):
    ax = axs[i]
    d = combined[combined.index == runs_to_plot[i]]
    ax.pie(d.to_numpy()[0],
           colors=colors,
           radius=pie_scale[pie_scale.index == runs_to_plot[i]][0],
           startangle=90)
for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.1, 0.5, storms[i],
                              horizontalalignment='right',
                              verticalalignment='center',
                              rotation='horizontal',
                              transform=axs[first_in_row[i]].transAxes)
axs[0].set_title('Present')
axs[1].set_title('Future')
legend_kwargs0 = dict(
    bbox_to_anchor=(2.3, 0.75),
    title=None,
    loc="upper right",
    frameon=True,
    prop=dict(size=10),
)
axs[3].legend(labels=combined.columns, **legend_kwargs0)
plt.subplots_adjust(wspace=0.0, hspace=0.05)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(work_dir, f'fld_area_pieChart_ensmean.png'), bbox_inches='tight', dpi=255, tight_layout=True)
plt.close()
