import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel, utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors, patheffects
from string import ascii_lowercase as abcd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec


# Script for reading WRF model output and writing to combined netcdf
# Author: Lauren Grimley


def calc_windspd(da):
    wind_u = da['u_10m_gr']
    wind_v = da['v_10m_gr']
    wind_spd = ((wind_u ** 2) + (wind_v ** 2)) ** 0.5
    da['wndspd'] = wind_spd
    return da


cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

mod_root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\flor_ensmean_present'
mod = SfincsModel(mod_root, mode='r')
reg = mod.region.to_crs(epsg=4326)

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\met')
d1 = cat.get_rasterdataset('flor_ensmean_present.nc', geom=reg)
d2 = cat.get_rasterdataset('flor_ensmean_future.nc', geom=reg)
d1 = calc_windspd(d1)
d2 = calc_windspd(d2)

thresholds = [10, 30, 12.7, 25.4]
vars = ['wndspd', 'wndspd', 'precip_r', 'precip_r']
storm = 'flor'

for i in range(len(vars)):
    var = vars[i]
    threshold = thresholds[i]
    if var is 'wndspd':
        units = 'm/s'
    else:
        units = 'mm/hr'
    sf_mean = []
    sf_max = []
    for t in d1.time:
        d1s = d1[var].sel(time=t.values)
        d1s = d1s.where(d1s >= threshold, np.nan)
        d2s = d2[var].sel(time=t.values, method='nearest')
        d2s = d2s.where(d2s >= threshold, np.nan)
        da = ((d2s - d1s) / d1s).compute()
        sf_mean.append(da.mean(dim=['x', 'y']).item())
        sf_max.append(da.max(dim=['x', 'y']).item())

    sc_df = pd.DataFrame()
    sc_df['datetime'] = d1.time.values
    sc_df['sf_mean'] = sf_mean
    sc_df['sf_max'] = sf_max
    sc_df.set_index('datetime', inplace=True, drop=True)
    sc_df.fillna(0.0, inplace=True)
    sc_df = sc_df.round(decimals=3)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3), tight_layout=True)
    sc_df.plot(ax=ax, label=True, linewidth=2)
    ax.set_ylabel(f'Domain Averaged Scale Factor for\n{var} >= {threshold} {units}')
    ax.set_xlabel('')
    ax.set_title(f'WRF {storm} Ensemble Mean', loc='left')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(f'domain_avg_{var}_scaling_factor_{storm}_{threshold}.png', bbox_inches='tight', dpi=225)
    plt.close()

# Need to deal with cells that had no rain and now do have rain, infinity and nan

da = ((d2 - d1) / d1).compute()
var = 'precip_r'

da_sel = da[var].where(da[var] >= 25.4, np.nan)
da_mean = da_sel.mean(dim=['x', 'y'])
da_max = da_sel.max(dim=['x', 'y'])


da_mean = da_mean.fillna(0.0)
da_mean = da_mean.where(np.inf, 0.0)

fig, axs = plt.subplots(
    nrows=len(variables), ncols=1,
    figsize=(6, 8),
    tight_layout=True, sharex=True)
axs = axs.flatten()
i = 0
for var in variables:
    ax = axs[i]
    vmin = round(da_mean[var].min().item(), 0)
    vmax = round(da_mean[var].max().item(), 0)
    da_mean[var].plot(ax=ax, label='Domain Averaged')
    # da_max[var].plot(ax=ax, label='Domain Max')
    ax.set_ylim([vmin, vmax])
    ax.set_title('')
    i += 1

plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig('flor_future_present_wrf_diff2.png', bbox_inches='tight', dpi=225)
plt.close()

# PLOTTING TEMPORAL VARIATIONS IN VARIABLES AT POINT LOCATIONS
hdp_loc = cat.get_geodataframe(r'Z:\users\lelise\data\meteo\ncei_hpd\ncei_hpd_locations.shp', geom=reg)
var = 'precip_r'
ts_mdf = pd.DataFrame()
for i in range(len(hdp_loc)):
    pt = hdp_loc.iloc[i]
    ts = da.sel(x=pt.geometry.x.item(), y=pt.geometry.y.item(), method='nearest')
    ts_df = ts[var].to_dataframe()
    ts_df.fillna(0.0, inplace=True)
    ts_df = ts_df.replace([np.inf, -np.inf], 0.0)
    ts_df = ts_df.drop({'x', 'y', 'spatial_ref'}, axis=1)
    ts_df.columns = [(var + '_' + str(i).zfill(2))]
    ts_mdf = pd.concat([ts_mdf, ts_df], axis=1)
    if i == 10:
        break

plt_timeseries = True
if plt_timeseries is True:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), tight_layout=True, sharex=True)
    ts_mdf.plot(ax=ax, label=True)
    ax.set_yscale('log')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig('test.png', bbox_inches='tight', dpi=225)
    plt.close()

# PLOTTING SPATIAL DIFFERENCE IN MAX VARIABLES
data_vars = list(da.data_vars.keys())
da_max = da.max(dim='time')
variables = ['precip_g', 'precip_r', 'wndspd']
label = ['Total Precip Diff\n(mm)',
         'Max Rain Rate Diff\n(mm/hr)',
         'Max Wind Speed Diff\n(m/s)']
plot_title = 'Matthew Future minus Present WRF'
fileout = plot_title + '.png'

please_plot = True
if please_plot is True:
    fig, axs = plt.subplots(
        nrows=len(variables), ncols=1,
        figsize=(6, 8),
        tight_layout=True,
        sharex=True, sharey=True)
    axs = axs.flatten()
    i = 0
    for var in variables:
        ax = axs[i]
        vmin = round(da_max[var].min().item(), 0)
        vmax = round(da_max[var].max().item(), 0)
        # Plot difference in water level raster
        ckwargs = dict(cmap='seismic', vmin=vmin, vmax=vmax)
        cs = da_max[var].plot(ax=ax, add_colorbar=False, **ckwargs, zorder=0)
        reg.plot(ax=ax, color='none', edgecolor='lightgrey', linewidth=1, zorder=1, alpha=1)

        # Add colorbar
        pos0 = ax.get_position()
        cax = fig.add_axes([pos0.x1 + 0.05, pos0.y0 + pos0.height * 0.25, 0.03, pos0.height * 0.7])
        cbar = fig.colorbar(cs,
                            cax=cax,
                            orientation='vertical',
                            spacing='uniform',
                            shrink=0.7,
                            label=label[i],
                            extend='max',
                            aspect=30,
                            )
        ax.set_title('')
        ax.set_ylabel(f"Latitude")
        ax.yaxis.set_visible(True)
        ax.set_xlabel(f"Longitude")
        if i < len(variables) - 1:
            ax.xaxis.set_visible(False)
        else:
            ax.xaxis.set_visible(True)
        ax.ticklabel_format(style='plain', useOffset=False)
        minx, miny, maxx, maxy = reg.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        i += 1

    axs[0].set_title(plot_title, loc='center')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(fileout, dpi=225, bbox_inches="tight")
    plt.close()
