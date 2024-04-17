import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel, utils
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors, patheffects
from string import ascii_lowercase as abcd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import scipy.stats as ss
import seaborn as sns
from scipy.optimize import curve_fit


# Script for reading WRF model output and writing to combined netcdf
# Author: Lauren Grimley


def calc_windspd(da):
    wind_u = da['wind_u']
    wind_v = da['wind_v']
    wind_spd = ((wind_u ** 2) + (wind_v ** 2)) ** 0.5
    da['wind_spd'] = wind_spd
    return da


def get_xy_coord(da, var, sel_type, outshp_filepath=None):
    ds = da[var]

    # Get the x,y index of the variable min/max at each timestep for both da1 and da2
    if sel_type == 'min':
        ind_x = ds.min(dim="y").idxmin(dim="x")
        ind_y = ds.min(dim="x").idxmin(dim="y")

    elif sel_type == 'max':
        ind_x = ds.max(dim="y").idxmax(dim="x")
        ind_y = ds.max(dim="x").idxmax(dim="y")

    x_coords = ind_x.data.compute()
    y_coords = ind_y.data.compute()

    gdf = gpd.GeoDataFrame(ds['time'].data,
                           geometry=gpd.points_from_xy(x_coords, y_coords, crs=4326))

    if outshp_filepath:
        gdf.to_file(outshp_filepath)

    return x_coords, y_coords, gdf


def calculate_dx_dy(x1, y1, x2, y2):
    # Calculate the change in x,y
    dx = (x1 - x2)
    dy = (y1 - y2)

    # Fill nan with zero
    dx[np.isnan(dx)] = 0
    dy[np.isnan(dy)] = 0

    return dx, dy


def interpolate_shifted_data_to_grid(da, dx, dy):
    # Apply shift at each time step
    ii = 0
    interp_da_list = []
    for t in da.time:
        da_t = da.sel(time=t, method='nearest')

        # Coords for original grid
        orig_x = da_t['x'].data
        orig_y = da_t['y'].data

        # Coords for shifted grid
        shifted_x = (orig_x + dx[ii])
        shifted_y = (orig_y + dy[ii])

        # Create a new dataset with the data assigned to the shifted coordinates
        ds = xr.Dataset(
            data_vars=dict(
                wind_u=(["y", "x"], da_t['wind_u'].data),
                wind_v=(["y", "x"], da_t['wind_v'].data),
                precip_g=(["y", "x"], da_t['precip_g'].data),
                mslp=(["y", "x"], da_t['mslp'].data),
                precip=(["y", "x"], da_t['precip'].data),
                wind_spd=(["y", "x"], da_t['wind_spd'].data),
            ),
            coords=dict(
                x=(["x"], shifted_x),
                y=(["y"], shifted_y),
            ),
            attrs=da.attrs,
        )

        # Interpolate the data back to the original grid at timestep t
        interp_da = ds.interp(coords=dict(x=(["x"], orig_x), y=(["y"], orig_y)), method='linear')
        interp_da_list.append(interp_da)
        ii += 1

    da_interp_shifted = xr.concat(interp_da_list, 'time')

    return da_interp_shifted


def shift_wrf_grid_to_mslp(da1, da2):
    # Some wrf runs have a different number of time steps (??)
    if len(da2.time) != len(da1.time):
        print('WRF times dont match! Clipping.')
        da2 = da2.sel(time=slice(da1.time.values.min(), da1.time.values.max()))

    # Get location of minimum pressure
    x1, y1, gdf1_minslp = get_xy_coord(da=da1, var='mslp', sel_type='min')
    x2, y2, gdf2_minslp = get_xy_coord(da=da2, var='mslp', sel_type='min')

    # Calculate the shift in x, y
    dx, dy = calculate_dx_dy(x1, y1, x2, y2)

    # Shift the data and interpolate back to original grid
    da2_shifted_mslp = interpolate_shifted_data_to_grid(da=da2, dx=dx, dy=dy)

    return da2_shifted_mslp


font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

'''  LOAD Data '''
cat_dir = r'Z:\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog.yml')
cat = hydromt.DataCatalog(yml)

# Load WRF output and calculate wind speed
os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_input\met')
storms = ['floyd', 'matthew', 'florence']
climates = ['present', 'future']

runs_dict = {'floyd': {'present': {}, 'future': {}, 'future_shifted_mslp': {}},
             'matthew': {'present': {}, 'future': {}, 'future_shifted_mslp': {}},
             'florence': {'present': {}, 'future': {}, 'future_shifted_mslp': {}}}
for storm in storms:
    for climate in climates:
        met_dir = f'{climate}_{storm}'
        for file in os.listdir(os.path.join(os.getcwd(), met_dir)):
            if file.endswith('.nc'):
                run_name = file.split('.')[0].split('_')[-1]
                run_filepath = os.path.join(os.getcwd(), met_dir, file)

                # Read in netcdf of WRF output, calculate wind speed
                da = cat.get_rasterdataset(run_filepath)
                da = calc_windspd(da)

                # Add data to dictionary
                runs_dict[storm][climate][f'{run_name}'] = da
                print(run_filepath)

'''  Shifting future track '''
shift_future_tracks = False
if shift_future_tracks is True:
    for storm in storms:
        for climate in climates:
            runs = list(runs_dict[storm][climate].keys())
            for run in runs:
                da_present = runs_dict[storm]['present'][run]
                da_future = runs_dict[storm]['future'][run]
                da_future_shifted = shift_wrf_grid_to_mslp(da1=da_present, da2=da_future)
                runs_dict[storm]['future_shifted_mslp'][f'{run}'] = da_future_shifted
                print(run)

'''  PLOTTING Shifted Tracks  '''
state_boundaries = cat.get_geodataframe(
    r'Z:\users\lelise\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
state_boundaries.to_crs(epsg=4326, inplace=True)
aoi = state_boundaries[state_boundaries['NAME'].isin(['South Carolina', 'North Carolina',
                                                      'Georgia', 'Virginia', 'Florida'])]
state_boundaries.set_index('NAME', inplace=True)
aoi_model = state_boundaries[state_boundaries.index.isin(['South Carolina', 'North Carolina'])]

plot_please = False
if plot_please:
    # Plotting
    variables = ['min_pressure', 'max_wind', 'max_rain_rate', 'total_precip']
    for v in variables:
        minx, miny, maxx, maxy = aoi_model.buffer(0.2).total_bounds
        figsize = (6, 12)

        if v == 'max_rain_rate':
            var = 'precip'
            var_label = 'Max Rain Rate (mm/hr)'
            var_bounds = np.arange(20, 100, 5)

        elif v == 'total_precip':
            var = 'precip'
            var_label = 'Total Precip (mm)'
            var_bounds = np.arange(100, 950, 50)

        elif v == 'min_pressure':
            var = 'mslp'
            var_label = 'Minimum Pressure (mbar)'
            var_bounds = np.arange(930, 1010, 5)
            # minx, miny, maxx, maxy = [-85.0, 25.0, -73.0, 38.0]

        elif v == 'max_wind':
            var = 'wind_spd'
            var_label = 'Max Wind Speeds (m/hr)'
            var_bounds = np.arange(10, 60, 5)

        for storm in storms:
            climates = list(runs_dict[storm].keys())
            runs = list(runs_dict[storm][climates[0]].keys())

            # Info for controling subplots
            nrow, ncol = [len(runs), len(climates)]
            n_subplots = nrow * ncol
            first_in_row = np.arange(0, n_subplots, ncol)
            last_in_row = np.arange(ncol - 1, n_subplots, ncol)
            first_row = np.arange(0, ncol - 1)
            last_row = np.arange(ncol, n_subplots, 1)

            # Create plot
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                                    figsize=figsize, tight_layout=True,
                                    layout='constrained', sharex=True, sharey=True)

            for k in range(nrow):
                run = runs[k]
                for i in range(ncol):
                    climate = climates[i]
                    ax = axs[k, i]
                    cmap = mpl.cm.jet
                    norm = mpl.colors.BoundaryNorm(var_bounds, cmap.N, extend='max')

                    data_for_plot = runs_dict[storm][climate][run]
                    if v in ['max_rain_rate', 'max_wind']:
                        data_for_plot = data_for_plot[var].max(dim='time')
                    elif v == 'total_precip':
                        data_for_plot = data_for_plot[var].sum(dim='time')
                    elif v == 'min_pressure':
                        data_for_plot = data_for_plot[var].min(dim='time')

                    data_for_plot.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=2, alpha=0.8)
                    ax.set_title('')
                    if k == 0:
                        ax.set_title(climate, loc='center')

                axs[k, 0].text(-0.05, 0.5, run,
                               horizontalalignment='right',
                               verticalalignment='center',
                               rotation='vertical',
                               transform=axs[k, 0].transAxes)

            axs = axs.flatten()
            for ii in range(len(axs)):
                axs[ii].set_xlim(minx, maxx)
                axs[ii].set_ylim(miny, maxy)
                state_boundaries.plot(ax=axs[ii], color='none', edgecolor='black', linewidth=1, linestyle='-', zorder=2)

                axs[ii].yaxis.set_visible(False)
                axs[ii].xaxis.set_visible(False)

            # Plot colorbar
            pos1 = axs[-1].get_position()  # get the original position
            cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 3, 0.03, pos1.height * 3.4])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cb = fig.colorbar(sm,
                              cax=cbar_ax,
                              shrink=2,
                              extend='both',
                              spacing='uniform',
                              label=var_label,
                              pad=0,
                              aspect=30)

            # Save and close plot
            plt.subplots_adjust(wspace=0.02, hspace=0.05, top=0.94)
            plt.margins(x=0, y=0)
            plt.suptitle(storm)
            plt.savefig(os.path.join(os.getcwd(), f'{storm}_meteo_comparison_{v}.png'),
                        bbox_inches='tight', dpi=255)
            plt.close()