#!/usr/bin/env python3
# coding: utf-8
#
# -------------- Script Overview -------------------
#
# Mapping the difference in water levels at locations
#
#   Author: Lauren Grimley, lauren.grimley@unc.edu
#   Last edited by: LEG 2/6/24
#
# Inputs:
#   Netcdf of water level timeseries (format compatible with hydromt)
#     * to convert ADCIRC fort.61.nc use script: adcirc_fort61_2netcdf.py
#
# Outputs:
#   Figure of water level difference stats at points
#
# Dependencies:
#
# Needed updates:
#   Generalize index selection (e.g., shapefile or SFINCS mask, none)
#   Model columns and naming lines 59-79 

import os
import datetime
import hydromt
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# Filepath to data catalog yml
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
cat = hydromt.DataCatalog(yml)

# mod = cat.get_geodataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs'
#                          r'\hydromt_sfincs_input\hindcast_data\waterlevel\flor_FLRA.nc')
# for sta in obs['station'].values:
#     obs_sel = obs.sel(station=sta)
#     obs_x = obs_sel.x.values.item()
#     obs_y = obs_sel.y.values.item()
#
#     mod_x = mod.x.values
#     mod_y = mod.y.values
#
#     # First, find the index of the grid point nearest a specific lat/lon.
#     absx = np.abs(mod_x - obs_x)
#     absy = np.abs(mod_y - obs_y)
#     c = np.maximum(absx, absy)
#     pt_index = np.argmin(c)
#     print(pt_index)
#
#     # Model station for comparison
#     mod_sel = mod.sel(index=pt_index)
#
#     # Convert to dataframes and merge
#     mod_wl = mod_sel.to_dataframe()
#     obs_wl = obs_sel.to_dataframe()
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\hydromt_sfincs_input\hindcast_data\waterlevel')
gages_sel = pd.DataFrame()
gages_sel['station'] = [8651370, 8661070, 8658163, 8656483,8652587]
gages_sel['mod_index'] = [
    5, # Duck, 8651370
    149, # Springmaid, 8661070
    107, #Wrightsville, 8658163
    66, # Beaufort, 8656483
    23, # Oregon Inlet, 8652587
]
adcirc_output = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\adcirc_output'
obs_folder = r'Z:\Data-Expansion\users\lelise\data\storm_data\hurricanes'
for sta in gages_sel['station'].values:
    if sta in [8658163, 8720030, 8661070]:
        storms = ['matt','flor']
        fs = (5, 4)
    else:
        storms = ['floy', 'matt','flor']
        fs = (5, 6)

    dplot = []
    tt = []
    for storm in storms:
        print(storm)
        if storm == 'flor':
            mod = xr.open_dataset(os.path.join(adcirc_output, 'flor_FLRA', 'fort.61.nc'))
            f = os.path.join(obs_folder, '2018_florence', 'waterlevel', 'noaa_florence_msl.nc')
            obs = cat.get_geodataset(data_like=f, crs=4326, time_tuple=(mod.time[0].values, mod.time[-1].values))
            tmin = datetime.datetime(2018, 9, 13, 0, 0, 0)
            tmax = datetime.datetime(2018, 9, 16, 0, 0, 0)
        elif storm == 'matt':
            mod = xr.open_dataset(os.path.join(adcirc_output, 'matt_OWI', 'fort.61.nc'))
            f = os.path.join(obs_folder, '2016_matthew', 'waterlevel', 'noaa_matthew_msl.nc')
            obs = cat.get_geodataset(data_like=f, crs=4326, time_tuple=(mod.time[0].values, mod.time[-1].values))
            tmin = datetime.datetime(2016, 10, 8, 0, 0, 0)
            tmax = datetime.datetime(2016, 10, 10, 12, 0, 0)
        elif storm == 'floy':
            mod = xr.open_dataset(os.path.join(adcirc_output, 'floy_ERA5', 'fort.61.nc'))
            f = os.path.join(obs_folder, '1999_floyd', 'waterlevel', 'noaa_floyd_msl.nc')

            tmin = datetime.datetime(1999, 9, 15, 12, 0, 0)
            tmax = datetime.datetime(1999, 9, 18, 12, 0, 0)
            obs = cat.get_geodataset(data_like=f, crs=4326, time_tuple=(tmin, tmax))

        tt.append([tmin, tmax])

        # Get observed water levels
        obs_sel = obs.sel(station=sta)
        print(sta)
        # Model station for comparison
        pt_index = gages_sel[gages_sel['station'] == sta]['mod_index'].values[0]
        mod_sel = mod.sel(station=pt_index)
        print(mod_sel['station_name'].values)

        # Convert to dataframes and merge
        mod_wl = mod_sel.to_dataframe()
        mod_wl = mod_wl['zeta'].resample('30min').mean()

        obs_wl = obs_sel.to_dataframe()
        obs_wl = obs_wl['waterlevel'].resample('30min').mean()

        dplot.append([mod_wl, obs_wl])

    fig, axs = plt.subplots(nrows=len(storms), ncols=1,
                            figsize=fs,
                            tight_layout=True,
                            sharey= True, sharex=False)
    axs = axs.flatten()
    for i in range(len(storms)):
        ax = axs[i]
        mod_wl, obs_wl = dplot[i]
        obs_wl.plot(ax=ax, color='black', linewidth=1.5, linestyle='--', label='NOAA')
        mod_wl.plot(ax=ax, linestyle='-', linewidth=1.5, alpha=1, label='ADCIRC')
        ax.set_xlim(tt[i])
        ax.set_ylabel('Water Level\n(m+MSL)')
        ax.set_xlabel('')
        ax.set_title(f'{storms[i]}: {sta}', loc='center', fontsize=10)
        ax.grid(axis='y')
        ax.set_axisbelow(True)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    fileout = str(obs_sel['station'].item()) + '.png'
    plt.savefig(fileout, bbox_inches='tight', dpi=225)
    plt.close()
