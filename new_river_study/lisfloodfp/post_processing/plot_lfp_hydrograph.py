#!/usr/bin/env python3
# coding: utf-8
########################################################################################################
#    Author: Lauren E. Grimley
#    Contact: lgrimley@unc.edu
########################################################################################################
import pandas as pd
import datetime as dt
import dataretrieval.nwis as nwis
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import geopandas as gpd
import os
import csv
import xarray as xr


def get_station_data(file):
    with open(file, 'r') as input:
        i = 0
        for line in input:
            if line:
                if "Stage information (stage,x,y,elev):" in line:
                    start_index = i + 1
                if "Output, depths:" in line:
                    end_index = i - 1
                    break
                i += 1
    index_list = range(start_index, end_index, 1)
    info = pd.read_csv(file, sep='\t', skiprows=lambda x: x not in index_list, header=None, squeeze=True)
    info.columns = ['station', 'x_coord', 'y_coord', 'bed_elev']

    data = pd.read_csv(file, delim_whitespace=True, skiprows=end_index + 4, header=None, encoding='utf-8')
    col_num = range(1, len(data.columns), 1)
    col_names = ["Station_" + str(i) for i in col_num]
    info['station'] = col_names
    col_names.insert(0, 'seconds')
    data.columns = col_names
    return info, data


def get_timestamp(tstart, data):
    data['datetime'] = data['seconds'].apply(lambda x: tstart + pd.Timedelta(seconds=x))
    data.set_index('datetime', inplace=True)
    data = data.drop(columns='seconds')
    return data


def depth_to_stage(data, info):
    for i in range(len(info.station)):
        data[info.station[i]] = data[info.station[i]] + info.bed_elev[i]
    return data


wlts = xr.open_dataset('/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/NewRiver/sfincs/sfincs_his.nc')
wl = wlts.point_zs
sta = wl.isel(stations=11)
df = pd.DataFrame()
df['datetime'] = sta.coords['time'].values
df['var'] = sta.values
df.set_index('datetime',drop=True, inplace=True)
df.plot()

stage_filename = ['florence-m5.stage']

####
station_file = "/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/NewRiver/scripts/extract_points.csv"
dir = '/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/NewRiver/lisflood/florence/output'
fig_dir = '/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/'

# Florence
tstart = '2018-9-10'
tend = '2018-9-20'
# Dorian
#tstart = '2019-8-20'
#tend = '2019-9-20'
# Matthew
#tstart = '2016-9-20'
#tend = '2016-10-20'

# Gum Branch
pcode = '00065'
sites = '02093000'
datum = 1.51

# Retrieve USGS Data
usgs = nwis.get_record(sites=sites, service='iv', start=tstart, end=tend, parameterCd=pcode)

# Gage datum adjustment to get feet above NAVD88 and convert ft to meters
if usgs.empty is False:
    usgs['00065'] = (usgs['00065'] - datum) * 0.30480
gb_usgs = usgs
gb_usgs['date_utc'] = gb_usgs.index - pd.Timedelta(hours=4)

gb_usgs.reset_index(drop=True, inplace=True)
gb_usgs.set_index(gb_usgs['date_utc'], inplace=True)

# Old Jacksonville Bridge
pcode = '62620'
sites = '0209303201'
datum = 0

# Retrieve USGS Data
usgs = nwis.get_record(sites=sites, service='iv', start=tstart, end=tend, parameterCd=pcode)

# Gage datum adjustment to get feet above NAVD88 and convert ft to meters
if usgs.empty is False:
    usgs['62620'] = (usgs['62620'] - datum) * 0.30480
ojb_usgs = usgs

# ---------------------------------- #
d = {}
for i in range(len(stage_filename)):
    # Read Stage File
    stage_file = os.path.join(dir, stage_filename[i])
    info, data = get_station_data(stage_file)
    tstart = dt.datetime(2018, 9, 7, 0, 20)
    # tstart = dt.datetime(2019, 8, 25, 0, 15)
    # tstart = dt.datetime(2016, 9, 28, 12, 20)
    data = depth_to_stage(data=get_timestamp(tstart, data), info=info)
    d[i] = data

# Attached Site Name to Info
stations = pd.read_csv(station_file, header=0, sep=',')
info['site_name'] = stations['site_name']

# Adjust datetime axis for plotting
t1 = np.datetime64(dt.datetime(2018, 9, 12, 0, 0))
t2 = np.datetime64(dt.datetime(2018, 9, 18, 0, 0))
for i in range(len(d)):
    mask = (d[i].index > t1) & (d[i].index <= t2)
    d[i] = d[i].loc[mask]


# Plot station hydrographs
plt.rcParams.update({'font.size': 12})
mrks = 8
lws = 2

for i in range(len(info.site_name)):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    miny = []
    maxy = []
    evenly_spaced_interval = np.linspace(0, 1, len(stage_filename))
    #colors = [cm.gist_rainbow(x) for x in evenly_spaced_interval]
    colors = ['b', 'r', 'g', 'c', 'm', 'purple', 'orange', 'brown', 'y']
    for j in range(len(d)):
        mod = d[j][info.station[i]]
        miny.append(mod.min())
        maxy.append(mod.max())
        #ax.plot(mod.index, mod, color=colors[j], lw=2)

    miny = min(miny)
    if miny < -0.5:
        miny = -0.5
    maxy = max(maxy)

    if info.site_name[i] == 'HWY17 Bridge at Jacksonville, NC' and ojb_usgs.empty is False:
        #ax.plot(ojb_usgs.index, ojb_usgs['62620'], color='k', lw=0.5, alpha=0.5)
        ax.plot(ojb_usgs.index, ojb_usgs['62620'], '.', color='k', markersize=mrks)
        if min(ojb_usgs['62620']) < miny:
            miny = min(ojb_usgs['62620'])
        if max(ojb_usgs['62620']) > maxy:
            maxy = max(ojb_usgs['62620'])
    elif info.site_name[i] == 'Gum Branch' and gb_usgs.empty is False:
        #ax.plot(gb_usgs.index, gb_usgs['00065'], color='k', lw=0.5, alpha=0.5)
        ax.plot(gb_usgs.index, gb_usgs['00065'], '.', color='k', markersize=mrks)
        if min(gb_usgs['00065']) < miny:
            miny = round(min(gb_usgs['00065']))
        if max(gb_usgs['00065']) > maxy:
            maxy = max(gb_usgs['00065'])

    for j in range(len(d)):
        mod = d[j][info.station[i]]
        ax.plot(mod.index, mod, color=colors[j], lw=lws)
        print(j)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlim(t1, t2)
    #ax.set_ylim(round(miny) - 1.25, round(maxy) + 1.25)

    labels = ['USGS', 'LFP 30m DEM w/ Channels']
    ax.legend(labels, fancybox=True, framealpha=0.25)

    ax.set(ylabel='Water Level (m)', title=info.site_name[i])

    for item in [fig, ax]:
        item.patch.set_visible(False)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Only show ticks on the left and bottom spines
    #ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.grid(axis='y', color='white', linestyle='-', linewidth=1)
    plt.tight_layout()
    plt.savefig(fig_dir + info.site_name[i] + '_poster')
    #plt.close()



