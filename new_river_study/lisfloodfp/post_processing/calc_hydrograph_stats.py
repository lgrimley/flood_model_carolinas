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


stage_filename = ['florence-m5.stage']

####
station_file = "/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/NewRiver/scripts/extract_points.csv"
dir = '/Users/laurengrimley/OneDrive - University of North Carolina at Chapel Hill/output/'
fig_dir = dir

# Florence
tstart = '2018-9-10'
tend = '2018-9-20'

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
ojb_usgs['date_utc'] = ojb_usgs.index

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


# Adjust datetime axis for plotting
t1 = np.datetime64(dt.datetime(2018, 9, 14, 0, 0))
t2 = np.datetime64(dt.datetime(2018, 9, 18, 0, 0))
for i in range(len(d)):
    mask = (d[i].index > t1) & (d[i].index <= t2)
    d[i] = d[i].loc[mask]


mod_id = 'Station_2'
meas = gb_usgs
var = '00065'

# Setup USGS timestamps
meas.reset_index(drop=True, inplace=True)
meas.index = pd.to_datetime(meas['date_utc']).dt.tz_localize(None)

mask = (meas.index > t1) & (meas.index <= t2)
obs = meas.loc[mask]

model_stats = pd.Series(data=d[0][mod_id], index=d[0].index)
model_stats_resample = model_stats.resample('30min').mean()

obs_stats = pd.Series(data=obs[var], index=obs.index)
obs_stats_resample = obs_stats.resample('30min').mean()
obs_stats_resample.drop(index=obs_stats_resample.index[-1], inplace=True)

# RMSE
rmse = ((obs_stats_resample-model_stats_resample) ** 2).mean() ** 0.5
print(rmse)

# Peak Error
pe = obs_stats_resample.max() - model_stats_resample.max()
print(pe)
pe_time = (obs_stats_resample.idxmax() - model_stats_resample.idxmax()).total_seconds()
print(pe_time/3600)




