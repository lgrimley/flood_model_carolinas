import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np

warnings.filterwarnings('ignore')  # setting ignore as a parameter


# Lauren Grimley
# Last updated on 1/27/2023

def cleanup_cocorahs_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['val'] = df['val'].astype(str)
    df['val'] = df['val'].replace({'\*': '', '\-': '', 'T': ''}, regex=True)
    df['val'] = df['val'].apply(lambda x: x.strip())
    df['val'] = df['val'].replace('', np.nan, regex=True)
    df.dropna(how='any', axis='index', inplace=True)
    df['val'] = df['val'].astype(float)
    df.set_index('date', inplace=True, drop=True)
    return df


# Change working directory
os.chdir(r'C:\Users\lelise\University of North Carolina at Chapel Hill\SunnyD Flood Sensor Network - General\Flood '
         r'Protocol\New Bern')

# Load sensor water level data
df = pd.read_csv('NB_02_2022-01-01_2022-12-31.csv')
df['datetime'] = pd.to_datetime(df['date'])
df.set_index('datetime', inplace=True, drop=True)

# Load precipitation data
precip = pd.read_csv('CoCoRaHS_NC-CN-9.csv')
precip.columns = ['date', 'val']
precip = cleanup_cocorahs_data(precip)

# Subset precipitation data (if needed)
tstart = '2022-01-01'
tend = '2022-12-31'
df = df[(df.index > tstart) & (df.index < tend)]
dfp = precip[(precip.index > tstart) & (precip.index < tend)]

# Update road and sensor elevations for site
sensor_elev = 0.75
road_elev = 1.3

# Plot the hydrograph and hyetograph
fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))
ax2 = ax.twinx()
ax.plot(df.index, df['sensor_water_level_adj'], color='black')
ax.set_ylabel('Water Level (ft+NAVD88)')
ax.set_ylim(0.5, 5)
ax.set_xlim(datetime.strptime(tstart, '%Y-%m-%d'), datetime.strptime(tend, '%Y-%m-%d'))
ax.hlines(y=sensor_elev, xmin=min(df.index), xmax=max(df.index), color='red',
          linestyle='--', linewidth=2, alpha=0.3)
ax.hlines(y=road_elev, xmin=min(df.index), xmax=max(df.index), color='grey',
          linestyle='--', linewidth=2, alpha=0.3)
ax2.bar(dfp.index, dfp['val'], color='blue', width=2, alpha=0.5, align='edge')
ax2.set_ylabel('Daily Precipitation (in)')
ax2.set_ylim(0, 8)
ax2.invert_yaxis()
ax.set_xlim(min(df.index), max(df.index))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('NB_02_WL_Rain')
plt.close()

# Load sub-daily precip
p = pd.read_csv(r'./Rainfall/PV1XX1G2_1.csv', skiprows=15, skip_blank_lines=True)
p['datetime_est'] = pd.to_datetime(p[p.columns[0]], utc=False)
#p['datetime_utc'] = pd.to_datetime(p[p.columns[0]]) + pd.Timedelta(hours=-5)
p.drop(p.columns[[0, 2, 3]], axis=1, inplace=True)
p.set_index('datetime_est', drop=True, inplace=True)
p[p.select_dtypes(object).apply(lambda row: row.str.contains("M"), axis=1).any(axis=1)] = np.nan
p[p.select_dtypes(object).apply(lambda row: row.str.contains("Q"), axis=1).any(axis=1)] = np.nan
p[p.columns[0]] = p[p.columns[0]].astype(float)
p = p[p.index.notna()]

daily_p = p.resample('D').sum()
storms = daily_p[daily_p[p.columns[0]] > 0]
storms.to_csv(r'C:\Users\lelise\University of North Carolina at Chapel Hill\SunnyD Flood Sensor Network - General\Flood Protocol\New Bern\Rainfall\storms_est.csv')

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))
ax.bar(daily_p.index, daily_p[p.columns[0]], color='blue', width=2, alpha=0.5, align='edge')
ax.hlines(y=12.7, xmin=min(daily_p.index), xmax=max(daily_p.index), color='red',
          linestyle='--', linewidth=2, alpha=0.3)
ax.set_xlim(min(daily_p.index), max(daily_p.index))
ax.set_ylabel('Daily Precipitation (mm)')
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('New Bern 2022 Daily Rainfall (mm).png')
plt.close()