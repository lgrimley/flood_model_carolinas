import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np

warnings.filterwarnings('ignore')  # setting ignore as a parameter


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


os.chdir(r'Z:\users\lelise\projects\SunnyD\NB_2022')

df = pd.read_csv('NB_02_2022-01-01_2022-12-31.csv')
df['datetime'] = pd.to_datetime(df['date'])
df.set_index('datetime', inplace=True, drop=True)

precip = pd.read_csv('CoCoRaHS_NC-CN-9.csv')
precip.columns = ['date', 'val']
precip = cleanup_cocorahs_data(precip)

tstart = '2022-01-01'
tend = '2022-12-31'
df = df[(df.index > tstart) & (df.index < tend)]
dfp = precip[(precip.index > tstart) & (precip.index < tend)]

# sensor_elev = 0.75  # 1.89
# road_elev = 1.3  # 4.65
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))
# ax2 = ax.twinx()
# ax.plot(df.index, df['sensor_water_level_adj'], color='black')
# ax.set_ylabel('Water Level (ft+NAVD88)')
# ax.set_ylim(0.5, 5)
# ax.set_xlim(datetime.strptime(tstart, '%Y-%m-%d'), datetime.strptime(tend, '%Y-%m-%d'))
# ax.hlines(y=sensor_elev, xmin=min(df.index), xmax=max(df.index), color='red',
#           linestyle='--', linewidth=2, alpha=0.3)
# ax.hlines(y=road_elev, xmin=min(df.index), xmax=max(df.index), color='grey',
#           linestyle='--', linewidth=2, alpha=0.3)
# ax2.bar(dfp.index, dfp['val'], color='blue', width=2, alpha=0.5, align='edge')
# ax2.set_ylabel('Daily Precipitation (in)')
# ax2.set_ylim(0, 8)
# ax2.invert_yaxis()
# ax.set_xlim(min(df.index), max(df.index))
# plt.gcf().autofmt_xdate()
# plt.tight_layout()
# plt.savefig('NB_02_WL_Rain')
# plt.close()

# def get_flood_events(df, threshold):
df = pd.read_csv('NB_02_2022-01-01_2022-12-31.csv')
df['datetime'] = pd.to_datetime(df['date'])
threshold =  df['road_elevation'][0]# 4.65
flood = df[df['sensor_water_level_adj'] >= threshold]
flood['gap'] = flood['datetime'].sort_values().diff() > pd.to_timedelta('12 hours')
flood_gaps = flood[flood['gap']]

t1 = flood['datetime'].iloc[0]
duration = []
max_depth = []
avg_depth = []
tstart = []
tend = []
counter = 1
for t2 in flood_gaps['datetime']:
    event = flood[(flood['datetime'] >= t1) & (flood['datetime'] < t2)]
    tstart.append(event['datetime'].min())
    tend.append(event['datetime'].max())
    duration.append((event['datetime'].max() - event['datetime'].min()).total_seconds()/3600)
    max_depth.append(event['sensor_water_level_adj'].max())
    avg_depth.append(event['sensor_water_level_adj'].mean())

    event.plot(x='datetime', y='sensor_water_level_adj')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    #plt.savefig('NB_01_event_'+str(counter))
    plt.close()
    t1 = t2
    counter += 1

events = pd.DataFrame()
events['tstart'] = tstart
events['tend'] = tend
events['duration_hr'] = duration
events['max_depth'] = max_depth
events['avg_depth'] = avg_depth
events['max_depth'] = events['max_depth']-threshold
events['avg_depth'] = events['avg_depth']-threshold
#events.to_csv('NB_01_2022events.csv')

events.plot.scatter(x='duration_hr', y='max_depth', marker='o',
                    c='orange', s=100, alpha=0.7,
                    edgecolors='black', linewidths=1)
plt.tight_layout()
plt.savefig('NB_01_event_maxD_duration')
plt.close()