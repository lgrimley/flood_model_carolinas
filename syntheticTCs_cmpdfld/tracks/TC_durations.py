import sys
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\synthetic_tc_cmpdfld')

from syntheticTC_utils import *

os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis')
track_dir = os.path.join(os.getcwd(), 'tracks')

# Load the storm tracks
track_file = os.path.join(track_dir, f'UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100')
tc_tracks = sio.loadmat(f'{track_file}.mat')

# Read in the ADCIRC modeled peaks and get the TC ids
gage_peaks = pd.read_csv('.\stormTide\gage_peaks.csv', index_col=0)
selected_tcs = gage_peaks.index.tolist()

tc_time_info = pd.DataFrame(index = selected_tcs, columns=['tstart','tend'])
for tc_id in selected_tcs:
    # Get the storm track date time information
    track_df = get_track_info_in_df(tc_id=tc_id, tc_tracks=tc_tracks)
    track_df = track_df[track_df['datetime'] != 0]
    tc_time_info.loc[tc_id, 'tstart'] = track_df['datetime'].iloc[0]
    tc_time_info.loc[tc_id, 'tend'] = track_df['datetime'].iloc[-1]

tc_time_info['duration'] = tc_time_info['tend'] - tc_time_info['tstart']

mean = np.mean(tc_time_info['duration'].values)
max = np.max(tc_time_info['duration'].values)
median, pct95 = np.percentile(tc_time_info['duration'], q=[50, 95])

tc_time_info['duration_days'] = [x.total_seconds()/86400.0 for x in tc_time_info['duration']]
fig, ax = plt.subplots(nrows=1, ncols=1,
                       figsize=(6, 4), tight_layout=True)
tc_time_info['duration_days'].plot.hist(ax=ax, legend=False)
ax.set_xlabel('TC Duration (days)')
plt.savefig(r'.\tracks\TC_duration_histogram.png')
plt.close()

# Mathew 8 day compound simulation took ~0.7 hours with approx. 32 CPUs
# Assume 8 days takes 1 hour

tc_time_info['cpu_time_hrs_x3'] = (tc_time_info['duration_days'] * (1/8))*3

total_cpu_time = np.ceil(tc_time_info['cpu_time_hrs_x3'].sum())*2

