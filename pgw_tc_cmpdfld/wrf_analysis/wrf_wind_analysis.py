import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


''' WIND SPEED '''
ws = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\wrf_analysis\scale_factors'
                 r'\scale_factors_carolinas_landMask\wind_spd_thresh_15_to_100\wind_spd_thresh_15_to_100_data.csv',
                 header=0)
print(ws.head(3))


storms = ['florence', 'floyd', 'matthew']
for storm in storms:
    s1 = ws[(ws['storm'] == storm) & (ws['run'] != 'ensmean')].round(2)
    s1_present = s1[s1['climate'] == 'present']
    s1_present_wnd_area = np.round(s1_present['count'].values, decimals=0)
    print(f'Mean count: {np.round(np.mean(s1_present_wnd_area),decimals=0)}')
    # print(f'Min count: {np.round(np.min(s1_present_wnd_area),decimals=0)}')
    # print(f'Max count: {np.round(np.max(s1_present_wnd_area),decimals=0)}')

    s1_future = s1[s1['climate'] == 'future']
    s1_future_wnd_area = s1_future['count'].values
    print(f'Mean count: {np.round(np.mean(s1_future_wnd_area),decimals=0)}')
    # print(f'Min count: {np.round(np.min(s1_future_wnd_area),decimals=0)}')
    # print(f'Max count: {np.round(np.max(s1_future_wnd_area),decimals=0)}')

    count_diff = s1_future_wnd_area - s1_present_wnd_area
    print(f'Mean difference: {np.round(np.mean(count_diff),decimals=0)}')
    # print(f'Min difference: {np.round(np.min(count_diff),decimals=0)}')
    # print(f'Max difference: {np.round(np.max(count_diff),decimals=0)}')
