import matplotlib.pyplot as plt
import xarray as xr
import scipy.io as sio
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from syntheticTC_utils import *

os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis')
fname = r'.\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100'
tc_tracks = sio.loadmat(f'{fname}.mat')

''' RAINFALL to netcdf '''
input_dir = r'.\rain\02_TCR_RainOutput_Gridded'
output_dir = r'.\rain\03_TCR_RainOutput_Gridded_hourly'
figs_dir = r'.\rain\TotalPrecip_Figs'
usa = gpd.read_file(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\us_boundary\cb_2018_us_state_500k'
                    r'\cb_2018_us_state_500k.shp')

for rain_file in os.listdir(input_dir):
    if rain_file.endswith('.nc'):
        tc_id = int(rain_file.split('.')[0])
        file = f'{str(tc_id).zfill(4)}.nc'
        out_name = os.path.join(output_dir, file)

        if os.path.exists(out_name) is False:
            print(tc_id)

            # Get the track information w/ datetime
            df = get_track_info_in_df(tc_id=tc_id, tc_tracks=tc_tracks)
            time = df['datetime'][df['datetime'] != 0]

            # Load the gridded TCR rainfall for the TC
            rain_data = xr.open_dataset(os.path.join(input_dir, file))

            # The first 2 time steps of the track have zero rainfall (e.g., 4 hours)
            # The end of storm is cut off if the rainfall is zero
            time_rain = time.iloc[2:(2+len(rain_data['time']))]
            rain_data['time'] = time_rain.values
            rain_data = rain_data.rio.write_crs("epsg:4326", inplace=True)
            rain_data = rain_data*2.0
            # Downscale 2-hour rain rates to hourly
            rain_data_1hr = rain_data.resample(time='1H').mean()

            totP_2hr = rain_data.sum()['precip'].item()
            totP_1hr = rain_data_1hr.sum()['precip'].item()
            if abs(totP_2hr - totP_1hr) > 100:
                print(f'Check {tc_id}: precip difference {abs(totP_2hr - totP_1hr)}')
                continue
            else:
                rain_data_1hr.to_netcdf(out_name)

                # # Plotting
                # track_gdf = track_as_gdf(df)
                # tot_rain = rain_data.sum(dim='time')['precip']
                # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)
                # tot_rain.plot(ax=ax, cmap='jet')
                # track_gdf.geometry.plot(ax=ax, color='black')
                # usa.geometry.plot(ax=ax, color='none', edgecolor='black')
                # plt.savefig(os.path.join(figs_dir, file.replace('.nc', '_total_precip.png')))
                # plt.close()

