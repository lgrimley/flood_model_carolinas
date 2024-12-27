import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from src.core import *
from src.utils import *

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585')
gcms = ['canesm_ssp585cal', 'cnrm6_ssp585cal', 'ecearth6_ssp585cal', 'ipsl6_ssp585cal', 'miroc6_ssp585cal']
missing = []
for gcm in gcms:
    missing.append(gcm)
    fname = f'UScoast6_AL_{gcm}_roEst1rmEst1_trk100.mat'
    tc_tracks = sio.loadmat(fr'.\tracks\{fname}')
    #tc_select = pd.read_csv(fr'.\tracks\{fname}_200km.csv')
    tc_select = pd.read_csv(rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585\stormTide\stormTide_TCIDs_and_gageCounts_{gcm}.csv')

    ''' RAINFALL to netcdf '''
    input_dir = fr'.\rain\TCR_Gridded_{gcm}'
    output_dir = fr'.\rain\TCR_Gridded_{gcm}_hourly'
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    # figs_dir = r'.\rain\TotalPrecip_Figs'
    # usa = gpd.read_file(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\us_boundary\cb_2018_us_state_500k'
    #                     r'\cb_2018_us_state_500k.shp')

    for tc_id in tc_select['tc_id'].tolist():
        tc_id = int(tc_id)
        file = f'{str(tc_id).zfill(4)}.nc'

        if os.path.exists(os.path.join(input_dir, file)) is False:
            missing.append(tc_id)
            print(f'{tc_id} raw rainfall has not been processed!')
            continue

        out_name = os.path.join(output_dir, file)
        if os.path.exists(out_name) is False:
            print(f'Regridding rainfall for {tc_id}')

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

