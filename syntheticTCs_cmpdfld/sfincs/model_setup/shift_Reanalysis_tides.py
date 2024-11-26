from src.utils import *

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

'''
Load station mapping table, track file, and reanalysis data
'''
# Mapping the stormTide ADCIRC to the reanalysis extraction locations
match_file = r'.\NCEP_Reanalysis\stormTide\map_Reanalysis_to_stormTideLocs.csv'
if os.path.exists(match_file) is False:

    reanalysis_locs = pd.read_csv(r'.\EDSReanalysis_data\full_data_meta.csv')
    stormTide_locs = pd.read_csv(r'.\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')
    stormTide_locs['gage_id'] = stormTide_locs['gage_id'].astype(int)

    reanalysis_locs = gpd.GeoDataFrame(reanalysis_locs,
                                       geometry=gpd.points_from_xy(x=reanalysis_locs['LON'],y=reanalysis_locs['LAT'],
                                                                   crs=4326)).to_crs(32617)
    stormTide_locs = gpd.GeoDataFrame(stormTide_locs,
                                       geometry=gpd.points_from_xy(x=stormTide_locs['x'],y=stormTide_locs['y'],
                                                                   crs=4326)).to_crs(32617)

    match = stormTide_locs.sjoin_nearest(reanalysis_locs, how='left', max_distance=2000)
    match['gage_id'] = match['gage_id'].astype(int)
    match.to_csv(r'.\NCEP_Reanalysis\stormTide\map_Reanalysis_to_stormTideLocs.csv')
else:
    match = pd.read_csv(match_file, index_col=0)
    match = gpd.GeoDataFrame(match, geometry=gpd.points_from_xy(x=match['LON'],y=match['LAT'],crs=4326))
    print('Mapping table for ADCIRC Reanalysis and Storm Tide stations loaded!')

# Load the storm tracks
track_file = r'.\NCEP_Reanalysis\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100.mat'
tc_tracks = sio.loadmat(track_file)

# Reanalysis data from ADCIRC EDS v2
reanalysis = xr.open_dataset(r'.\EDSReanalysis_data\EDSReanalysis_V2_1992_2022.nc', engine='netcdf4')
stormTide_dir = r'.\NCEP_Reanalysis\stormTide\adcirc_waterlevel_netcdf'


for tc_id in [1234, 2645, 2773, 3429]:
    track_df = get_track_info_in_df(tc_id, tc_tracks)
    track_df = track_df[track_df['datetime'] != 0]
    track_tstart = track_df['datetime'].iloc[0]
    track_tend = track_df['datetime'].iloc[-1]

    # Open the netcdf of water level time series for the TC

    file = f'{str(tc_id).zfill(4)}.nc'
    adcirc_st = xr.open_dataset(os.path.join(stormTide_dir, file), engine='netcdf4')
    adcirc_st_stations =  adcirc_st.index.values
    adcirc_st_tstart = pd.to_datetime(adcirc_st.time.min().values)
    adcirc_st_tend = pd.to_datetime(adcirc_st.time.max().values)

    print(f'Track start at: {track_tstart}')
    print(f'ADCIRC start at: {adcirc_st_tstart}')
    print(f'Track end at: {track_tend}')
    print(f'ADCIRC end at: {adcirc_st_tend}')

    min_tstart = min(track_tstart, adcirc_st_tstart)
    max_tend = max(track_tend, adcirc_st_tend)


    stations_st = []
    highTide_time_st = []
    stations_re = []
    highTide_time_re = []
    for station_id in adcirc_st.index.values:
        # Get the high tide time lag of the ADCIRC Storm Tide gages
        sta = adcirc_st.sel(index=station_id).sel(time=slice(min_tstart, min_tstart + pd.to_timedelta('12h')))
        sta_highTide_time = pd.to_datetime(sta['waterlevel'].idxmax(dim='time').values)
        highTide_time_st.append(sta_highTide_time)
        stations_st.append(station_id)

        # Get the high tide time lag of the corresponding Reanalysis gages
        re_id = match[match['gage_id'] == station_id]['Point'].item()
        sta = reanalysis.sel(index=re_id).sel(time=slice(min_tstart, min_tstart + pd.to_timedelta('12h')))
        sta_highTide_time = pd.to_datetime(sta['waterlevel'].idxmax(dim='time').values)
        highTide_time_re.append(sta_highTide_time)
        stations_re.append(re_id)

    tide_df = pd.DataFrame()
    tide_df['stations_st'] = stations_st
    tide_df['high_tide_st'] = highTide_time_st
    tide_df['stations_re'] = stations_re
    tide_df['high_tide_re'] = highTide_time_re
    tide_df['st_minus_re'] = tide_df['high_tide_st'] - tide_df['high_tide_re']
    tide_df['abs'] = np.abs(tide_df['st_minus_re'])
    outfile = rf'.\NCEP_Reanalysis\stormTide\ADCIRC_storm_vs_reanalysis\{tc_id}_comparison.csv'
    tide_df.to_csv(outfile)