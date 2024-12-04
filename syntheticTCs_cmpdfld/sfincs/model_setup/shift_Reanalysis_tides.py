import hydromt
from src.utils import *
from src.core import NCEP_DataPaths, SyntheticTrack
import matplotlib.pyplot as plt

tc_index = 2773
DatCat = hydromt.DataCatalog(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\data_catalog_NCEP.yml')
track = SyntheticTrack(DataPaths=NCEP_DataPaths, DatCat=DatCat, tc_index=tc_index)
stormTide = track.stormTide
tides = track.reanalysis_data
map_index = track.coastal_locations_mapped
tstart, tend = track.track_time

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\reanalysis_shift')
stations_st = []
highTide_time_st = []
stations_re = []
highTide_time_re = []
for station_id in stormTide.index.values:
    # Get the high tide time lag of the ADCIRC Storm Tide gages
    sta_st = stormTide.sel(index=station_id).to_dataframe()
    sta_st['delta_waterlevel'] = sta_st['waterlevel'].diff()
    st_empty = sta_st[(sta_st['delta_waterlevel'] == 0.0)]
    st_empty_tstart = st_empty.index[0]
    tide_st = stormTide.sel(index=station_id).sel(time=slice(st_empty_tstart - pd.to_timedelta('12h'), st_empty_tstart))

    tide_st = tide_st.to_dataframe()
    tide_st['delta_waterlevel'] = tide_st['waterlevel'].diff() # Calculate the difference in water levels across time
    increasing_trend = tide_st[tide_st['delta_waterlevel'] > 0] # Identify where water level is increasing
    sta_highTide_time = increasing_trend['waterlevel'].idxmax()
    highTide_time_st.append(sta_highTide_time)
    stations_st.append(station_id)

    # Get the high tide time lag of the corresponding Reanalysis gages
    re_id = map_index[map_index['gage_id'] == station_id]['Point'].item()
    sta_re = tides.sel(index=re_id).sel(time=slice(tend, tend + pd.to_timedelta('12h')))
    sta_re = sta_re.to_dataframe()
    sta_re['delta_waterlevel'] = sta_re['waterlevel'].diff() # Calculate the difference in waterlevels across time
    increasing_trend = sta_re[sta_re['delta_waterlevel'] > 0] # Identify where water level is increasing
    sta_highTide_time_re = increasing_trend['waterlevel'].idxmax()
    highTide_time_re.append(sta_highTide_time_re)
    stations_re.append(re_id)
    print(station_id)

    # Plot water level
    compare = pd.concat(objs=[sta_st, sta_re], axis=1, ignore_index=False)
    dfp = compare['waterlevel']
    dfp.columns = ['stormTide', 'reanalysis']

    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
    dfp.plot(ax=ax, legend=True)
    ax.axvline(x=sta_highTide_time_re, color= 'grey')
    ax.axvline(x=sta_highTide_time, color='black')
    ax.set_xlabel('')
    ax.set_ylabel('Water Level (m+MSL)')
    plt.title(f'{np.abs(sta_highTide_time - sta_highTide_time_re)}')
    plt.savefig(rf'{str(tc_index).zfill(4)}_tideShift_{station_id}_backend.png')
    plt.close()


tide_df = pd.DataFrame()
tide_df['stations_st'] = stations_st
tide_df['high_tide_st'] = highTide_time_st
tide_df['stations_re'] = stations_re
tide_df['high_tide_re'] = highTide_time_re
tide_df['st_minus_re'] = tide_df['high_tide_st'] - tide_df['high_tide_re']
tide_df['abs'] = np.abs(tide_df['st_minus_re'])
outfile = f'{tc_index}_comparison_backend.csv'
tide_df.to_csv(outfile)