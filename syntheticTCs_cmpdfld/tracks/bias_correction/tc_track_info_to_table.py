import os
import pandas as pd
from src.utils import get_track_info_in_df, get_landfall_info, get_track_datetime
from hydromt_sfincs import SfincsModel
import scipy.io as sio

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

mod = SfincsModel(root=r'.\03_MODEL_RUNS\sfincs_base_mod', mode='r')
# this helps clip the track to the areas closer to the domain before searching for the landfall location
region = mod.region.to_crs(4326).buffer(5)

# ADCIRC station locations that are used for calculating approximate landfall location
gage_locs = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')

# Load track file
#matfile = r'.\02_DATA\NCEP_Reanalysis\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100.mat'
matfile = r'.\02_DATA\CMIP6_585\tracks\UScoast6_AL_canesm_ssp585cal_roEst1rmEst1_trk100.mat'
tc_tracks = sio.loadmat(f'{matfile}')

# Load the TC IDs that were modeled in ADCIRC (almost all tracks are within 200km of the gage locations)
#modeled_tcs = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\stormTide\gage_peaks_ZerosRemoved_ncep.csv', index_col=0)
modeled_tcs = pd.read_csv(r'.\02_DATA\CMIP6_585\stormTide\gage_peaks_ZerosRemoved_canesm_ssp585.csv', index_col=0)
modeled_tcs['tc_id'] = modeled_tcs.index.tolist()
tc_ids = modeled_tcs['tc_id'].tolist()

# For each TC track, figure out the approximate landfall time and wind speed, save to dataframe
landfall_info_df = pd.DataFrame()

issue_tcs = []
for tc_id in tc_ids:
    tc_df = get_track_info_in_df(tc_id, tc_tracks)
    tc_df = get_track_datetime(tc_df)
    tc_df_clean = tc_df[tc_df['vt100'] != 0]

    try:
        # Get the landfall information
        landfall_df = get_landfall_info(tc_df=tc_df_clean, gage_locs=gage_locs, clip_gdf=region)
        landfall_df['tc_id'] = tc_id
        landfall_info_df = pd.concat(objs = [landfall_info_df, landfall_df], axis=0, ignore_index=True)
    except:
        issue_tcs.append(tc_id)
        print(f'Issue with {tc_id}')
        break


landfall_info_df.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585\tracks\canesm_landfall_track_info.csv')




