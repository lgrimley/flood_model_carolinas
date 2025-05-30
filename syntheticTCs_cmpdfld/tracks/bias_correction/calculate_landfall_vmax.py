import os
import pandas as pd
from src.utils import get_track_info_in_df, get_landfall_info, get_track_datetime
from hydromt_sfincs import SfincsModel
import scipy.io as sio


mod = SfincsModel(root=r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\sfincs_base_mod', mode='r')
region = mod.region.to_crs(4326).buffer(1.8)

# ADCIRC station locations that are used for calculating approximate landfall location
gage_locs = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_245\tracks')
for file in os.listdir(os.getcwd()):
    if file.endswith('.mat'):
        print(file)
        tc_tracks = sio.loadmat(f'{file}')
        gcm = file.split('_')[2]
        ssp = file.split('_')[3].removesuffix('cal')
        if ssp == '245':
            modeled_tcs = pd.read_csv(fr'../stormTide/gage_peaks_{gcm}_{ssp}.csv', index_col=0)
            modeled_tcs['tc_id'] = modeled_tcs.index.tolist()
            tc_ids = modeled_tcs['tc_id'].tolist()
        else:
            # Load the TC IDs that were modeled in ADCIRC (almost all tracks are within 200km of the gage locations)
            modeled_tcs = pd.read_csv(fr'../stormTide/gage_peaks_{gcm}_{ssp}.csv', index_col=0)
            modeled_tcs['tc_id'] = modeled_tcs.index.tolist()
            tc_ids = modeled_tcs['tc_id'].tolist()
        # For each TC track, figure out the approximate landfall time and wind speed, save to dataframe
        landfall_df = pd.DataFrame()
        counter = 0
        tot = len(tc_ids)
        for tc_id in tc_ids:
            tc_df = get_track_info_in_df(tc_id, tc_tracks)
            tc_df = get_track_datetime(tc_df)
            lf_df = get_landfall_info(tc_df=tc_df, gage_locs=gage_locs, clip_gdf=region)
            lf_df['tc_id'] = tc_id
            landfall_df = pd.concat(objs = [landfall_df, lf_df], axis=0, ignore_index=True)
            print(f'{counter} out of {tot}')
            counter += 1
            break

        out = landfall_df[['tc_id', 'vstore100']]
        out.to_csv(fr'{gcm}_{ssp}_landfall_vmax.csv')

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\tracks')
matfile = 'UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100.mat'
tc_tracks = sio.loadmat(f'{matfile}')

# Load the TC IDs that were modeled in ADCIRC (almost all tracks are within 200km of the gage locations)
modeled_tcs = pd.read_csv(r'../stormTide/gage_peaks_ZerosRemoved_ncep.csv', index_col=0)
modeled_tcs['tc_id'] = modeled_tcs.index.tolist()
tc_ids = modeled_tcs['tc_id'].tolist()

# For each TC track, figure out the approximate landfall time and wind speed, save to dataframe
landfall_df = pd.DataFrame()
counter = 0
tot = len(tc_ids)
issue_tcs = []
for tc_id in tc_ids:
    tc_df = get_track_info_in_df(tc_id, tc_tracks)
    tc_df = get_track_datetime(tc_df)
    tc_df2 = tc_df[tc_df['vt100'] != 0]
    try:
        lf_df = get_landfall_info(tc_df=tc_df2, gage_locs=gage_locs, clip_gdf=region)
        lf_df['tc_id'] = tc_id
        landfall_df = pd.concat(objs = [landfall_df, lf_df], axis=0, ignore_index=True)
    except:
        issue_tcs.append(tc_id)
        print(f'Issue with {tc_id}')
    print(f'{counter} out of {tot}')
    counter += 1

out = landfall_df
out.to_csv(fr'ncep_landfall_vmax_v2.csv')



# gage_tcs = pd.read_csv(r'.\stormTide\gage_peaks_ZerosRemoved.csv', index_col = 0)
#
# landfall_df = pd.DataFrame()
# for tc_id in gage_tcs.index.tolist():
#     tc_df = get_track_info_in_df(tc_id, tc_tracks)
#     tc_df = get_track_datetime(tc_df)
#     lf_df = get_landfall_info(tc_df=tc_df, gage_locs=gage_locs)
#     lf_df['tc_id'] = tc_id
#     landfall_df = pd.concat(objs = [landfall_df, lf_df], axis=0, ignore_index=True)
#
# gage_vstore = []
# for gage_id in gage_tcs.columns.tolist():
#     gage_data = pd.DataFrame(gage_tcs[gage_id].dropna())
#     vstore = []
#     for tc_id in gage_data.index.tolist():
#         vstore_max = landfall_df[landfall_df['tc_id'] == tc_id]['vstore100']
#         vstore.append(vstore_max.item())
#     gage_vstore.append(np.sort(vstore))


# gage_vstore = []
# for gage_id in gage_tcs.columns.tolist():
#     gage_data = pd.DataFrame(gage_tcs[gage_id].dropna())
#     vstore = []
#     for tc_id in gage_data.index.tolist():
#         tc_df = get_track_info_in_df(tc_id, tc_tracks)
#         vstore_max = tc_df['vstore100'].max()
#         vstore.append(vstore_max)
#     gage_vstore.append(np.sort(vstore))


# Plot the Empirical CDF
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), sharex=True, sharey=True)
# for i in range(len(gage_tcs.columns)):
#     gage_id = int(gage_tcs.columns[i])
#     sorted_data = gage_vstore[i]
#     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
#     ax.plot(sorted_data, cdf, linestyle='-', label=gage_id, alpha=0.7)
#     ax.set_xlabel('Vstore100', fontsize=12)
#     ax.set_ylabel('CDF', fontsize=12)
#     ax.grid(False)
#     #ax.legend()
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.margins(0,0)
# plt.savefig(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\Vmax_BiasCorrection\vmax_cdf_all.png',
#             dpi=300, bbox_inches="tight")
# plt.close()
#
#
# # Plot the Empirical CDF
# fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), sharex=True, sharey=True)
# for i in range(len(gage_tcs.columns)):
#     gage_id = int(gage_tcs.columns[i])
#     if gage_id <= 194:
#         ax = axs[0]
#     elif 194 < gage_id <= 201:
#         ax = axs[1]
#     else:
#         ax = axs[2]
#
#     sorted_data = gage_vstore[i]
#     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
#     ax.plot(sorted_data, cdf, marker='o', fillstyle='none', linestyle='-', label=gage_id, alpha=0.7)
#     ax.set_xlabel('Vstore100', fontsize=12)
#     ax.set_ylabel('CDF', fontsize=12)
#     ax.grid(False)
#     ax.legend()
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.margins(0,0)
# plt.savefig(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\Vmax_BiasCorrection\vmax_cdf.png',
#             dpi=300, bbox_inches="tight")
# plt.close()

# Plot the Empirical CDF
# fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(6.5, 11), sharex=True, sharey=True)
# axs = axs.flatten()
# #axs[-3].set_axis_off()
# axs[-2].set_axis_off()
# axs[-1].set_axis_off()
# for i in range(len(gage_tcs.columns)):
#     ax = axs[i]
#     gage_id = int(gage_tcs.columns[i])
#     sorted_data = gage_vstore[i]
#     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
#     ax.plot(sorted_data, cdf, marker='.', fillstyle='none', linestyle='-', color='black',
#             label=gage_id, alpha=0.7)
#     ax.legend(frameon=False)
#     ax.set_xlabel('Vstore100', fontsize=8)
#     ax.set_ylabel('CDF', fontsize=8)
#     ax.grid(False)
# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# plt.margins(0,0)
# plt.savefig(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\Vmax_BiasCorrection\vmax_cdf_all_indiv.png',
#             dpi=300, bbox_inches="tight")
# plt.close()