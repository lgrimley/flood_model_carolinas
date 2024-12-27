import os

import pandas as pd

from src.core import *
from src.utils import get_track_info_in_df, track_points_to_linestring, get_track_datetime
import numpy as np
import matplotlib.pyplot as plt
from hydromt_sfincs import SfincsModel
import sys
from src.core import *


mod = SfincsModel(root=r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_base_mod', mode='r')
region = mod.region.to_crs(4326).buffer(2.5)

def get_landfall_info(tc_df, gage_locs, clip_gdf):
    tc_gdf = gpd.GeoDataFrame(tc_df, geometry=gpd.points_from_xy(x=tc_df['lon100'], y=tc_df['lat100'], crs=4326))
    tc_gdf = tc_gdf.clip(clip_gdf)
    gage_locs_gdf = gpd.GeoDataFrame(gage_locs,
                                     geometry=gpd.points_from_xy(x=gage_locs['x'], y=gage_locs['y'], crs=4326))

    # Create an empty list to store the results
    min_distances = []

    # Loop through each point in df1
    for i, point1 in tc_gdf.iterrows():
        # Calculate the distance from the current point in df1 to all points in df2
        distances = gage_locs_gdf.geometry.apply(lambda point2: point1.geometry.distance(point2))

        # Find the smallest distance and the corresponding point from df2
        min_distance = np.round(distances.min(), 5)
        closest_point = gage_locs_gdf.loc[distances.idxmin()]

        # Store the result for the current point in df1
        min_distances.append((i, min_distance, closest_point['gage_id']))

    # Create a DataFrame with the results
    result_df = gpd.GeoDataFrame(min_distances, columns=["Index_in_df1", "Min_Distance", "Closest_Point_in_df2"])

    lf_idx = result_df.loc[result_df['Min_Distance'].idxmin()]
    landfall_info = tc_gdf[tc_gdf.index == lf_idx['Index_in_df1']]
    landfall_info['Min_Distance'] = lf_idx['Min_Distance']
    landfall_info['Closest_Point_in_df2'] = lf_idx['Closest_Point_in_df2']

    return landfall_info


gage_locs = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585\tracks')
for file in os.listdir(os.getcwd()):
    if file.endswith('.mat'):
        print(file)
        tc_tracks = sio.loadmat(f'{file}')
        select = pd.read_csv(fr'{file}_200km.csv')

        landfall_df = pd.DataFrame()
        counter = 0
        tot = len(select['tc_id'])
        for tc_id in select['tc_id'].tolist():
            tc_df = get_track_info_in_df(tc_id, tc_tracks)
            tc_df = get_track_datetime(tc_df)
            lf_df = get_landfall_info(tc_df=tc_df, gage_locs=gage_locs, clip_gdf=region)
            lf_df['tc_id'] = tc_id
            landfall_df = pd.concat(objs = [landfall_df, lf_df], axis=0, ignore_index=True)
            print(f'{counter} out of {tot}')
            counter += 1

        out = landfall_df[['tc_id', 'vstore100']]
        out.to_csv(fr'{file}_select_vstore100.csv')

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