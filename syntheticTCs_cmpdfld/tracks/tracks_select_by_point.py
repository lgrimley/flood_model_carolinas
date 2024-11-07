import os

import scipy.io as sio
import geopandas as gpd
import pandas as pd
import numpy as np
from syntheticTC_utils import track_points_to_linestring

# Load points (e.g., ADCIRC locations)
points = pd.read_csv(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')
points_gdf = gpd.GeoDataFrame(points,
                              geometry=gpd.points_from_xy(x=points['x'].values, y=points['y'].values, crs=4326))
points_gdf_buffer = gpd.GeoDataFrame(points_gdf, geometry=points_gdf.geometry.buffer(distance=2))

# Load the TC Track file
fname = r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100'
tc_tracks = sio.loadmat(f'{fname}.mat')
tc_tracks_polyline = track_points_to_linestring(tc_tracks, output_shpfile=None)

gage = 194
# Select tracks that intersect
pt = points_gdf_buffer[points_gdf_buffer['gage_id'] == gage]
select_trks = tc_tracks_polyline.overlay(pt, how='intersection')
trk_ids = select_trks['tc_id'].unique()

# Compare with the TC IDs stored for ADCIRC simulation
adc = pd.read_csv(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\stormTide\WL_inds_subset.csv',
                  index_col=0)
adc_pt = adc[f'{gage}'].unique()
adc_pt = adc_pt[~np.isnan(adc_pt)]

common_tcs = select_trks[select_trks['tc_id'].isin(adc_pt)]

os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis\tracks')
adcirc_tcs = pd.read_csv('adcirc_modeled_TCs_all.csv')
buffer_200km_tcs = pd.read_csv('UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100_200km.csv')

common_tcs = adcirc_tcs[adcirc_tcs['tc_id'].isin(buffer_200km_tcs['tc_id'])]
no_surge_tcs = buffer_200km_tcs[~buffer_200km_tcs['tc_id'].isin(adcirc_tcs['tc_id'])]
print(f'{len(common_tcs)} of the ADCIRC TCs are also within the 200km buffer.')