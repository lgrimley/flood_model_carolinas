import os
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel
import fiona


def calculate_depth_above_FFE(waterlevel, buildings):
    zsmax = waterlevel.sel(x=buildings.centroid.x.to_xarray(),
                           y=buildings.centroid.y.to_xarray(),
                           method='nearest').values
    buildings['zsmax'] = zsmax.transpose()
    buildings['FFE_m'] = buildings['FFE'] * 0.3048
    buildings['depth_above_FFE_m'] = buildings['zsmax'] - buildings['FFE_m']
    return buildings


# Load windshield survey data
gdb_wss = r'Z:\users\lelise\projects\NBLL\geospatial\windshielfd_survey_from_BF\NewBern.gdb'
layers_wss = fiona.listlayers(gdb_wss)
wss_com = gpd.read_file(gdb_wss, layer=layers_wss[2], driver='FileGDB').to_crs(32618)
wss_res = gpd.read_file(gdb_wss, layer=layers_wss[3], driver='FileGDB').to_crs(32618)

# Load buildings data (clipped)
gdb_buildings = r'Z:\users\lelise\projects\NBLL\damage_florence\2023-02-22_joined_data\included_data.gdb'
layers_buildings = fiona.listlayers(gdb_buildings)
buildings_gdf = gpd.read_file(gdb_buildings, layer='buildings').to_crs(32618)
mod = SfincsModel(root=r'Z:\users\lelise\projects\NBLL\sfincs\neuse_river_estuary\50m_sbg_v2',
                  mode='r', data_libs=r'Z:\users\lelise\data\data_catalog.yml')
buildings_gdf = calculate_depth_above_FFE(waterlevel=mod.results['zsmax'], buildings=buildings_gdf)

# Join points of windshield survey of residential damage to buildings
df_join_res = gpd.sjoin_nearest(left_df=wss_res,
                                right_df=buildings_gdf,
                                max_distance=30,
                                distance_col='distance_to_feature')
print(df_join_res['distance_to_feature'].describe())

# Join points of windshield survey of commercial damage to buildings
df_join_com = gpd.sjoin_nearest(left_df=wss_com,
                                right_df=buildings_gdf,
                                max_distance=30,
                                distance_col='distance_to_feature')
print(df_join_com['distance_to_feature'].describe())

df = pd.concat([df_join_res, df_join_com], axis=0, ignore_index=True)
df_flood = df[df['PRIMCAUSE'] == 'Flood']
print(df_flood['TYPDAMAGE'].unique())
df_flood_major = df_flood[(df_flood['TYPDAMAGE'] == 'Major') | (df_flood['TYPDAMAGE'] == 'Destroyed') | (df_flood['TYPDAMAGE'] == 'Inaccessible')]
df_flood_minor = df_flood[(df_flood['TYPDAMAGE'] == 'Minor') | (df_flood['TYPDAMAGE'] == 'Affected')]

minor_threshold = 0
major_threshold = 0.45

# x = number of hits
x_min = df_flood_minor[df_flood_minor['depth_above_FFE_m'] > minor_threshold]
x_maj = df_flood_major[df_flood_major['depth_above_FFE_m'] >= major_threshold]

# y = number of misses
y_min = df_flood_minor[df_flood_minor['depth_above_FFE_m'] <= minor_threshold]
y_maj = df_flood_major[df_flood_major['depth_above_FFE_m'] < major_threshold]

# e = total number of events (e=x+y)
e_min = x_min.geometry.count() + y_min.geometry.count()
e_maj = x_maj.geometry.count() + y_maj.geometry.count()

# pod = probability of detection (pod=x/e)
# The percent of events that are forecasted/predicted
pod_min = x_min.geometry.count() / e_min
print('Minor Flood Threshold ' + str(minor_threshold) + ' meters yields a POD of: ' + str(pod_min.round(2)))
pod_maj = x_maj.geometry.count() / e_maj
print('Major Flood Threshold ' + str(major_threshold) + ' meters yields a POD of: ' + str(pod_maj.round(2)))