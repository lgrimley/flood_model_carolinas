#!/usr/bin/env python
# coding: utf-8

import os
import re
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
import fiona
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel

# Specify location of damage data
damage_gdb = r'Z:\users\lelise\geospatial\flood_damage\included_data.gdb'

# Load the model the results
model_root = r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2016_Matthew\matt_hindcast_v6_200m_LPD2m_avgN'
mod = SfincsModel(root=model_root, mode='r')
mod.read_results()

# Load data catalog and files
cat = hydromt.DataCatalog(r'Z:\users\lelise\data\data_catalog.yml')
studyarea = mod.region
dem = cat.get_rasterdataset(data_like=os.path.join(r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2018_Florence'
                                                   r'\mod_v6\flo_hindcast_v6_200m_LPD2m_avgN\subgrid\dep_subgrid.tif'),
                            geom=studyarea)
sbgfldpth = cat.get_rasterdataset(data_like=r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2016_Matthew'
                                            r'\matt_hindcast_v6_200m_LPD2m_avgN\floodmap_0.05_hmin.tif',
                                  geom=studyarea)

# Setup output directory
out_dir1 = os.path.join(model_root, 'validation', 'nfip_damage')
if not os.path.exists(out_dir1):
    os.makedirs(out_dir1)

''' PART 1 - Determine damage status using NFIP claims/policy data at each structure'''


def get_building_event_damage_status(joined_gdb_filepath, studyarea_gdf, tstart=None, tend=None):
    # Function reads joined NC parcel, building, NFIP claims and policies GDB created in Feb 2023 (EPSG:32617)
    # and clips to study area and time period. Script outputs txt and shp files of clipped data

    # Read in area of interest shapefile and project
    studyarea_gdf = studyarea_gdf.to_crs(epsg=32617)

    # Read in structures information and clip to the study area
    buildings = gpd.read_file(joined_gdb_filepath, layer='buildings', mask=studyarea_gdf)

    # Get ids of buildings in study area
    studyarea_building_ids = buildings['building_id'].unique()

    # Get ids of buildings in study area that had a non-zero claim payout during study period
    claims = gpd.read_file(joined_gdb_filepath, layer='claims')
    claims['Date_of_Loss'] = pd.to_datetime(claims['Date_of_Loss'])
    claims['study_area'] = claims['building_id'].isin(studyarea_building_ids).astype(int)
    claims_filter = (claims['building_id'].isin(studyarea_building_ids))
    claims_filter = claims_filter & (claims['Date_of_Loss'] >= tstart)
    claims_filter = claims_filter & (claims['Date_of_Loss'] <= tend)
    claims_filter = claims_filter & (claims['Net_Total_Payments'] > 0.0)
    flooded_building_ids = claims[claims_filter]['building_id'].unique()

    # Get ids of buildings in study area that had a policy but no playout during study period
    # (for simplicity, define active policies based on peak date of event)
    policies = gpd.read_file(joined_gdb_filepath, layer='policies')
    policies['study_area'] = policies['building_id'].isin(studyarea_building_ids).astype(int)
    policies_filter = (policies['building_id'].isin(studyarea_building_ids))
    policies_filter = policies_filter & (policies['Policy_Effective_Date'] <= tstart)
    policies_filter = policies_filter & (policies['Policy_Expiration_Date'] >= tend)
    policies_filter = policies_filter & (~policies['building_id'].isin(flooded_building_ids))
    nonflooded_building_ids = policies[policies_filter]['building_id'].unique()

    # Get ids of buildings whose status is unknown (i.e., those who are uninsured)
    building_id_filter = (~buildings['building_id'].isin(flooded_building_ids))
    building_id_filter = building_id_filter & (~buildings['building_id'].isin(nonflooded_building_ids))
    inconclusive_building_ids = buildings['building_id'][building_id_filter]
    # Check
    if (len(buildings['building_id'].unique()) == len(nonflooded_building_ids) + len(flooded_building_ids) + len(
            inconclusive_building_ids)) is True:
        print('total buildings IDs is equal to the sum of the flooded, nonflooded, and inconclusive')

    # Create status codes for flooded, not flooded, and inconclusive
    inconclusive_status = np.ones(inconclusive_building_ids.shape) * np.nan
    print('No. of buildings w/ no policy or claim:', len(inconclusive_status))
    flood_status = np.ones(flooded_building_ids.shape)
    print('No. of buildings w/ claims:', len(flood_status))
    print('% of buildings', round((len(flood_status) / len(inconclusive_status) * 100), 2))
    nonflood_status = np.zeros(nonflooded_building_ids.shape)
    print('No. of buildings w/ policies and no claim:', str(len(nonflood_status)))
    print('% of buildings', round((len(nonflood_status) / len(inconclusive_status) * 100), 2))

    # Add to buildings df
    buildings['event_damage_status'] = np.nan
    buildings.loc[buildings['building_id'].isin(flooded_building_ids), 'event_damage_status'] = 1
    buildings.loc[buildings['building_id'].isin(nonflooded_building_ids), 'event_damage_status'] = 0
    buildings['x_coord'] = buildings.geometry.x
    buildings['y_coord'] = buildings.geometry.y

    return buildings


buildings = get_building_event_damage_status(joined_gdb_filepath=damage_gdb,
                                             studyarea_gdf=studyarea,
                                             tstart='2016-09-27',
                                             tend='2016-10-16')

''' PART 2 - Calculate flood depth information at buildings'''


def calculate_flood_depths_at_pt(gdf, waterdepth_da, waterlevel_da=None, terrain_da=None):
    # Extract depth at point
    print('Extracting depth at each point...')
    hmax = waterdepth_da.sel(x=gdf['geometry'].x.to_xarray(),
                             y=gdf['geometry'].y.to_xarray(),
                             method='nearest').values
    gdf['hmax'] = hmax.transpose()

    # Calculate flood depths: above the first floor elevation (FFE) and ground elevation
    if not (waterlevel_da is None):
        print('Calculating depth above FFE using water level...')
        # Extract water level at buildings
        zsmax = waterlevel_da.sel(x=gdf['geometry'].x.to_xarray(),
                                  y=gdf['geometry'].y.to_xarray(),
                                  method='nearest').values
        gdf['zsmax'] = zsmax.transpose()
        gdf['hmax_above_FFE_m'] = gdf['zsmax'] - gdf['FFE'] * 0.3048

    if not (terrain_da is None):
        print('Calculating depth above ground elevation using water level...')
        # Extract ground elevation at buildings
        gnd_elev = terrain_da.sel(x=gdf['geometry'].x.to_xarray(),
                                  y=gdf['geometry'].y.to_xarray(),
                                  method='nearest').values
        gdf['gnd_elev'] = gnd_elev.transpose()
        gdf['hmax_above_gnd_elev'] = gdf['zsmax'] - gdf['gnd_elev']

    return gdf


buildings = calculate_flood_depths_at_pt(gdf=buildings.to_crs(mod.crs),
                                         waterdepth_da=sbgfldpth,
                                         waterlevel_da=mod.results['zsmax'].max(dim='timemax'),
                                         terrain_da=dem)
pd.DataFrame(buildings.drop(columns='geometry')).to_csv(os.path.join(out_dir1, 'depth_at_event_nfip_buildings.csv'),
                                                        sep=',')

buildings = pd.read_csv(os.path.join(out_dir1, 'depth_at_event_nfip_buildings.csv'))
buildings['elev_grp'] = 'xx'
buildings['elev_grp'][buildings['gnd_elev'] <= 15.0] = 'Coastal (Elevation <= 15m)'
buildings['elev_grp'][(buildings['gnd_elev'] > 15.0)] = 'Inland (Elevation > 15m)'

''' PART 3 - Flag flooded and non flooded hits/misses '''


def hit_miss_flags(gdf, depth_threshold=None):
    if depth_threshold is None:
        depth_threshold = 0.01

    # Hit/miss for claims (flooded)
    flood_hit_filter = (gdf['event_damage_status'] == 1) & (gdf['hmax'] >= depth_threshold)
    flood_miss_filter = (gdf['event_damage_status'] == 1) & (gdf['hmax'] < depth_threshold)
    flood_miss_filter = flood_miss_filter & (gdf['hmax'] < 0)

    # Hit/miss for policies without claims (no flood)
    nonflood_hit_filter = (gdf['event_damage_status'] == 0) & (gdf['hmax'] < depth_threshold)
    nonflood_miss_filter = (gdf['event_damage_status'] == 0) & (gdf['hmax'] >= depth_threshold)

    gdf.loc[flood_hit_filter, 'flood_hitmiss'] = 1
    gdf.loc[flood_miss_filter, 'flood_hitmiss'] = 0
    gdf.loc[nonflood_hit_filter, 'nonflood_hitmiss'] = 1
    gdf.loc[nonflood_miss_filter, 'nonflood_hitmiss'] = 0

    return gdf


hmin = 0.5
out_dir = os.path.join(out_dir1, ('depth_threshold_' + str(hmin)))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

buildings.loc[buildings['hmax'].isna(), 'hmax'] = -9999.0  # Set no data to -9999.0 (e.g., zero water depth)
buildings = hit_miss_flags(gdf=buildings[buildings['event_damage_status'].notna()], depth_threshold=hmin)

# Plotting
# font = {'family': 'Arial',
#         'size': 10}
# mpl.rc('font', **font)
# mpl.rcParams.update({'axes.titlesize': 10})
#
# props = dict(boxes="white", whiskers="black", caps="black")
# boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
# flierprops = dict(marker='+', markerfacecolor='none', markersize=3,
#                   markeredgecolor='black')
# medianprops = dict(linestyle='-', linewidth=2, color='black')
# meanpointprops = dict(marker='D',
#                       markeredgecolor='black',
#                       markerfacecolor='lightgrey',
#                       markersize=4)
# fig, axs = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(6, 3))
# bp = flooded_buildings['hmax'].plot.box(ax=axs,
#                                         vert=False,
#                                         color=props,
#                                         boxprops=boxprops,
#                                         flierprops=flierprops,
#                                         medianprops=medianprops,
#                                         meanprops=meanpointprops,
#                                         meanline=False,
#                                         showmeans=True,
#                                         patch_artist=True)
# axs.set_xlabel('NC Water Depth (m) at Buildings w/ NFIP Claim')
# # axs.set_yticklabels([f'Compound\n(n={nc_n[0]})', f'Coastal\n(n={nc_n[1]})', f'Runoff\n(n={nc_n[2]})'])
# kwargs = dict(linestyle='--', linewidth=0.75, color='lightgrey', alpha=0.8)
# axs.grid(visible=True, which='major', axis='x', **kwargs)
# #axs.set_xscale("log")
# axs.set_xlim(0, 10)
# pos1 = axs.get_position()  # get the original position
# plt.margins(x=0, y=0)
# plt.savefig(os.path.join(out_dir1, 'depth_at_structures.png'),
#             dpi=225,
#             bbox_inches="tight")
# plt.close()


# Save to shapefile and txt file
# pd.DataFrame(buildings.drop(columns='geometry')).to_csv(os.path.join(out_dir, 'coastal_clipped_buildings.csv'), sep=',')


''' PART 4 - Confusion Matrix '''


def forecast_scores(x, y, z, w):
    # https://www.cawcr.gov.au/projects/verification/
    scores = pd.DataFrame(columns=['Name', 'Score', 'Perfect Score'])
    scores['Name'] = ['Hits', 'Misses', 'False Alarm', 'Correct Non-forecast', 'Events', 'Cases',
                      'Accuracy', 'Bias', 'POD', 'FAR', 'POFD', 'Success Ratio', 'Critical Success Index']
    scores['Perfect Score'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 0, 1, 0, 0, 1, 1]
    events = x + y
    cases = events + z + w

    # accuracy = hits + correct negatives / total
    accuracy = (x + w) / cases

    # Bias score (frequency bias)
    bias = (x + z) / events
    bias = bias - 1

    # POD (hit rate) = probability of detection (pod=x/e); The percent of events that are forecasted/predicted
    pod = x / events

    # FAR = false alarm rate (far=z/(x+z))
    # Measure of failure; ratio of unsuccessful forecasts to the total number of positive forecasts
    far = z / (x + z)

    # Probability of false detection (false alarm rate); POFD = z / (z+w)
    pofd = z / (z + w)

    # sr = success ratio; sr=x/(x+z)
    sr = x / (x + z)

    # csi = critical success index; csi=x/(x+y+z)
    # ratio of the number of hits to the number of events plus the number of false alarms
    csi = x / (events + z)

    scores['Score'] = np.around([x, y, z, w, events, cases, accuracy, bias, pod, far, pofd, sr, csi], decimals=2)

    return scores


# x = number of hits
# Number of claims that had a flood depth over the threshold
# y = number of misses
# Number of claims that did NOT have a modeled flood depth over the threshold
# z = false alarms (e.g., predicting water where none is reported)
# Number of policies that had a modeled flood depth over the threshold, but no claim was made
# w = correct negative forecasts (e.g., correct non-forecasts)
# Number of policies that did NOT have a modeled flood depth over the threshold and also no claim

flooded = buildings[~buildings['flood_hitmiss'].isna()]  # remove nans
flooded_hits = flooded[flooded['flood_hitmiss'] == 1]
flooded_miss = flooded[flooded['flood_hitmiss'] == 0]

nonflooded = buildings[~buildings['nonflood_hitmiss'].isna()]  # remove nans
nonflooded_hits = nonflooded[nonflooded['nonflood_hitmiss'] == 1]
nonflooded_misses = nonflooded[nonflooded['nonflood_hitmiss'] == 0]

# flooded_hits.to_file(os.path.join(out_dir, 'hits.shp'))
# flooded_miss.to_file(os.path.join(out_dir, 'misses.shp'))
# nonflooded_hits.to_file(os.path.join(out_dir, 'correct_nonforecast.shp'))
# nonflooded_misses.to_file(os.path.join(out_dir, 'false_alarm.shp'))

x, y = buildings['flood_hitmiss'].value_counts()
w, z = buildings['nonflood_hitmiss'].value_counts()
scores = forecast_scores(x, y, z, w)
scores.to_csv(os.path.join(out_dir, 'forecasting_scores.csv'))

''' PART 5 - Plotting '''
font = {'family': 'Times New Roman',
        'size': 10}
mpl.rc('font', **font)

# Plotting the data on a map with contextual layers
layers_dict = {
    'lyr1': {'dataset': 'enc_domain_HUC6_clipped',
             'color': 'grey',
             'edgecolor': 'none',
             'linewidth': 0,
             'alpha': 0.25,
             'zorder': 0,
             },
    'lyr2': {'dataset': 'carolinas_coastal_wb',
             'color': 'steelblue',
             'edgecolor': 'none',
             'linewidth': 0,
             'alpha': 1,
             'zorder': 1,
             },
    'lyr3': {'dataset': 'carolinas_major_rivers',
             'color': 'steelblue',
             'edgecolor': 'steelblue',
             'linewidth': 0.5,
             'alpha': 1,
             'zorder': 1,
             },
    'lyr4': {'dataset': r'Z:\users\lelise\geospatial\fris\bathy_v5\NHDArea_LowerPeeDee.shp',
             'color': 'steelblue',
             'edgecolor': 'steelblue',
             'linewidth': 0.5,
             'alpha': 1,
             'zorder': 1,
             },
    'lyr5': {'dataset': r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp',
             'color': 'none',
             'edgecolor': 'black',
             'linewidth': 1,
             'alpha': 1,
             'zorder': 1,
             },
}

# Download and prep shapefile layers for plots
for layer in layers_dict.keys():
    l_gdf = cat.get_geodataframe(layers_dict[layer]['dataset'])
    l_gdf.to_crs(mod.crs, inplace=True)
    layers_dict[layer]['gdf'] = l_gdf.clip(mod.region)

# More plot prep
wkt = dem.raster.crs.to_wkt()
utm_zone = dem.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(layers_dict['lyr5']['gdf'].buffer(10000).total_bounds)[[0, 2, 1, 3]]
plt_gdf = [flooded_hits, flooded_miss, nonflooded_hits, nonflooded_misses]
plt_label = ['Hits', 'Misses', 'Correct Non-Forecasts', 'False Alarms']

# Now PLOT!
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 5.5), subplot_kw={'projection': utm}, tight_layout=True)
axs = axs.flatten()
for i in range(len(plt_label)):
    # Plot background/geography layers
    for layer in layers_dict.keys():
        l_gdf = layers_dict[layer]['gdf']
        l_gdf.plot(ax=axs[i],
                   color=layers_dict[layer]['color'],
                   edgecolor=layers_dict[layer]['edgecolor'],
                   linewidth=layers_dict[layer]['linewidth'],
                   zorder=layers_dict[layer]['zorder'],
                   alpha=layers_dict[layer]['alpha'])

    # Plot the stat at each gage
    plt_gdf[i].centroid.plot(
        color='black',
        ax=axs[i],
        markersize=8,
        marker='.',
        alpha=0.8,
        zorder=2)

    # Add title and save figure
    axs[i].set_extent(extent, crs=utm)
    axs[i].set_title(plt_label[i], loc='left')

    if i == 0:
        axs[i].set_ylabel(f"y coordinate UTM zone {utm_zone} [m]")
        axs[i].yaxis.set_visible(True)
        axs[i].xaxis.set_visible(False)
    elif i == 1:
        axs[i].yaxis.set_visible(False)
        axs[i].xaxis.set_visible(False)
    elif i == 2:
        axs[i].set_ylabel(f"y coordinate UTM zone {utm_zone} [m]")
        axs[i].yaxis.set_visible(True)
        axs[i].set_xlabel(f"x coordinate UTM zone {utm_zone} [m]")
        axs[i].xaxis.set_visible(True)
    elif i == 3:
        axs[i].yaxis.set_visible(False)
        axs[i].set_xlabel(f"x coordinate UTM zone {utm_zone} [m]")
        axs[i].xaxis.set_visible(True)

    axs[i].ticklabel_format(style='plain', useOffset=False)
    axs[i].set_aspect('equal')

plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(out_dir, 'contingency_matrix_' + str(hmin) + 'm.png'), bbox_inches='tight',
            dpi=225)
plt.close()
