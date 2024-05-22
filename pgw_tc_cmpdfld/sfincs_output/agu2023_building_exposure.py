#!/usr/bin/env python
# coding: utf-8
# This script was used to calculate flood depths at buildings across the Carolinas domain in 12/23
import os
import re
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import fiona
import matplotlib as mpl
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel

# Load the model the results
cat = hydromt.DataCatalog(r'Z:\users\lelise\data\data_catalog.yml')
os.chdir(r'Z:\users\lelise\projects\Carolinas\Chapter2\sfincs_models')

# Setup output directory
out_dir = os.path.join(os.getcwd(), '00_analysis', 'floodmaps')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


mod = SfincsModel(root='flor_ensmean_present', mode='r')


''' PART 1 - Determine damage status using NFIP claims/policy data at each structure'''
# Read in area of interest shapefile and project
studyarea_gdf = mod.region.to_crs(epsg=32617)

# Read in structures information and clip to the study area
nc_buildings = gpd.read_file(r'Z:\users\lelise\geospatial\flood_damage\included_data.gdb',
                             layer='buildings',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
nc_buildings['STATE'] = 'NC'
b1 = nc_buildings.drop(nc_buildings.columns[~nc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of NC Buildings in Study Area:', str(len(nc_buildings)))

# Load SC buildings from NSI
sc_buildings = gpd.read_file(r'Z:\users\lelise\geospatial\infrastructure\nsi_2022_45.gpkg',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
sc_buildings['STATE'] = 'SC'
b2 = sc_buildings.drop(sc_buildings.columns[~sc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of SC Buildings in Study Area:', str(len(sc_buildings)))

buildings = pd.concat(objs=[b1, b2],
                      axis=0,
                      ignore_index=True)

l_gdf = cat.get_geodataframe(
    r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
l_gdf = l_gdf[l_gdf['Name'].isin(['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee'])]
l_gdf.to_crs(epsg=32617, inplace=True)
l_gdf = l_gdf[["HUC6", "Name", "geometry"]]
buildings = gpd.tools.sjoin(left_df=buildings,
                            right_df=l_gdf,
                            how='left')

''' PART 2 - Calculate flood depth at buildings'''

scenarios = [
    'flor_ensmean_future_45cm',
    'flor_ensmean_future_45cm_coastal',
    'flor_ensmean_future_45cm_runoff',
    'flor_ensmean_future_110cm',
    'flor_ensmean_future_110cm_coastal',
    'flor_ensmean_future_110cm_runoff',
    # 'flor_ensmean_present',
    # 'flor_ensmean_present_coastal',
    # 'flor_ensmean_present_runoff',
    # 'matt_ensmean_present',
    # 'matt_ensmean_present_coastal',
    # 'matt_ensmean_present_runoff',
    # 'matt_ensmean_future_45cm',
    # 'matt_ensmean_future_45cm_coastal',
    # 'matt_ensmean_future_45cm_runoff',
    # 'matt_ensmean_future_110cm',
    # 'matt_ensmean_future_110cm_coastal',
    # 'matt_ensmean_future_110cm_runoff',
]
gdf = buildings
sbg_res = 5
hmin = 0.1
fld_area = []
stats = []
for key in scenarios:
    print('Calculating info for flood scenario:', key)
    # Load depth raster for model scenario
    fldpth_da = cat.get_rasterdataset(data_like=os.path.join(out_dir, ('floodmap_sbg_0.1_hmin_' + key + '.tif')))

    # Extract depth at building centroids
    hmax = fldpth_da.sel(x=gdf['geometry'].x.to_xarray(),
                         y=gdf['geometry'].y.to_xarray(),
                         method='nearest').values
    gdf[key] = hmax.transpose()
    print(gdf[key].describe())
    stats.append(gdf[key].describe())

    # Calculate flood extent using depth raster
    flooded_cells_mask = (fldpth_da >= hmin)
    flooded_cells_count = np.count_nonzero(flooded_cells_mask)
    flooded_area = flooded_cells_count * (sbg_res * sbg_res) / (1000 ** 2)
    print(key, ' flooded sbg area (sq.km):', round(flooded_area, 2))
    fld_area.append(flooded_area)

stats_out = pd.DataFrame(stats).to_csv(os.path.join(mod.root, 'ch2_flor_future_building_exposure_stats.csv'))

gdf2 = gdf
gdf2['coastal'].fillna(0, inplace=True)
gdf2['runoff'].fillna(0, inplace=True)
gdf2['compound'].fillna(0, inplace=True)

gdf2['max_indiv'] = gdf2[['coastal', 'runoff']].max(axis=1)
gdf2['max_indiv'] = gdf2['max_indiv']
gdf2['diff_depth'] = gdf2['compound'] - gdf2['max_indiv']
print(gdf2['diff_depth'].describe())

gdf_out = gdf2[~(gdf2['compound'].isna() & gdf2['runoff'].isna() & gdf2['coastal'].isna())]
gdf_out.to_file(r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2018_Florence\mod_v6'
                r'\flo_hindcast_v6_200m_LPD2m_avgN\scenarios\00_driver_analysis\flooded_buildings.shp')


##################### CRAZY STUFF #################################
compound_fld_locs = gdf2[gdf2['compound'] > 0.0]
runoff_fld_locs = gdf2[gdf2['runoff'] > 0.0]
coastal_fld_locs = gdf2[gdf2['coastal'] > 0.0]

run_only = runoff_fld_locs[(runoff_fld_locs['compound'] <= 0.0) & (runoff_fld_locs['coastal'] <= 0.0)]
run_only.to_file(r'Z:\users\lelise\projects\Carolinas\Chapter1\sfincs\2018_Florence\mod_v6'
                 r'\flo_hindcast_v6_200m_LPD2m_avgN\scenarios\00_driver_analysis\runoff_flooded_locs_only.shp')
print(run_only.count())
coast_only = coastal_fld_locs[coastal_fld_locs['compound'].isna() & coastal_fld_locs['runoff'].isna()]
print(coast_only.count())

print('No. NC buildings exposed:', len(compound_fld_locs[compound_fld_locs['STATE'] == 'NC']))
print('% of total NC buildings that were exposed:',
      (len(compound_fld_locs[compound_fld_locs['STATE'] == 'NC']) / len(gdf[gdf['STATE'] == 'NC']) * 100))

print('No. SC buildings exposed:', len(compound_fld_locs[compound_fld_locs['STATE'] == 'SC']))
print('% of total SC buildings that were exposed:',
      (len(compound_fld_locs[compound_fld_locs['STATE'] == 'SC']) / len(gdf[gdf['STATE'] == 'SC']) * 100))

fld_exacerbate = compound_fld_locs[compound_fld_locs['diff_depth'] > 0.05]
print('No. of buildings where compound increased flddpth > 0.05m:', len(fld_exacerbate))
y = round((len(fld_exacerbate) / len(compound_fld_locs['compound'])) * 100, 1)
print('% of buildings exposed to compound flooding:', y)
print(fld_exacerbate['diff_depth'].describe())

nofld_to_fld = compound_fld_locs[compound_fld_locs['diff_depth'].isna()]
print('No. of buildings where fld only from compound scenario:', len(nofld_to_fld))
x = round((len(nofld_to_fld) / len(compound_fld_locs['compound'])) * 100, 1)
print('% of buildings exposed to compound flooding:', x)
print(nofld_to_fld['compound'].describe())

# Plotting
font = {'family': 'Arial',
        'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

props = dict(boxes="white", whiskers="black", caps="black")
boxprops = dict(facecolor='white', linestyle='--', linewidth=1, color='black')
flierprops = dict(marker='+', markerfacecolor='none', markersize=3,
                  markeredgecolor='black')
medianprops = dict(linestyle='-', linewidth=2, color='black')
meanpointprops = dict(marker='D',
                      markeredgecolor='black',
                      markerfacecolor='lightgrey',
                      markersize=4)
##############################
fig, axs = plt.subplots(nrows=2, ncols=1,
                        tight_layout=True,
                        figsize=(5, 4),
                        sharex=True, sharey=False)
axs = axs.flatten()
i = 0
ds = fld_exacerbate['diff_depth'] * 100
bp = ds.plot.box(ax=axs[i],
                 vert=False,
                 color=props,
                 boxprops=boxprops,
                 flierprops=flierprops,
                 medianprops=medianprops,
                 meanprops=meanpointprops,
                 meanline=False,
                 showmeans=True,
                 patch_artist=True)
#axs[i].set_xlabel('Increase in Depth (cm)')
axs[i].set_yticklabels('')
axs[i].set_title(f'Increase in Depth at buildings in Compound Scenario from Max Individual \n(n={ds.count()}; {y}% of '
                 f'total exposed)',
                 loc='left')
kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.9)
axs[i].grid(visible=True, which='major', axis='x', **kwargs)
kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.9)
axs[i].grid(visible=True, which='minor', axis='x', **kwargs)
axs[i].set_xscale("log")
axs[i].set_xlim([0, 200])
i = 1
ds = nofld_to_fld['compound'] * 100
bp = ds.plot.box(ax=axs[i],
                 vert=False,
                 color=props,
                 boxprops=boxprops,
                 flierprops=flierprops,
                 medianprops=medianprops,
                 meanprops=meanpointprops,
                 meanline=False,
                 showmeans=True,
                 patch_artist=True)
axs[i].set_xlabel('Water Depth (cm)')
axs[i].set_title(f'Depth at buildings flooded in Compound Scenario Only\n(n={ds.count()}; {x}% of total exposed) ',
                 loc='left')
axs[i].set_yticklabels('')
kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.9)
axs[i].grid(visible=True, which='major', axis='x', **kwargs)
kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.9)
axs[i].grid(visible=True, which='minor', axis='x', **kwargs)
#axs[i].set_xscale("log")
#axs[i].set_xlim([0, 200])
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(out_dir, 'building_exposure_compound_vs_individual.png'),
            dpi=225,
            bbox_inches="tight")
plt.close()

###########################
fig, axs = plt.subplots(nrows=3, ncols=2, tight_layout=True,
                        figsize=(6.5, 5), sharex=True)
axs = axs.flatten()
axs[5].set_visible(False)
for i in range(len(l_gdf['Name'])):
    basin = l_gdf['Name'].iloc[i]
    ds = gdf[gdf['Name'] == basin][fld_map_key].dropna(axis='index', how='all')
    print(fld_map_key)
    scenario_n = [ds['compound'].count(), ds['runoff'].count(), ds['coastal'].count()]
    bp = ds[fld_map_key].plot.box(ax=axs[i],
                                  vert=False,
                                  color=props,
                                  boxprops=boxprops,
                                  flierprops=flierprops,
                                  medianprops=medianprops,
                                  meanprops=meanpointprops,
                                  meanline=False,
                                  showmeans=True,
                                  patch_artist=True)
    axs[i].set_xlabel('Flood Depth (m)')
    axs[i].set_title(basin, loc='left')
    axs[i].set_yticklabels(
        [f'Compound\n(n={scenario_n[0]})', f'Runoff\n(n={scenario_n[2]})', f'Coastal\n(n={scenario_n[1]})'])
    kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.9)
    axs[i].grid(visible=True, which='major', axis='x', **kwargs)
    kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.9)
    axs[i].grid(visible=True, which='minor', axis='x', **kwargs)
    axs[i].set_xscale("log")
plt.setp(axs, xlim=(-1, 20))
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(out_dir, 'scenario_building_depth_by_huc6_V2_test.png'),
            dpi=225,
            bbox_inches="tight")
plt.close()

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(6, 3))
nc = gdf[gdf['STATE'] == 'NC'][fld_map_key].dropna(axis='index', how='all')
nc_n = [nc['compound'].count(), nc['coastal_wse_wind'].count(), nc['discharge_rainfall'].count()]
bp = nc[fld_map_key].plot.box(ax=axs[0],
                              vert=False,
                              color=props,
                              boxprops=boxprops,
                              flierprops=flierprops,
                              medianprops=medianprops,
                              meanprops=meanpointprops,
                              meanline=False,
                              showmeans=True,
                              patch_artist=True)
axs[0].set_xlabel('NC Water Depth at Buildings (m)')
axs[0].set_yticklabels([f'Compound\n(n={nc_n[0]})', f'Coastal\n(n={nc_n[1]})', f'Runoff\n(n={nc_n[2]})'])
kwargs = dict(linestyle='--', linewidth=0.75, color='lightgrey', alpha=0.8)
axs[0].grid(visible=True, which='major', axis='x', **kwargs)
axs[0].set_xscale("log")
axs[0].set_xlim(-1, 15)
pos1 = axs[0].get_position()  # get the original position

sc = gdf[gdf['STATE'] == 'SC'][fld_map_key].dropna(axis='index', how='all')
sc_n = [sc['compound'].count(), sc['coastal_wse_wind'].count(), sc['discharge_rainfall'].count()]
bp = sc[fld_map_key].plot.box(ax=axs[1],
                              vert=False,
                              color=props,
                              boxprops=boxprops,
                              flierprops=flierprops,
                              medianprops=medianprops,
                              meanprops=meanpointprops,
                              meanline=False,
                              showmeans=True,
                              patch_artist=True)
axs[1].set_xlabel('SC Water Depth at Buildings (m)')
axs[1].set_xscale("log")
axs[1].set_xlim(-1, 15)
axs[1].set_yticklabels([f'Compound\n(n={sc_n[0]})', f'Coastal\n(n={sc_n[1]})', f'Runoff\n(n={sc_n[2]})'])
axs[1].grid(visible=True, which='major', axis='x', **kwargs)

plt.margins(x=0, y=0)
plt.savefig(os.path.join(out_dir, 'depth_at_structures_by_state_logscale.png'),
            dpi=225,
            bbox_inches="tight")
plt.close()
