#!/usr/bin/env python
# coding: utf-8

import os
import re
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import fiona
import xarray as xr
import matplotlib as mpl
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel

# Filepath to data catalog yml
cat_dir = r'Z:\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
os.chdir(r'Z:\users\lelise\projects\Carolinas_SFINCS\Chapter1_FlorenceValidation\sfincs_models\mod_v4_flor')
model_root = r'ENC_200m_sbg5m_avgN_adv1_eff75'
mod = SfincsModel(model_root, mode='r', data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
cat = mod.data_catalog
print(f'Reading in SFINCS model: {model_root}')

''' Load in preprocessed model results to extract at the buildings including '''
# (1) downscaled eak flood depths
# (2) peak water levels
# (3) compound peak water level minus max individual processes/driver

res = 200
da = xr.open_dataset(os.path.join(os.getcwd(), 'floodmaps', f'{res}m', 'floodmaps.nc'))
dep = xr.open_dataset(os.path.join(os.getcwd(), 'subgrid', 'dep_subgrid.tif'))
print(f'Reading downscaled floodmaps at {res}m resolution (created with downscale_floodmaps.py) and the subgrid.')

# Create output directory if it doesn't already exist
out_dir = os.path.join(os.getcwd(), 'process_attribution', f'{res}m')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
os.chdir(out_dir)

# Load files created using "compound_flooding_analysis.py" script
da_class = xr.open_dataarray('flor_peakWL_attributed_all.nc')
da_diff = xr.open_dataset('flor_peakWL_compound_minus_maxIndiv_all.nc')
print(f'Reading in peak water levels and the difference in compound minus individual dirvers/processes.')

''' PART 1 - Determine damage status using NFIP claims/policy data at each structure'''
# Read in area of interest shapefile and project
studyarea_gdf = mod.region.to_crs(epsg=32617)

# Read in structures information and clip to the study area
nc_buildings = gpd.read_file(r'Z:\users\lelise\data\storm_data\hurricanes\X_observations\nfip_flood_damage_NC'
                             r'\included_data.gdb',
                             layer='buildings',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
nc_buildings['STATE'] = 'NC'
b1 = nc_buildings.drop(nc_buildings.columns[~nc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of NC Buildings in Study Area:', str(len(nc_buildings)))

# Load SC buildings from NSI
sc_buildings = gpd.read_file(r'Z:\users\lelise\data\geospatial\infrastructure\nsi_2022_45.gpkg',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
sc_buildings['STATE'] = 'SC'
b2 = sc_buildings.drop(sc_buildings.columns[~sc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of SC Buildings in Study Area:', str(len(sc_buildings)))

# Combine NC and SC data into single dataframe
buildings = pd.concat(objs=[b1, b2], axis=0, ignore_index=True)

# Join buildings data to HUC6 watershed, this will take a while...
basins = cat.get_geodataframe(
    r'Z:\users\lelise\data\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
basins = basins[basins['Name'].isin(['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee'])]
basins.to_crs(epsg=32617, inplace=True)
basins = basins[["HUC6", "Name", "geometry"]]
buildings = gpd.tools.sjoin(left_df=buildings, right_df=basins, how='left')

print(buildings.groupby('Name').count())

''' PART 2 - Extract flood depths and process attribution '''
# Extract depth at building centroids
gdf = buildings.copy()
gdf['xcoords'] = gdf['geometry'].x.to_xarray()
gdf['ycoords'] = gdf['geometry'].y.to_xarray()

# Rename run IDs
rename = ['compound', 'coastal', 'runoff', 'discharge', 'rainfall', 'stormTide', 'wind']
da['run'] = xr.IndexVariable('run', rename)
print(da.run.values)

# Extract water depths from compound scenario
da_fldp = da.sel(run='compound')
hmax = da_fldp['hmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['hmax'] = hmax.transpose()

# Extract water depths from coastal scenario
da_fldp = da.sel(run='coastal')
hmax = da_fldp['hmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['hmax_coastal'] = hmax.transpose()

# Extract water depths from compound scenario
da_fldp = da.sel(run='runoff')
hmax = da_fldp['hmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['hmax_runoff'] = hmax.transpose()

# Extract attribution code
hmax_class = da_class.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['hmax_class'] = hmax_class.transpose()

# Extract diff in compound minus max individual
hmax_diff = da_diff.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['hmax_diff'] = hmax_diff.transpose()

# Extract gnd elevation at buildings
depv = dep.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['gnd_elev'] = depv.transpose()

# Save data
gdf2 = gdf.copy()
gdf2.to_csv(os.path.join(out_dir, 'building_pts_with_depth.csv'))

gdf = pd.read_csv(os.path.join(out_dir, 'building_pts_with_depth.csv'), index_col=0)

gdf = gdf[gdf['hmax_class'] > 0]
gdf.to_csv(os.path.join(out_dir, 'building_pts_with_depth_floodedOnly.csv'))

''' Part 3 - Get STATS '''
# Reclass coastal compound and runoff compound to 5
gdf['hmax_class'][gdf['hmax_class'] == 2] = 5
gdf['hmax_class'][gdf['hmax_class'] == 4] = 5

mast_stats = pd.DataFrame()
for threshold in [0, 0.15, 0.5, 1.0, 1.5]:
    for classification in [1, 3, 5]:
        sub = gdf[gdf['hmax_class'] == classification]
        sub = sub[sub['hmax'] > threshold]
        dep_stats = pd.DataFrame(sub['hmax'].describe(percentiles=[0.05, 0.10, 0.50, 0.90, 0.95])).T
        gnd_dep_stats = pd.DataFrame(sub['gnd_elev'].describe(percentiles=[0.05, 0.10, 0.50, 0.90, 0.95])).T
        dep_dif_stats = pd.DataFrame(sub['hmax_diff'].describe(percentiles=[0.05, 0.10, 0.50, 0.90, 0.95])).T

        df = pd.concat([dep_stats, dep_dif_stats, gnd_dep_stats], axis=0, ignore_index=True)
        df['dep_thresh'] = threshold
        df['class'] = classification
        df['stat_ID'] = ['hmax', 'hmax_diff', 'gnd_elev']

        mast_stats = pd.concat([mast_stats, df], axis=0, ignore_index=True)
mast_stats = mast_stats.round(2)
mast_stats.to_csv(os.path.join(out_dir, 'building_depth_stats_byClass_Threshold.csv'), index=False)

mast_stats = pd.DataFrame()
for threshold in [0, 0.15, 1.0]:
    for classification in [1, 3, 5]:
        sub = gdf[gdf['hmax_class'] == classification]
        sub = sub[sub['hmax'] > threshold]
        dep_stats = pd.DataFrame(sub.groupby('Name')['hmax'].describe(percentiles=[0.05, 0.10, 0.50, 0.90, 0.95]))
        dep_stats['Name'] = dep_stats.index
        dep_stats['dep_thresh'] = threshold
        dep_stats['type'] = 'hmax'
        dep_stats['classification'] = classification

        dep_dif_stats = pd.DataFrame(
            sub.groupby('Name')['hmax_diff'].describe(percentiles=[0.05, 0.10, 0.50, 0.90, 0.95]))
        dep_dif_stats['Name'] = dep_dif_stats.index
        dep_dif_stats['dep_thresh'] = threshold
        dep_dif_stats['type'] = 'hmax_diff'
        dep_dif_stats['classification'] = classification

        gnd_dep_stats = pd.DataFrame(
            sub.groupby('Name')['gnd_elev'].describe(percentiles=[0.05, 0.10, 0.50, 0.90, 0.95]))
        gnd_dep_stats['Name'] = gnd_dep_stats.index
        gnd_dep_stats['dep_thresh'] = threshold
        gnd_dep_stats['type'] = 'gnd_elev'
        gnd_dep_stats['classification'] = classification

        df = pd.concat([dep_stats, dep_dif_stats, gnd_dep_stats], axis=0, ignore_index=True)

        mast_stats = pd.concat([mast_stats, df], axis=0, ignore_index=True)

mast_stats = mast_stats.round(2)
mast_stats.to_csv(os.path.join(out_dir, 'building_depth_stats_byClass_Threshold_byHUC.csv'), index=False)

''' Part 4 - Plotting '''
gdf['group'] = ''
gdf['group'][gdf['hmax_class'] == 1] = 'Coastal'
gdf['group'][gdf['hmax_class'] == 3] = 'Runoff'
gdf['group'][gdf['hmax_class'] == 5] = 'Compound'

import seaborn as sns

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
flierprops = dict(marker='+', markerfacecolor='none', markersize=3, markeredgecolor='black')
medianprops = dict(linestyle='-', linewidth=2, color='black')
meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='lightgrey', markersize=4)
for threshold in [0.15, 1]:
    ds = gdf[gdf['hmax_class'] > 0]
    ds = ds[ds['hmax'] > threshold]
    bb = ['Lower Pee Dee', 'Cape Fear', 'Onslow Bay', 'Neuse', 'Pamlico', 'Domain']
    fig, axs = plt.subplots(nrows=3, ncols=2, tight_layout=True, figsize=(6.2, 5),
                            sharex=True, sharey=False)
    axs = axs.flatten()
    for i in range(len(bb)):
        ax = axs[i]
        if i == 5:
            dsb = ds
        else:
            dsb = ds[ds['Name'] == bb[i]]
        codes, counts = np.unique(dsb['group'], return_counts=True)
        print(bb[i])
        print(codes, counts)
        bp = sns.boxplot(data=dsb,
                         x='hmax', y='group',
                         ax=ax,
                         order=['Runoff', 'Coastal', 'Compound'],
                         orient='h',
                         color='white', linecolor='black', linewidth=0.75, width=0.7, gap=0.2,
                         flierprops=flierprops,
                         medianprops=medianprops,
                         meanprops=meanpointprops,
                         meanline=False,
                         showmeans=True,
                         patch_artist=True,
                         )

        ax.set_xlabel('Flood Depth (m)')
        ax.set_ylabel('')
        ax.set_title('')
        ax.set_title(bb[i], loc='center')
        kwargs = dict(linestyle='-', linewidth=1, color='grey', alpha=0.9)
        ax.grid(visible=True, which='major', axis='x', **kwargs)
        kwargs = dict(linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.9)
        ax.grid(visible=True, which='minor', axis='x', **kwargs)
        ax.set_xscale("log")

        ytl_new = []
        for c in ['Runoff', 'Coastal', 'Compound']:
            try:
                ind = codes.tolist().index(c)
                count = counts[ind]
                text_new = f'{c}\n(n={count})'
                ytl_new.append(text_new)
            except:
                ytl_new.append('')
        ax.set_yticklabels(ytl_new)

    plt.setp(axs, xlim=(0, 20), ylim=(-0.5, 2.5))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, f'building_exposure_{threshold}.png'), dpi=225, bbox_inches="tight")
    plt.close()

    # Compound minus max individual
    ds = gdf[gdf['hmax_class'] == 5]
    ds = ds[ds['hmax'] > threshold]

    ds2 = ds.copy()
    ds2['Name'] = 'Domain'
    ds2 = pd.concat([ds, ds2], axis=0, ignore_index=True)

    counts = ds2.groupby('Name')['hmax_diff'].count()
    fig, axs = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(6.2, 3), sharex=False, sharey=False)
    ax = axs[0]
    vp = sns.violinplot(data=ds2,
                        x='hmax_diff',
                        y='Name',
                        ax=ax,
                        density_norm='width',
                        common_norm=True,
                        fill=False, gap=0.05, linewidth=0.75,
                        color='black',
                        inner_kws=dict(box_width=3, whis_width=0.75, color="black")
                        )
    ax.set_ylabel('')
    ax.set_xscale('log')
    ax.set_xlim(-.1, 2.5)
    ax.set_xlabel('Depth Difference (m)\ncompound - max. individual')
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title('(a)', loc='center', fontsize=10)
    ax.set_yticklabels(labels=[f'P (n={counts[5]})',
                               f'LPD (n={counts[2]})',
                               f'CF (n={counts[0]})',
                               f'OB (n={counts[4]})',
                               f'N (n={counts[3]})',
                               f'Domain\n(n={counts[1]})'
                               ], rotation=0)

    ax = axs[1]
    colors = ['black', "lightgray", "gray", "darkgray", "gray"]
    basin = ['Onslow Bay', 'Lower Pee Dee', 'Neuse', 'Cape Fear', 'Pamlico']
    legend_nick = ['OB', 'LPD', 'N', 'CF', 'P']
    marker = ['x', "o", "^", "s", "d", ]

    for i in range(len(basin)):
        dd = ds[ds['Name'] == basin[i]]
        ax.scatter(x=dd['hmax_diff'], y=dd['gnd_elev'],
                   color=colors[i],
                   marker=marker[i],
                   s=20, edgecolors='black', alpha=0.9,
                   )
    ax.legend(legend_nick, loc='upper right', fontsize=10)
    ax.set_ylabel('Ground Elevation\n(m+NAVD88)')
    ax.set_xlabel('Depth Difference (m)\ncompound - max. individual')
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 2.1)
    ax.set_title('(b)', loc='center', fontsize=10)
    # ax.set_xscale('log')

    plt.savefig(os.path.join(out_dir, f'building_exposure_{threshold}_wlDiff.png'), dpi=225, bbox_inches="tight")
    plt.close()
