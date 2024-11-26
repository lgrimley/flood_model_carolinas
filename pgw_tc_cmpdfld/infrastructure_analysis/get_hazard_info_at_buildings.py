#!/usr/bin/env python
# coding: utf-8
import os
import re
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import xarray as xr
import matplotlib as mpl
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel


# Filepath to data catalog yml
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
cat = mod.data_catalog
studyarea_gdf = mod.region.to_crs(epsg=32617)


''' 

Read in NC and SC buildings data 

'''
# Read in structures information and clip to the study area. This might take a little while and only needs to be run once...
nc_buildings = gpd.read_file(filename=r'Z:\Data-Expansion\users\lelise\data\storm_data\hurricanes\X_observations\nfip_flood_damage_NC'
                             r'\included_data.gdb',
                             layer='buildings',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
nc_buildings['STATE'] = 'NC'
b1 = nc_buildings.drop(nc_buildings.columns[~nc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of NC Buildings in Study Area:', str(len(nc_buildings)))

# Load SC buildings from NSI. This might take a little while and only needs to be run once...
sc_buildings = gpd.read_file(filename=r'Z:\Data-Expansion\users\lelise\data\geospatial\infrastructure\nsi_2022_45.gpkg',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
sc_buildings['STATE'] = 'SC'
b2 = sc_buildings.drop(sc_buildings.columns[~sc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
print('Number of SC Buildings in Study Area:', str(len(sc_buildings)))

# Combine NC and SC data into single dataframe
buildings = pd.concat(objs=[b1, b2], axis=0, ignore_index=True)
print('Number of Buildings in Study Area:', str(len(buildings)))

# Make a copy of the building data to start adding model output to
gdf = buildings.copy()
gdf['xcoords'] = gdf['geometry'].x.to_xarray()
gdf['ycoords'] = gdf['geometry'].y.to_xarray()

'''

LOAD MODEL OUTPUTS

'''

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3')
da_zsmax = xr.open_dataset('pgw_zsmax.nc', engine='netcdf4')  # Max water level
da_vmax = xr.open_dataset('pgw_vmax.nc', engine='netcdf4')  # Max velocity
da_tmax = xr.open_dataset('pgw_tmax.nc', engine='netcdf4')  # Time of inundation
da_zsmax_class = xr.open_dataset(r'.\process_attribution\processes_classified.nc')
da_zsmax_ensmean = xr.open_dataset(r'.\ensemble_mean\fut_ensemble_zsmax_mean.nc')
da_zsmax_ensmean_class = xr.open_dataset(r'.\ensemble_mean\processes_classified_ensmean_mean.nc')
dep = xr.open_dataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_20m.tif')


# GUT RUN IDS FOR HURRICANE FLORENCE ONLY
run_ids = da_zsmax.run.values
subset_list = [r for r in run_ids if 'flor_fut' in r]
subset_list = [r for r in subset_list if 'compound' in r]
subset_list = [r for r in subset_list if 'SF8' not in r]
print(subset_list)


''' 

Extract model output at buildings 
 
'''

# PRESENT PEAK WATER LEVEL
da_zsmax_pres = da_zsmax.sel(run='flor_pres_compound')
v = da_zsmax_pres['zsmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'pres_zsmax'] = v.transpose()


# FUTURE ENSEMBLE MEAN PEAK WATER LEVEL
da_zsmax_fut = da_zsmax_ensmean.sel(run='flor_fut_compound_mean')
v = da_zsmax_fut['flor_fut_coastal_mean'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'fut_zsmax'] = v.transpose()

# GROUND ELEVATION
v = dep['band_data'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['gnd_elev'] = v.transpose()

# PEAK FLOOD DEPTH PRESENT, FUTURE, AND DIFFERENCE
gdf['fut_depth'] = gdf['fut_zsmax'] - gdf['gnd_elev']
gdf['pres_depth'] = gdf['pres_zsmax'] - gdf['gnd_elev']
gdf['depth_diff'] = gdf['fut_depth'] - gdf['pres_depth']

''' 

Subset the buildings based on flood depths - extract velocity, tmax, flood process classification

'''
# SUBSET BUILDINGS TO THOSE THAT FLOODED IN THE PRESENT OR FUTURE GREATER THAN THRESHOLD
hmin = 0.1
gdf_fld = gdf[(gdf['fut_depth'] > hmin) | (gdf['pres_depth'] > hmin)]
print(len(gdf_fld))
print(gdf_fld['depth_diff'].describe())

# PRESENT PEAK VELOCITY
da_vmax_pres = da_vmax.sel(run='flor_pres_compound')
v = da_vmax_pres['vmax'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
gdf_fld[f'pres_vmax'] = v.transpose()

# FUTURE MEAN PEAK VELOCITY
da_vmax_fut = da_vmax.sel(run=subset_list).mean(dim='run')
v = da_vmax_fut['vmax'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
gdf_fld[f'fut_vmax'] = v.transpose()

# Difference in Future and Present Peak Velocity
gdf_fld[f'vmax_diff'] = gdf_fld[f'fut_vmax'] - gdf_fld[f'pres_vmax']

# PRESENT TIME OF INUNDATION
da_tmax_pres = da_tmax.sel(run='flor_pres_compound')
v = da_tmax_pres['tmax'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
gdf_fld[f'pres_tmax'] = v.transpose()

# FUTURE MEAN TIME OF INUNDATION
da_tmax_fut = da_tmax.sel(run=subset_list).mean(dim='run')
v = da_tmax_fut['tmax'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
gdf_fld[f'fut_tmax'] = v.transpose()

# Difference in Future and Present Tmax
gdf_fld[f'tmax_diff'] = gdf_fld[f'fut_tmax'] - gdf_fld[f'pres_tmax']

# FLOOD PROCESS CLASSIFICATION
# Present
da_zsmax_class_pres = da_zsmax_class.sel(run='flor_pres')
v = da_zsmax_class_pres['flor_pres'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
gdf_fld[f'pres_zsmax_class'] = v.transpose()

# FUTURE ENSEMBLE MEAN
da_zsmax_ensmean_class = da_zsmax_ensmean_class.sel(run='flor_fut_ensmean')
v = da_zsmax_ensmean_class['flor_fut_ensmean'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
gdf_fld[f'fut_zsmax_class'] = v.transpose()

gdf_fld2 = np.round(gdf_fld, decimals=3)
gdf_fld2.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\infrastructure_exposure\florence_buildings_exposed.csv')


test = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\infrastructure_exposure\florence_buildings_exposed.csv', index_col=0)
test = test.drop('geometry', axis=1)
test2 = gpd.GeoDataFrame(test, geometry=gpd.points_from_xy(x=test['xcoords'].values, y=test['ycoords'].values, crs=32617))

test2['pres_zsmax_class'][(test2['pres_zsmax_class'] == 2) | (test2['pres_zsmax_class'] == 4)] = 5
print(np.unique(test2['pres_zsmax_class']))





