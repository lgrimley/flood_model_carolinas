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
from shapely.constructive import buffer

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

Read data

'''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\infrastructure_exposure')
washouts = pd.read_csv('NCDOT_Florence_washouts.csv')
washouts = gpd.GeoDataFrame(washouts,
                            geometry=gpd.points_from_xy(x=washouts['xcoord'], y=washouts['ycoord'], crs=4326))
# Make a copy of the building data to start adding model output to
gdf = washouts.copy().to_crs(32617)
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

Extract model output  
 
'''
# PRESENT PEAK WATER LEVEL
da_zsmax_pres = da_zsmax.sel(run='flor_pres_compound')
v = da_zsmax_pres['zsmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'pres_zsmax'] = v.transpose()

# PRESENT PEAK VELOCITY
da_vmax_pres = da_vmax.sel(run='flor_pres_compound')
v = da_vmax_pres['vmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'pres_vmax'] = v.transpose()

# PRESENT TIME OF INUNDATION
da_tmax_pres = da_tmax.sel(run='flor_pres_compound')
v = da_tmax_pres['tmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'pres_tmax'] = v.transpose()

# FLOOD PROCESS CLASSIFICATION
da_zsmax_class_pres = da_zsmax_class.sel(run='flor_pres')
v = da_zsmax_class_pres['flor_pres'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'pres_zsmax_class'] = v.transpose()

# GROUND ELEVATION
v = dep['band_data'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['gnd_elev'] = v.transpose()

# PEAK FLOOD DEPTH
gdf['pres_depth'] = gdf['pres_zsmax'] - gdf['gnd_elev']

'''

Hit/Miss

'''

downscaled = xr.open_dataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\flood_depths_20m\flor_pres_compound.nc')
#downscaled['flor_pres_compound'].raster.to_raster(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\flood_depths_20m\florence_pres_compound.tif')

# GROUND ELEVATION
data = downscaled['flor_pres_compound']
gdf_buff = gdf.copy()
clip = gdf_buff.clip(mod.region)
gdf_buff['geometry'] = gdf_buff.buffer(60)
mask = data.raster.rasterize(gdf_buff, 'OBJECTID', nodata=-9999.0, all_touched=True)
#mask.raster.to_raster(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\flood_depths_20m\test.tif', nodata=-9999.0)

# Loop through the basins and calculate the basin average
max_depth = []
for i in range(len(gdf_buff)):
    point_id = gdf_buff['OBJECTID'][i]
    hmax = data.where(mask == point_id).max(dim=['x', 'y'])
    max_depth.append(hmax.values.item())

gdf_buff['buff_depth'] = max_depth

hit = gdf_buff[~gdf_buff['buff_depth'].isna()]

