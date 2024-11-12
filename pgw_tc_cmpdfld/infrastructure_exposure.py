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


# Load data
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3')
da_zsmax = xr.open_dataset('pgw_zsmax.nc', engine='netcdf4')
da_vmax = xr.open_dataset('pgw_vmax.nc', engine='netcdf4')
da_tmax = xr.open_dataset('pgw_tmax.nc', engine='netcdf4')


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

''' 
PART 2 - Extract flood depths and process attribution 
'''
gdf = buildings.copy()
gdf['xcoords'] = gdf['geometry'].x.to_xarray()
gdf['ycoords'] = gdf['geometry'].y.to_xarray()

# Get mean values
das = [da_zsmax, da_vmax, da_tmax]
das_mean = []
for da in das:
    run_ids = da.run.values
    subset_list = [r for r in run_ids if 'flor_fut' in r]
    subset_list = [r for r in subset_list if 'compound' in r]
    subset_list = [r for r in subset_list if 'SF8' not in r]
    da_mean = da.sel(run=subset_list).mean(dim='run')
    das_mean.append(da_mean)

vars = ['zsmax', 'vmax', 'tmax']
for i in range(len(vars)):
    da =  das_mean[i][vars[i]]
    v = da.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'fut_{vars[i]}'] = v.transpose()

# Extract gnd elevation at buildings
dep = xr.open_dataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_20m.tif')
depv = dep['band_data'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['gnd_elev'] = depv.transpose()

# Present
da = da_zsmax.sel(run='flor_pres_compound')
v = da['zsmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'pres_zsmax'] = v.transpose()

da = da_vmax.sel(run='flor_pres_compound')
v = da['vmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'pres_vmax'] = v.transpose()

da = da_tmax.sel(run='flor_pres_compound')
v = da['tmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf[f'pres_tmax'] = v.transpose()

# Haz = H * (Vc + 1.5)

gdf['fut_depth'] = gdf['fut_zsmax'] - gdf['gnd_elev']
gdf['pres_depth'] = gdf['pres_zsmax'] - gdf['gnd_elev']

flooded = gdf[(gdf['fut_depth'] > 0) | (gdf['pres_depth'] > 0)]

flooded['fut_depth'][flooded['fut_depth'] <= 0] = 0
flooded['pres_depth'][flooded['pres_depth'] <= 0] = 0.0

flooded.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\florence_pres_fut_buildings.csv')
