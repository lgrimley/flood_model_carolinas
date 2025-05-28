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
mpl.use('TkAgg')
plt.ion()


# Filepath to data catalog yml
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
cat = mod.data_catalog

# We are going to clip the buildings to the model study area
studyarea_gdf = mod.region.to_crs(epsg=32617)


''' 
Read in NC and SC buildings data 
'''

# Read in structures information and clip to the study area. This might take a little while and only needs to be run once...
nc_buildings = gpd.read_file(filename=r'Z:\Data-Expansion\users\lelise\data\geospatial\NC_Buildings_p.gdb\NC_Buildings_p.gdb',
                             layer='NC_Buildings',
                             mask=studyarea_gdf).to_crs(studyarea_gdf.crs)
nc_buildings['geometry'] = nc_buildings.centroid
nc_buildings['STATE'] = 'NC'
#b1 = nc_buildings.drop(nc_buildings.columns[~nc_buildings.columns.isin(['STATE', 'geometry'])], axis=1)
b1 = nc_buildings
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
Extract ground elevation at building centroids 
'''

elevation_da = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_5m.tif')
ground = elevation_da.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['gnd_elev_m'] = ground.transpose()

# Write out a cleaned CSV of the building data
print(gdf.columns)
gdf.drop(columns=['Shape_Length','Shape_Area', 'CID', 'geometry'], inplace=True)
gdf.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure\SFINCS_buildings.csv')