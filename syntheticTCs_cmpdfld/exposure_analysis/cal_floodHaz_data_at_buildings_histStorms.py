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

LOAD MODEL OUTPUTS

'''

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final')
da_zsmax = xr.open_dataset('pgw_zsmax.nc', engine='netcdf4')  # Max water level
da_vmax = xr.open_dataset('pgw_vmax.nc', engine='netcdf4')  # Max velocity
da_tmax = xr.open_dataset('pgw_tmax.nc', engine='netcdf4')  # Time of inundation
da_zsmax_class = xr.open_dataset(r'.\process_attribution\processes_classified.nc')
da_zsmax_ensmean = xr.open_dataset(r'.\ensemble_mean\fut_ensemble_zsmax_mean.nc')
da_zsmax_ensmean_class = xr.open_dataset(r'.\ensemble_mean\processes_classified_ensmean_mean.nc')
dep = xr.open_dataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_20m.tif')


# GUT RUN IDS FOR HURRICANE FLORENCE ONLY
run_ids = da_zsmax.run.values
storm = 'flor'
scenario = 'compound'
subset_list = [r for r in run_ids if storm in r]
subset_list = [r for r in subset_list if scenario in r]
subset_list = [r for r in subset_list if 'SF8' not in r]
print(subset_list)


''' 
Extract model output at buildings 
'''

# GROUND ELEVATION
v = dep['band_data'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
gdf['gnd_elev'] = v.transpose()
#small_file = gdf[['BLDG_ID','PID','xcoords','ycoords', 'gnd_elev']]
#small_file.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure\buildings_within_domain.csv')


for storm in ['floy', 'matt']:
    # FLOOD PROCESS CLASSIFICATION
    # Present
    d = da_zsmax_class.sel(run=f'{storm}_pres')
    d = d.rename({list(d.keys())[0]:'class'})
    v = d['class'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'{storm}_hclass'] = v.transpose()

    # Future
    d = da_zsmax_ensmean_class.sel(run=f'{storm}_fut_ensmean')
    d= d.rename({list(d.keys())[0]:'class'})
    v = d['class'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'{storm}_pclass'] = v.transpose()

    for scenario in ['compound', 'runoff', 'coastal']:
        name_prefix = f'{storm}_{scenario}'
        # PRESENT PEAK WATER LEVEL
        da_zsmax_pres = da_zsmax.sel(run=f'{storm}_pres_{scenario}')
        v = da_zsmax_pres['zsmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
        gdf[f'{name_prefix}_hzsmax'] = v.transpose()

        # FUTURE ENSEMBLE MEAN PEAK WATER LEVEL
        da_zsmax_fut = da_zsmax_ensmean.sel(run=f'{storm}_fut_{scenario}_mean')
        da_zsmax_fut = da_zsmax_fut.rename({list(da_zsmax_fut.keys())[0]:'zsmax'})
        v = da_zsmax_fut['zsmax'].sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
        gdf[f'{name_prefix}_pzsmax'] = v.transpose()

        # PEAK FLOOD DEPTH PRESENT, FUTURE, AND DIFFERENCE
        gdf[f'{name_prefix}_pdepth'] = gdf[f'{name_prefix}_pzsmax'] - gdf['gnd_elev']
        gdf[f'{name_prefix}_hdepth'] = gdf[f'{name_prefix}_hzsmax'] - gdf['gnd_elev']
        gdf[f'{name_prefix}_depth_diff'] = gdf[f'{name_prefix}_pzsmax'] - gdf[f'{name_prefix}_hzsmax']
        print(scenario)
    print(storm)

fld_build = gdf[gdf[[
       'flor_compound_hzsmax', 'flor_compound_pzsmax',
       'flor_runoff_hzsmax', 'flor_runoff_pzsmax', 'flor_coastal_hzsmax',
       'flor_coastal_pzsmax',
       'floy_compound_hzsmax', 'floy_compound_pzsmax',
       'floy_runoff_hzsmax', 'floy_runoff_pzsmax',  'floy_coastal_hzsmax',
       'floy_coastal_pzsmax',
       'matt_compound_hzsmax', 'matt_compound_pzsmax',
       'matt_runoff_hzsmax', 'matt_runoff_pzsmax',  'matt_coastal_hzsmax',
       'matt_coastal_pzsmax',
        ]].notna().any(axis=1)]
outfile = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure\buildings_tc_exposure_rp_real.csv'
fld_build.to_csv(outfile)


# '''
# Subset the buildings based on flood depths - extract velocity, tmax, flood process classification
# '''
# # SUBSET BUILDINGS TO THOSE THAT FLOODED IN THE PRESENT OR FUTURE GREATER THAN THRESHOLD
# hmin = 0.1
# gdf_fld = gdf[(gdf['fut_depth'] > hmin) | (gdf['pres_depth'] > hmin)]
# print(len(gdf_fld))
# print(gdf_fld['depth_diff'].describe())
#
# # PRESENT PEAK VELOCITY
# da_vmax_pres = da_vmax.sel(run='flor_pres_compound')
# v = da_vmax_pres['vmax'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
# gdf_fld[f'pres_vmax'] = v.transpose()
#
# # FUTURE MEAN PEAK VELOCITY
# da_vmax_fut = da_vmax.sel(run=subset_list).mean(dim='run')
# v = da_vmax_fut['vmax'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
# gdf_fld[f'fut_vmax'] = v.transpose()
#
# # Difference in Future and Present Peak Velocity
# gdf_fld[f'vmax_diff'] = gdf_fld[f'fut_vmax'] - gdf_fld[f'pres_vmax']
#
# # PRESENT TIME OF INUNDATION
# da_tmax_pres = da_tmax.sel(run='flor_pres_compound')
# v = da_tmax_pres['tmax'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
# gdf_fld[f'pres_tmax'] = v.transpose()
#
# # FUTURE MEAN TIME OF INUNDATION
# da_tmax_fut = da_tmax.sel(run=subset_list).mean(dim='run')
# v = da_tmax_fut['tmax'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
# gdf_fld[f'fut_tmax'] = v.transpose()
#
# # Difference in Future and Present Tmax
# gdf_fld[f'tmax_diff'] = gdf_fld[f'fut_tmax'] - gdf_fld[f'pres_tmax']




# # For Marissa
# da_zsmax_class_pres_MW = da_zsmax_class_pres
# da_zsmax_class_pres_MW2 = xr.where(cond=((da_zsmax_class_pres_MW == 2) | (da_zsmax_class_pres_MW == 4)),
#                                   x=5, y=da_zsmax_class_pres_MW)
# da_zsmax_class_pres_MW2 = xr.where(cond=(da_zsmax_class_pres_MW2 == 0),
#                                   x=np.nan, y=da_zsmax_class_pres_MW2)
# r = da_zsmax_class_pres_MW2['flor_pres'].raster
# r.set_crs(32617)
# r.to_raster(r'Z:\Data-Expansion\users\lelise\data_share\SFINCS_OUTPUT\pgw_version_20240720\classified\florence_present_peakWL.tif', nodata=0)
#
# # FUTURE ENSEMBLE MEAN
# da_zsmax_ensmean_class = da_zsmax_ensmean_class.sel(run='flor_fut_ensmean')
# v = da_zsmax_ensmean_class['flor_fut_ensmean'].sel(x=gdf_fld['geometry'].x.to_xarray(), y=gdf_fld['geometry'].y.to_xarray(), method='nearest').values
# gdf_fld[f'fut_zsmax_class'] = v.transpose()
#
# gdf_fld2 = np.round(gdf_fld, decimals=3)
# gdf_fld2.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\infrastructure_exposure\florence_buildings_exposed.csv')
#
# da_zsmax_class_fut_MW = da_zsmax_ensmean_class
# da_zsmax_class_fut_MW = xr.where(cond=((da_zsmax_class_fut_MW == 2) | (da_zsmax_class_fut_MW == 4)),
#                                   x=5, y=da_zsmax_class_fut_MW)
# da_zsmax_class_fut_MW = xr.where(cond=(da_zsmax_class_fut_MW == 0),
#                                   x=np.nan, y=da_zsmax_class_fut_MW)
# r = da_zsmax_class_fut_MW['flor_fut_ensmean'].raster
# r.set_crs(32617)
# r.to_raster(r'Z:\Data-Expansion\users\lelise\data_share\SFINCS_OUTPUT\pgw_version_20240720\classified\florence_future_peakWL.tif', nodata=0)






