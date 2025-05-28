#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from hydromt_sfincs import SfincsModel
import time
import numpy as np
import geopandas as gpd
start_time = time.time()


# Filepath to data catalog yml
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_sfincs_Carolinas])
cat = mod.data_catalog
studyarea_gdf = mod.region.to_crs(epsg=32617)

# Connect to the working directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure')

# Read in NC and SC buildings data that was previously clipped
buildings_gdf = pd.read_csv('SFINCS_buildings.csv',  index_col=0, low_memory=True)
gdf = gpd.GeoDataFrame(buildings_gdf, geometry=gpd.points_from_xy(x=buildings_gdf['xcoords'],
                                                                  y=buildings_gdf['ycoords'], crs=32617))

# Read in SFINCS MODEL OUTPUTS
results_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final'
# Max water level
da_zsmax = cat.get_rasterdataset(os.path.join(results_dir,'pgw_zsmax.nc'), crs=32617)
# Max velocity
#da_vmax = cat.get_rasterdataset(os.path.join(results_dir,'pgw_vmax.nc'), crs=32617)
# Time of inundation
#da_tmax = cat.get_rasterdataset(os.path.join(results_dir,'pgw_tmax.nc'), crs=32617)
# Water level attribution code (coastal, runoff, compound)
da_zsmax_class = cat.get_rasterdataset(os.path.join(results_dir, r'.\process_attribution\processes_classified.nc'), crs=32617)
# Future ensemble mean water level and attribution code
da_zsmax_ensmean = cat.get_rasterdataset(os.path.join(results_dir, r'.\ensemble_mean\fut_ensemble_zsmax_mean.nc'), crs=32617)
da_zsmax_ensmean_class = cat.get_rasterdataset(os.path.join(results_dir,r'.\ensemble_mean\processes_classified_ensmean_mean.nc'), crs=32617)

''' Extract model output at buildings '''
for storm in ['flor', 'floy', 'matt']:
    # Present (Hindcast)
    data = da_zsmax_class.sel(run=f'{storm}_pres')
    vals = data.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'{storm}_pres_class'] = np.round(vals.transpose(), decimals=3)

    # Future
    data = da_zsmax_ensmean_class.sel(run=f'{storm}_fut_ensmean')
    vals = data.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'{storm}_fut_class'] = np.round(vals.transpose(), decimals=3)

    for scenario in ['compound']:
        name_prefix = f'{storm}_{scenario}'
        # PRESENT PEAK WATER LEVEL
        da_zsmax_pres = da_zsmax.sel(run=f'{storm}_pres_{scenario}')
        vals = da_zsmax_pres.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
        gdf[f'{name_prefix}_pres_zsmax'] = np.round(vals.transpose(), decimals=3)
        gdf[f'{name_prefix}_pres_hmax'] = gdf[f'{name_prefix}_pres_zsmax'] - gdf['elev_sbg5m']

        # FUTURE ENSEMBLE MEAN PEAK WATER LEVEL
        da_zsmax_fut = da_zsmax_ensmean.sel(run=f'{storm}_fut_{scenario}_mean')
        vals = da_zsmax_fut.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
        gdf[f'{name_prefix}_fut_zsmax'] = np.round(vals.transpose(), decimals=3)
        gdf[f'{name_prefix}_fut_hmax'] = gdf[f'{name_prefix}_fut_zsmax'] - gdf['elev_sbg5m']

        print(f'Done extracting water levels for {scenario}')

    print(f'Done extracting data for {storm}')

end_time = time.time()
print(f"Script ran in {end_time - start_time:.2f} seconds")

outfile = r'SFINCS_buildings_FlorFloyMatt.csv'
gdf.to_csv(outfile)






