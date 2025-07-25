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
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\sfincs_base_mod'
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
da_zsmax_hist = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\aep\ncep_MaxWL_returnPeriods_compound.nc', crs=32617)
da_class_hist = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\aep\ncep_RP_attribution.nc', crs=32617)
da_zsmax_fut = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\canesm_ssp585\aep\projected_MaxWL_returnPeriods_compound.nc', crs=32617)
da_class_fut = cat.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\canesm_ssp585\aep\gcm_RP_attribution.nc', crs=32617)

''' Extract model output at buildings '''
for rp in [100]:
    # Present (Hindcast)
    data = da_class_hist.sel(return_period=rp)['zsmax_attr']
    vals = data.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'{rp}yr_hist_class'] = np.round(vals.transpose(), decimals=3)

    # Future
    data = da_class_fut.sel(return_period=rp)['zsmax_attr']
    vals = data.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'{rp}yr_fut_class'] = np.round(vals.transpose(), decimals=3)

    # PRESENT PEAK WATER LEVEL
    data = da_zsmax_hist.sel(return_period=rp)
    vals = data.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'{rp}yr_hist_zsmax'] = np.round(vals.transpose(), decimals=3)
    gdf[f'{rp}yr_hist_hmax'] = gdf[f'{rp}yr_hist_zsmax'] - gdf['elev_sbg5m']

    # FUTURE ENSEMBLE MEAN PEAK WATER LEVEL
    da_zsmax_fut = da_zsmax_fut.sel(return_period=rp)
    vals = da_zsmax_fut.sel(x=gdf['geometry'].x.to_xarray(), y=gdf['geometry'].y.to_xarray(), method='nearest').values
    gdf[f'{rp}yr_fut_zsmax'] = np.round(vals.transpose(), decimals=3)
    gdf[f'{rp}yr_fut_hmax'] = gdf[f'{rp}yr_fut_zsmax'] - gdf['elev_sbg5m']

    print(f'Done extracting data for {rp}yr')

end_time = time.time()
print(f"Script ran in {end_time - start_time:.2f} seconds")

outfile = r'SFINCS_buildings_100yr.csv'
gdf.to_csv(outfile)






