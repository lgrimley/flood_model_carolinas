#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from hydromt_sfincs import SfincsModel
import time
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
buildings_gdf = pd.read_csv('SFINCS_buildings.csv', index_col=0)
xcoords = buildings_gdf['xcoords'].values
ycoords = buildings_gdf['ycoords'].values

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
for storm in ['floy', 'matt', 'flor']:
    # FLOOD PROCESS CLASSIFICATION
    # Extract the water level classification for the present scenario (hindcast)
    data = da_zsmax_class.sel(run=f'{storm}_pres')
    vals = data.sel(x=xcoords, y=ycoords, method='nearest').values
    buildings_gdf[f'{storm}_pres_class'] = vals.transpose()

    # Future
    data = da_zsmax_ensmean_class.sel(run=f'{storm}_fut_ensmean')
    vals = data.sel(x=xcoords, y=ycoords, method='nearest').values
    buildings_gdf[f'{storm}_fut_class'] = vals.transpose()

    for scenario in ['compound', 'runoff', 'coastal']:
        name_prefix = f'{storm}_{scenario}'
        # PRESENT PEAK WATER LEVEL
        da_zsmax_pres = da_zsmax.sel(run=f'{storm}_pres_{scenario}')
        vals = da_zsmax_pres['zsmax'].sel(x=xcoords, y=ycoords, method='nearest').values
        buildings_gdf[f'{name_prefix}_pres_zsmax'] = vals.transpose()

        # FUTURE ENSEMBLE MEAN PEAK WATER LEVEL
        da_zsmax_fut = da_zsmax_ensmean.sel(run=f'{storm}_fut_{scenario}_mean')
        vals = da_zsmax_fut['zsmax'].sel(x=xcoords, y=ycoords, method='nearest').values
        buildings_gdf[f'{name_prefix}_fut_zsmax'] = vals.transpose()

        print(f'Done extracting water levels for {scenario}')
    print(f'Done extracting data for {storm}')

end_time = time.time()
print(f"Script ran in {end_time - start_time:.2f} seconds")

# Get the SFINCS simulation run IDs and filter
# run_ids = da_zsmax.run.values
# storm = 'flor'
# scenario = 'compound'
# subset_list = [r for r in run_ids if storm in r]
# subset_list = [r for r in subset_list if scenario in r]
# subset_list = [r for r in subset_list if 'SF8' not in r] # Remove SF8 because this was the SF based on the ensemble mean
# print(subset_list)

# PEAK FLOOD DEPTH PRESENT, FUTURE, AND DIFFERENCE
gdf[f'{name_prefix}_pdepth'] = gdf[f'{name_prefix}_pzsmax'] - gdf['gnd_elev']
gdf[f'{name_prefix}_hdepth'] = gdf[f'{name_prefix}_hzsmax'] - gdf['gnd_elev']
gdf[f'{name_prefix}_depth_diff'] = gdf[f'{name_prefix}_pzsmax'] - gdf[f'{name_prefix}_hzsmax']
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






