import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\synthetic_tc_cmpdfld')

import hydromt
from src.utils import TCR_precip_stats2netcdf, TC_windspd_stats2netcdf

# Load in data catalog
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)

# Change directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis')

# Read in the ADCIRC modeled peaks and get the TC ids
gage_peaks = pd.read_csv('.\stormTide\gage_peaks_ZerosRemoved_ncep.csv', index_col=0)
selected_tcs = gage_peaks.index.tolist()

''' Processes TCR rainfall by basin '''
# Calculate the storm total precipitation and max rain rate across the entire grid
ds = TCR_precip_stats2netcdf(tc_ids=selected_tcs,rr_threshold=5,
                             inputdir=Path(r'.\rain\03_TCR_RainOutput_Gridded_hourly'),
                             outputdir=Path(r'.\rain'))

# Get the rainfall grid, add spatial reference
grid = ds.sel(tc_id=selected_tcs[0])['total_precip'].rio.write_crs(4326)
# Use the grid to create a mask for each basin (assigned the index value)
mask = grid.raster.rasterize(mod_domain, 'index', nodata=-9999.0, all_touched=False )
mask.raster.to_raster(r'.\rain\basin_mask_TCR.tif', nodata=-9999.0)

# Loop through the basins and calculate the basin average
basin_precip_stats = pd.DataFrame(index=selected_tcs)
for i in range(len(mod_domain.index)):
    basin = mod_domain['Name'][i].replace(" ","")

    # Calculate basin cumulative precip
    df1 = ds['total_precip'].where(mask == i).sum(dim=['x', 'y']).to_dataframe()
    var_name1 = f'{basin}_totP'
    df1.columns = ['spatial_ref', var_name1]

    # Calculate max rain rate
    df2 = ds['max_RR'].where(mask == i).max(dim=['x', 'y']).to_dataframe()
    var_name2 = f'{basin}_maxRR'
    df2.columns = ['spatial_ref', var_name2]

    # Calculate the mean rain rate using a min threshold
    df3 = ds['mean_RR'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    var_name3 = f'{basin}_meanRR'
    df3.columns = ['spatial_ref', var_name3]

    # Calculate the mean rain rate using a min threshold
    df4 = ds['mean_RR_thresh'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    var_name4 = f'{basin}_meanRRthresh'
    df4.columns = ['spatial_ref', var_name4]

    basin_precip_stats = pd.concat(objs=[basin_precip_stats, df1[var_name1], df2[var_name2],
                                         df3[var_name3], df4[var_name4]], axis=1, ignore_index=False)
    print(f'Done processing {basin}')

basin_precip_stats = basin_precip_stats.round(2)
basin_precip_stats.to_csv(r'.\rain\basin_tc_precipitation_stats.csv')


''' Processes Wind by basin '''

ds = TC_windspd_stats2netcdf(tc_ids=selected_tcs,
                             inputdir=Path(r'.\wind\02_CLE15_WindOutput_Gridded'),
                             outputdir = Path(r'.\wind'))

# Get the wind grid, add spatial reference
grid = ds.sel(tc_id=selected_tcs[0]).rio.write_crs(4326)
# Use the grid to create a mask for each basin (assigned the index value)
mask = grid.raster.rasterize(mod_domain, 'index', nodata=-9999.0, all_touched=False )
mask.raster.to_raster(r'.\wind\basin_mask_wind.tif', nodata=-9999.0)

# Loop through the basins and calculate the basin average
basin_windspd_stats = pd.DataFrame(index=selected_tcs)
for i in range(len(mod_domain.index)):
    basin = mod_domain['Name'][i].replace(" ","")

    # max windspeed
    df1 = ds['max_windspd'].where(mask == i).max(dim=['x', 'y']).to_dataframe()
    var_name1 = f'{basin}_maxWS'
    df1.columns = ['spatial_ref', var_name1]

    # mean - max windspeed
    df3 = ds['max_windspd'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    var_name3 = f'{basin}_meanMaxWS'
    df3.columns = ['spatial_ref', var_name3]

    # mean windspeed
    df2 = ds['mean_windspd'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    var_name2 = f'{basin}_meanWS'
    df2.columns = ['spatial_ref', var_name2]

    basin_windspd_stats = pd.concat(objs=[basin_windspd_stats, df1[var_name1], df2[var_name2], df3[var_name3]],
                                    axis=1, ignore_index=False)
    print(f'Done processing {basin}')

basin_windspd_stats = basin_windspd_stats.round(2)
basin_windspd_stats.to_csv(r'.\wind\basin_tc_windspd_stats.csv')