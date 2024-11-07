import sys

sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\synthetic_tc_cmpdfld')

import hydromt
from syntheticTC_utils import *

# Load in data catalog
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)

# Change directory
os.chdir(r'Z:\Data-Expansion\users\lelise\Chapter3\NCEP_Reanalysis')

# Read in the ADCIRC modeled peaks and get the TC ids
gage_peaks = pd.read_csv('.\stormTide\gage_peaks.csv', index_col=0)
selected_tcs = gage_peaks.index.tolist()

''' Processes TCR rainfall by basin '''
# Calculate the storm total precipitation and max rain rate across the entire grid
da_sum, da_max = TCR_precip_stats_to_netcdf(tc_ids=selected_tcs,
                                            inputdir=r'.\rain\03_TCR_RainOutput_Gridded_hourly',
                                            outputdir=r'.\rain')
# Get the rainfall grid, add spatial reference
grid = da_sum.sel(run=selected_tcs[0]).rio.write_crs(4326)
# Use the grid to create a mask for each basin (assigned the index value)
mask = grid.raster.rasterize(mod_domain, 'index', nodata=-9999.0, all_touched=False )
mask.raster.to_raster(r'.\rain\basin_mask_TCR.tif', nodata=-9999.0)

# Loop through the basins and calculate the basin average
maxTotP = pd.DataFrame(index=selected_tcs)
avgTotP = pd.DataFrame(index=selected_tcs)
maxRainRate = pd.DataFrame(index=selected_tcs)
for i in range(len(mod_domain.index)):
    basin = mod_domain['Name'][i]

    # Calculate basin averaged precipitation
    basin_avgTotP = da_sum['precip'].where(mask == i).mean(dim=['x', 'y']).to_dataframe()
    basin_avgTotP.columns = ['spatial_ref', f'{basin}']
    avgTotP = pd.concat(objs=[avgTotP, basin_avgTotP[f'{basin}']], axis=1, ignore_index=False)

    # Calculate max total precipitation
    basin_maxTotP = da_sum['precip'].where(mask == i).max(dim=['x', 'y']).to_dataframe()
    basin_maxTotP.columns = ['spatial_ref', f'{basin}']
    maxTotP = pd.concat(objs=[maxTotP, basin_maxTotP[f'{basin}']], axis=1, ignore_index=False)

    # Calculate the max rain rate over the basin
    basin_maxRR = da_max['precip'].where(mask == i).max(dim=['x', 'y']).to_dataframe()
    basin_maxRR.columns = ['spatial_ref', f'{basin}']
    maxRainRate = pd.concat(objs=[maxRainRate, basin_maxRR[f'{basin}']], axis=1, ignore_index=False)

    print(f'Done processing {basin}')

maxRainRate = maxRainRate.round(2)
avgTotP = avgTotP.round(1)
maxTotP = maxTotP.round(1)
maxRainRate.to_csv(r'.\rain\TC_maxRainRate_by_basin.csv')
avgTotP.to_csv(r'.\rain\basinAvgTotPrecip_by_basin.csv')
maxTotP.to_csv(r'.\rain\basinMaxTotPrecip_by_basin.csv')

''' Processes Wind by basin '''

vmax_da = get_TCs_Vmax(tc_ids=selected_tcs,
                       inputdir=r'.\wind\02_CLE15_WindOutput_Gridded',
                       outputdir = r'.\wind')
# Get the wind grid, add spatial reference
grid = vmax_da.sel(run=selected_tcs[0]).rio.write_crs(4326)
# Use the grid to create a mask for each basin (assigned the index value)
mask = grid.raster.rasterize(mod_domain, 'index', nodata=-9999.0, all_touched=False )
mask.raster.to_raster(r'.\wind\basin_mask_wind.tif', nodata=-9999.0)

# Loop through the basins and calculate the basin average
Vmax = pd.DataFrame(index=selected_tcs)
for i in range(len(mod_domain.index)):
    basin = mod_domain['Name'][i]

    # Calculate basin averaged precipitation
    basin_maxVmax = vmax_da.where(mask == i).max(dim=['x', 'y']).to_dataframe()
    basin_maxVmax.columns = ['spatial_ref', f'{basin}']
    Vmax = pd.concat(objs=[Vmax, basin_maxVmax[f'{basin}']], axis=1, ignore_index=False)

    print(f'Done processing {basin}')

Vmax = Vmax.round(2)
Vmax.to_csv(r'.\wind\TC_Vmax_by_basin.csv')


