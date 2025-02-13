import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\synthetic_tc_cmpdfld')
from src.utils import TCR_precip_stats2netcdf, TC_windspd_stats2netcdf
import hydromt


# Change directory
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585')

# Read in the ADCIRC modeled peaks and get the TC ids
gage_peaks = pd.read_csv('.\stormTide\gage_peaks_ZerosRemoved_canesm_ssp585.csv', index_col=0)
selected_tcs = gage_peaks.index.tolist()

''' Processes TCR rainfall by basin '''
# Calculate the storm total precipitation and max rain rate across the entire grid
ds = TCR_precip_stats2netcdf(tc_ids=selected_tcs,rr_threshold=5,
                             inputdir=Path(r'.\rain\03_TCR_RainOutput_Gridded_hourly'),
                             outputdir=Path(r'.\rain'))


ds2 = TC_windspd_stats2netcdf(tc_ids=selected_tcs,
                             inputdir=Path(r'.\wind\CLE15_ReGridded_canesm_ssp585cal'),
                             outputdir = Path(r'.\wind')
                             )

# Load in data catalog
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)

# Get the rainfall grid, add spatial reference
grid = ds.sel(tc_id=selected_tcs[0])['total_precip'].rio.write_crs(4326)
# Use the grid to create a mask for each basin (assigned the index value)
mask = grid.raster.rasterize(mod_domain, 'index', nodata=-9999.0, all_touched=False )
mask.raster.to_raster(r'.\rain\basin_mask_TCR.tif', nodata=-9999.0)

# Get the wind grid, add spatial reference
grid = ds2.sel(tc_id=selected_tcs[0]).rio.write_crs(4326)
# Use the grid to create a mask for each basin (assigned the index value)
mask = grid.raster.rasterize(mod_domain, 'index', nodata=-9999.0, all_touched=False )
mask.raster.to_raster(r'.\wind\basin_mask_wind.tif', nodata=-9999.0)