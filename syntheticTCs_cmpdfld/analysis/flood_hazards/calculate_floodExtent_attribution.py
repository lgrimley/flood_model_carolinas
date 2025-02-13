import os
import xarray as xr
import pandas as pd
from src.utils import calculate_flooded_area_by_process
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import hydromt


'''

This code calculates the flooded area by each process for the entire model domain and each HUC6 watershed

'''

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\canesm_ssp585')

# Lazily load the zsmax data
file_paths = [file for file in os.listdir() if ('attribution' in file) & (file.endswith('.nc'))]
ds_list = [xr.open_dataset(file, chunks={'x': 300, 'y': 300, 'tc_id': 25})for file in file_paths]
da = xr.concat(ds_list, dim='tc_id')['zsmax_attr']

# Mask out the cells that are considered water bodies
water_mask = xr.open_dataarray(r'../waterbody_mask.nc')
water_mask = (water_mask == 0.0)
da = da.where(water_mask)

# Read in the data catalog to get the model and basin geom
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)
# Get the grid, add spatial reference
# Use the grid to create a mask for each basin (assigned the index value)
# grid = da.sel(tc_id=da['tc_id'][0]).rio.write_crs(32617)
# mask = grid.raster.rasterize(mod_domain, 'index', nodata=-9999.0, all_touched=False )
# mask.raster.to_raster(r'../basin_mask.tif', nodata=-9999.0)
# mask.to_netcdf(r'../basin_mask.nc')
basin_mask = xr.open_dataarray(r'../basin_mask.nc')

# Calculate the total area flooded by each process and save to a pandas dataframe
fld_area_df = pd.DataFrame()
# Loop through the zsmax classified outputs
tc_ids = da['tc_id'].values
counter = 0
# Loop through the TC outputs stored
for tc_index in tc_ids:
    data = da.sel(tc_id=tc_index)
    # Calculate the flooded area by each process across the entire domain
    fld_area = calculate_flooded_area_by_process(da=data, tc_index=tc_index)
    fld_area['AOI'] = 'Domain'
    fld_area_df = pd.concat(objs=[fld_area_df, fld_area], axis=0, ignore_index=False)
    # Loop through the HUC6 basins and calculate the flooded area by each process
    for i in range(len(mod_domain.index)):
        basin_data = data.where(basin_mask == i)
        fld_area = calculate_flooded_area_by_process(da = basin_data, tc_index=tc_index)
        fld_area['AOI'] = mod_domain['Name'][i]
        fld_area_df = pd.concat(objs=[fld_area_df, fld_area], axis=0, ignore_index=False)

    counter += 1
    print(f'Flood area calculated for {tc_index} ({counter} out of {len(tc_ids)})')


# Output the results to a csv
fld_area_df.round(3).to_csv('overland_flooded_area_table.csv')


