import xarray as xr
import numpy as np
import os
import time
import pandas as pd
import hydromt



def sort_data(data: xr.DataArray) -> [xr.DataArray, xr.DataArray]:
    # Sort data using Dask's map_blocks to sort each chunk
    def sort_block(block):
        sorted_block = np.sort(block, axis=0) # ascending order from smallest to largest value
        sorted_block = xr.DataArray(sorted_block, coords=block.coords)
        # smallest values are at the top (lowest index)
        return sorted_block


    print('Sorting data...')

    # Fill NaNs with a placeholder so they don't interfere with sorting
    data_filled = data.fillna(-9999.0)

    # Apply block-wise sort across 'tc_id'
    sorted_data = data_filled.chunk({'tc_id': -1, 'y': 300, 'x': 300}).map_blocks(sort_block, template=data_filled)

    # Replace placeholders back with NaN
    sorted_data = sorted_data.where(sorted_data != -9999.0, np.nan)

    # Assign descending rank: largest value = rank 1
    n_storms = len(sorted_data)
    rank = np.arange(1, n_storms + 1)[::-1]

    # Assign new 'rank' coordinate in place of 'tc_id'
    sorted_data['tc_id'] = rank
    sorted_data = sorted_data.rename({'tc_id':'rank'})

    # Preserve spatial_ref if present
    sorted_data['spatial_ref'] = data['spatial_ref']
    sorted_data.compute()

    return sorted_data


'''

HISTORIC SET

'''
# Lazily load the zsmax data for all storms
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep')
file_paths = [file for file in os.listdir() if ('zsmax' in file) & (file.endswith('.nc'))]
ds_list = [xr.open_dataarray(file, chunks={'x': 300, 'y': 300, 'tc_id': 25},
                             engine='netcdf4').to_dataset(dim='scenario') for file in file_paths]
ds = xr.concat(ds_list, dim='tc_id')
scenarios =['compound', 'runoff', 'coastal']

# Loop over SFINCS scenarios
for scenario in scenarios:
    sorted_wl_file = rf'.\aep\ncep_MaxWL_probabilities_{scenario}.nc'

    if os.path.exists(sorted_wl_file) is False:
        # Select the data for the specific scenario
        data = ds[scenario].chunk(dict(tc_id=-1))

        # Sort and assign ECDF probabilities
        sorted_data = sort_data(data=data)

        # Non-exceedance probability: P(X <= x)
        # Descending rank = largest gets rank 1
        probability = sorted_data['rank'] / (len(sorted_data) + 1)
        sorted_data = sorted_data.assign_coords(probability=('rank', probability))

        # Write to netcdf
        print(f'Creating {sorted_wl_file}...')
        sorted_data.to_netcdf(sorted_wl_file)
        print('Done!')
    else:
        # Load existing file
        sorted_data = xr.open_dataarray(sorted_wl_file)



'''

FUTURE DATA W/ BIAS CORRECTION

'''
# Load in data catalog
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod_domain = cat.get_geodataframe(data_like='enc_domain_HUC6_clipped').to_crs(4326)

gcm = 'canesm'
os.chdir(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\{gcm}_ssp585')
basin_mask = xr.open_dataarray(r'../basin_mask.nc')
file_paths = [file for file in os.listdir() if ('zsmax' in file) & (file.endswith('.nc'))]
ds_list = [xr.open_dataarray(file, chunks={'x': 300, 'y': 300, 'tc_id': 25},
                             engine='netcdf4').to_dataset(dim='scenario') for file in file_paths]
ds = xr.concat(ds_list, dim='tc_id')

# Load the storm weights and populate and xarray to match the data
storm_weights = pd.read_csv(
    fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\BiasCorrection\{gcm}_ssp585_weighted.csv',
    index_col=0, header=None)
storm_weights.columns = ['vmax', 'weight']
storm_weights_dict = dict(zip(storm_weights.index, storm_weights["weight"]))

start_time = time.time()
for scenario in ['compound','coastal','runoff']:
    # Lazily load the data
    data = ds[scenario].chunk(dict(tc_id=100, x=300, y=300))
    print('Loading data into memory...')
    data = data.compute()

    data_filled = data.fillna(-9999.0)
    data_weights = data_filled.copy()
    for tc_id in data_weights.tc_id.values:
        data_weights.loc[tc_id, :, :] = storm_weights_dict.get(tc_id, 0)

    n_storms = len(data)
    rank = np.arange(1, n_storms + 1)[::-1]

    # Perform sorting and reordering along the first axis (tc_id axis)
    print('Sorting data...')
    d = data_filled.values
    index_array = np.argsort(d , axis=0)  # Sort along first axis (tc_id)

    # sorted water level data
    sorted_data = np.take_along_axis(d, index_array, axis=0)
    sorted_data = xr.DataArray(sorted_data, coords=data_filled.coords)
    sorted_data = sorted_data.where(cond=(sorted_data != -9999.0), other= np.nan)
    sorted_data['tc_id'] = rank
    sorted_data = sorted_data.rename({'tc_id': 'rank'})
    sorted_data['spatial_ref'] = data['spatial_ref']

    # sorted weights
    d = data_weights.values
    sorted_weights = np.take_along_axis(d, index_array, axis=0)
    sorted_weights = xr.DataArray(sorted_weights, coords=data_filled.coords)
    sorted_weights['tc_id'] = rank
    sorted_weights = sorted_weights.rename({'tc_id': 'rank'})
    sorted_weights['spatial_ref'] = data['spatial_ref']

    # Write to netcdf
    print('Writing data to netcdfs....')
    sorted_data.to_netcdf(f'sorted_data_{scenario}.nc')
    sorted_weights.to_netcdf(f'sorted_weights_{scenario}.nc')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Sorting time: {np.round(elapsed_time) / 60} minutes")
