import pandas as pd
import xarray as xr
import numpy as np
import os
import hydromt
from hydromt_sfincs import SfincsModel
import dask.array as da
from dask.diagnostics import ProgressBar

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep')

# Lazily load the zsmax data
file_paths = [file for file in os.listdir() if ('zsmax' in file) & (file.endswith('.nc'))]
ds_list = [xr.open_dataarray(file, chunks={'x': 300, 'y': 300, 'tc_id': 25}).to_dataset(dim='scenario') for file in file_paths]
ds = xr.concat(ds_list, dim='tc_id')
print(ds)

# Load the data into memory
data = ds['compound'].chunk(dict(tc_id=-1))
data = data[:,400:500,400:500].compute()


def calculate_probability(sorted_values):
    """
    Calculate the probability for each sorted water level.
    Probability = Rank / (N + 1), where N is the number of non-NaN values for each (x, y).
    """
    # Get the valid values (non-NaN values)
    valid_values = sorted_values[~np.isnan(sorted_values)]
    N = len(valid_values)  # The total number of valid values

    if N == 0:
        return np.nan  # Return NaN if no valid values are available

    # Rank of the values (1-based index)
    rank = np.arange(1, N + 1)

    # Calculate probabilities using the formula
    probabilities = rank / (N + 1)

    return probabilities



# Check for finite values (not NaN or infinite)
finite_mask = np.isfinite(data)
count_finite_values = finite_mask.sum(dim='tc_id')


# Step 1: Sort the data for each (x, y) location along the tc_id dimension
sorted_da = xr.apply_ufunc(np.sort,
                           data,
                           input_core_dims=[['tc_id']],
                           output_core_dims=[['tc_id']],  # Output should have the same core dim
                           vectorize=False,  # Apply to chunks, not element-wise
                           dataset_fill_value=np.nan,
                           dask='parallelized',
                           output_dtypes=np.float32,
                           keep_attrs=True
                           )

sorted_da_no_nan = sorted_da.where(~np.isnan(sorted_da))


def calculate_probabilities(cell_data):
    # Remove NaNs and find the unique values
    unique_values = np.unique(cell_data[~np.isnan(cell_data)])

    # If there are no valid water levels, return an empty array (or NaNs)
    if len(unique_values) == 0:
        return np.array(np.nan)

    # Sort the unique values
    sorted_unique_values = np.sort(unique_values)

    # Calculate ranks (0-based)
    ranks = np.argsort(sorted_unique_values)

    # Calculate probabilities as rank/(number of unique values)
    probabilities = (ranks + 1) / len(sorted_unique_values)

    return probabilities




# yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
# base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
# sfincs_mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
# dep = sfincs_mod.data_catalog.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_base_mod\gis\dep.tif')
