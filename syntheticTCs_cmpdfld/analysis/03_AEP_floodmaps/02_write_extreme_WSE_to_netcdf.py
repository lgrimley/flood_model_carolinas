#!/usr/bin/env python
# coding: utf-8
from hydromt_sfincs import SfincsModel
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

# Filepath to data catalog yml
yml_sfincs_Carolinas = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\sfincs_base_mod'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_sfincs_Carolinas])
cat = mod.data_catalog

'''
Historic TC water levels 
'''

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\aep\sorted_peakWSE')
basin_wide_arrival_rate = 3.38
basin_n_storms = 5018
scenario = 'compound'
chunks_size = {'probability':20, 'x': 500, 'y': 500}

#  The probability dimension in the sorted_data is the ECDF
#  (e.g., probability that a value is less than or equal to a given value)
# Select that peak water levels that have a lower probability than the 1% AEP
sorted_data = cat.get_rasterdataset(f'ncep_sorted_peakWSE_compound.nc',chunks=chunks_size)

# Round probs to the same decimals to avoid floating point issues
sorted_data['probability'] = np.round(sorted_data['probability'].values, decimals=5)

# Get the probability for the 1% annual event
n_storms = len(sorted_data)
lambda_param = basin_wide_arrival_rate * (n_storms / basin_n_storms)
T = 100
probability_precise = np.round(1 - np.exp(-lambda_param / T), decimals=5)
probability = np.round(1 - np.exp(-lambda_param / T), decimals=2)

# Create mask where exceedance probability is less than threshold (more extreme events)
mask_precise = sorted_data['probability'] <= probability_precise
mask = sorted_data['probability'] <= probability

# Number of data points that exceed the 1%
print(np.sum(mask_precise))
print(np.sum(mask))

# Select all data along 'probability' dimension that meet this condition
extreme_data = sorted_data.sel(probability=mask).compute()
extreme_data.to_netcdf('ncep_peakWSE_compound_extreme.nc')


'''
Future TC water levels
'''

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\canesm_ssp585\aep\sorted_data')
basin_wide_arrival_rate = 3.38
basin_n_storms = 6200
chunks_size = {'rank':20, 'x': 500, 'y': 500}
scenario='compound'

# Select the last 200 ranks (assuming 'rank' is the first dimension)
sorted_data = cat.get_rasterdataset(f'canesm_sorted_peakWSE_{scenario}.nc',chunks=chunks_size)

# Get the total number of ranks
n_ranks_total = sorted_data.sizes['rank']

# Create a slice for the last 50 ranks
extreme_slice = slice(n_ranks_total - 20, n_ranks_total)
sorted_data_extreme = sorted_data.isel(rank=extreme_slice).compute()

# Load the corresponding weights for the sorted water level data
sorted_weights = cat.get_rasterdataset(f'canesm_sorted_weights_{scenario}.nc',chunks=chunks_size)
sorted_weights_extreme = sorted_weights.isel(rank=extreme_slice).compute()

# Calculate the cumulative sum of the sorted weights to get the bias corrected CDF
cdf = sorted_weights.cumsum(dim='rank', skipna=False)
cdf = cdf.isel(rank=extreme_slice).compute()
cdf = np.round(cdf, decimals=5)

# Estimate threshold probability for 1% AEP (using Poisson formulation)
n_storms = n_ranks_total
lambda_param = basin_wide_arrival_rate * (n_storms / basin_n_storms)
T = 100  # return period
exceedance_prob = np.exp(-lambda_param / T)
cdf_threshold = np.round(exceedance_prob, decimals=5)

# Create a mask for events more extreme than the 1% AEP
# These are values where CDF <= threshold (i.e., exceedance probability ≥ 1%)
mask = cdf <= cdf_threshold

# Select all events where CDF is below threshold
# We'll do a boolean mask along the 'rank' dimension
extreme_data = sorted_data_extreme.where(mask).dropna(dim='rank', how='all').compute()

# Save the extreme events to NetCDF
extreme_data.to_netcdf(f'canesm_peakWSE_compound_extreme.nc')
