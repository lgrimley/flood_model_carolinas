
import os
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from hydromt_sfincs import SfincsModel
matplotlib.use('TkAgg')
plt.ion()


'''

Return Period Flood Maps for Historical

'''
def get_return_period_waterlevels(sorted_data, outputfile, basin_wide_arrival_rate=3.38, basin_n_storms = 5018):
    n_storms = len(sorted_data)
    lambda_param = basin_wide_arrival_rate * (n_storms / basin_n_storms)
    return_periods = [1, 2, 5, 10, 25, 30, 50, 100, 200, 250, 500, 1000]
    return_period_water_levels = []

    for T in return_periods:
        exceedance_probability = 1 - np.exp(-lambda_param / T)
        closest_idx = np.abs(sorted_data['probability'] - exceedance_probability).argmin(dim='probability')
        rp_data = sorted_data.isel(probability=closest_idx)  # Select the water level at the closest probability
        return_period_water_levels.append(rp_data)

    return_period_da = xr.concat(objs=return_period_water_levels, dim='return_period')
    return_period_da['return_period'] = xr.IndexVariable(dims='return_period', data=return_periods)
    return_period_da['spatial_ref'] = sorted_data['spatial_ref']
    print(f'Creating {outputfile}')
    return_period_da.to_netcdf(outputfile)

    return return_period_da


os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\aep')
for scenario in ['compound', 'coastal', 'runoff']:
    sorted_data = xr.open_dataarray(fr'.\sorted_peakWSE\ncep_sorted_peakWSE_{scenario}.nc')
    # Get return period water levels
    rp_zs = get_return_period_waterlevels(sorted_data,
                                          outputfile=rf'.\probabilistic_WSE\ncep_AEP_WSE_{scenario}.nc',
                                          basin_wide_arrival_rate=3.38, basin_n_storms=5018)


'''

Return Period Flood Maps for Future

'''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\canesm_ssp585\aep')
basin_wide_arrival_rate = 3.38
basin_n_storms = 6200

for scenario in ['compound', 'coastal', 'runoff']:
    sorted_data = xr.open_dataarray(fr'.\sorted_data\canesm_sorted_peakWSE_{scenario}.nc')
    sorted_weights = xr.open_dataarray(fr'.\sorted_data\canesm_sorted_weights_{scenario}.nc')
    cumulative_weights = sorted_weights.cumsum(dim='rank', skipna=False) # Calculate the cumulative sum of the sorted weights to form the CDF
    print('Finished calculating cumulative probability...')
    n_storms = len(sorted_data)
    lambda_param = basin_wide_arrival_rate * (n_storms / basin_n_storms)

    return_periods = [1, 2, 5, 10, 25, 30, 50, 100, 200, 250, 500, 1000]
    return_period_water_levels = []
    for T in return_periods:
        exceedance_probability = np.round(np.exp(-lambda_param / T), decimals=5)
        probability = np.round(1 - exceedance_probability, decimals=5)

        # Calculate the difference between the cdf and the chosen probability
        diff = np.abs(cumulative_weights - exceedance_probability)

        # Find the closest rank
        closest_rank = diff.argmin(dim='rank').compute()

        # Select these water levels
        return_period_data = sorted_data.isel(rank=closest_rank)
        return_period_water_levels.append(return_period_data)
        print(f'Finished processing return period {T} for {scenario} scenario')

    return_period_da = xr.concat(objs=return_period_water_levels, dim='return_period')
    return_period_da['return_period'] = xr.IndexVariable(dims='return_period', data=return_periods)
    return_period_da['spatial_ref'] = sorted_data['spatial_ref']
    outputfile = fr'\probabilistic_WSE\canesm_AEP_WSE_{scenario}.nc'
    print(f'Creating {outputfile}')
    return_period_da.to_netcdf(outputfile)






