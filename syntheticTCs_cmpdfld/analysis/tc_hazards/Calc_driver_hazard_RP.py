import pandas as pd
import sys
import os
import numpy as np


def combine_driver_data(basin: str, fld_extent_df: pd.DataFrame, wind_df: pd.DataFrame,
                        rain_df: pd.DataFrame, stormtide_df: pd.DataFrame, stormtide_loc: str=None) -> pd.DataFrame():
    # Deal with the meteo data
    meteo_df = pd.concat(objs=[wind_df, rain_df], ignore_index=False, axis=1)
    df_list = [meteo_df[meteo_df.index == i].T  for i in meteo_df.index]
    meteo_df = pd.concat(objs=df_list,axis=1)
    basins = [x.split('_')[0] for x in meteo_df.index]
    vars = [''.join(x.split('_')[1:]) for x in meteo_df.index]
    tuples = [(basins[i], vars[i]) for i in range(len(vars))]
    meteo_df.index = pd.MultiIndex.from_tuples(tuples, names=['Basin', 'Variable'])
    basin_meteo = meteo_df.xs(basin, level='Basin').T
    basin_meteo.index = basin_meteo.index.astype(int)

    # Query storm tide data for location
    if stormtide_loc is not None:
        basin_st = pd.DataFrame(stormtide_df[stormtide_loc].dropna())
    else:
        basin_st = pd.DataFrame(stormtide_df.dropna())
    basin_st.columns = ['stormtide']
    basin_st.index = basin_st.index.astype(int)

    # Query flood extent data for basin
    fld_extent_df['AOI'] = [s.replace(" ", "") for s in fld_extent_df['AOI']]
    fld_extent_df = fld_extent_df[['Coastal', 'Compound', 'Runoff', 'Total_Flooded', 'AOI']]
    fld_extent_data = pd.DataFrame(fld_extent_df[fld_extent_df['AOI'].str.contains(basin)]['Compound'])
    fld_extent_data.columns = ['FldArea']

    # Combine the meteo, stormtide, and flood area data into a single dataframe
    basin_data = pd.concat(objs=[basin_meteo, basin_st, fld_extent_data], axis=1, ignore_index=False).fillna(0)

    return basin_data


os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

''' 
HISTORIC PERIOD DATA 
'''

# Load meteo information
wind_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\rain\basin_tc_precipitation_stats.csv', index_col=0)
stormtide_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\stormTide\gage_peaks_ZerosRemoved_ncep.csv', index_col=0)
fld_extent_df = pd.read_csv(r'.\04_RESULTS\ncep\overland_flooded_area_table.csv', index_col=0)
outdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_jointprob\basin_data'

stormtide_pt =[None] #['188','194','198','204','206']
stormtide_basin = ['Domain']#['LowerPeeDee', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico']
n_storms = 1309
n_basin_storms = 5018
lambda_param = 3.38 * (n_storms / n_basin_storms)

for i in range(len(stormtide_basin)):
    basin=stormtide_basin[i]
    if basin == 'Domain':
        stormtide_max = stormtide_df.max(axis=1)
        basin_data = combine_driver_data(basin=basin, fld_extent_df=fld_extent_df, wind_df=wind_df,
                                         rain_df=rain_df, stormtide_df=stormtide_max, stormtide_loc=None)
    else:
        basin_data = combine_driver_data(basin=basin, fld_extent_df=fld_extent_df, wind_df=wind_df,
                                         rain_df=rain_df, stormtide_df=stormtide_df, stormtide_loc=stormtide_pt[i])
    vars = basin_data.columns

    for v in vars:
        # Sort the dataframe by flood area and calculate return period
        sorted = basin_data.sort_values(by=v, axis=0, ascending=True)
        ecdf = np.arange(1, n_storms+1) / n_storms
        eprob = 1 - ecdf
        sorted[f'{v}_rp'] = (1 / (1 - np.exp(-lambda_param*eprob)))
        basin_data = sorted

    basin_data.to_csv(os.path.join(outdir,f'{basin}_data_rp_historic.csv'))

''' 
FUTURE PERIOD DATA 
'''
# Load meteo information
wind_df = pd.read_csv(r'.\02_DATA\CMIP6_585\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df = pd.read_csv(r'.\02_DATA\CMIP6_585\rain\basin_tc_precipitation_stats.csv', index_col=0)
stormtide_df = pd.read_csv(r'.\02_DATA\CMIP6_585\stormTide\gage_peaks_ZerosRemoved_canesm_ssp585.csv', index_col=0)
fld_extent_df = pd.read_csv(r'.\04_RESULTS\canesm_ssp585\overland_flooded_area_table.csv', index_col=0)

storm_weights = pd.read_csv(r'.\02_DATA\BiasCorrection\canesm_ssp585_weighted.csv', index_col=0, header=None)
storm_weights.columns = ['vmax','weight']

outdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_jointprob\basin_data'
stormtide_pt = [None] #['188','194','198','204','206']
stormtide_basin = ['Domain']#['LowerPeeDee', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico']
n_storms = 1543
n_basin_storms = 6200
lambda_param = 3.38 * (n_storms / n_basin_storms)

for i in range(len(stormtide_basin)):
    basin=stormtide_basin[i]
    if basin == 'Domain':
        stormtide_max = stormtide_df.max(axis=1)
        basin_data = combine_driver_data(basin=basin, fld_extent_df=fld_extent_df, wind_df=wind_df,
                                         rain_df=rain_df, stormtide_df=stormtide_max, stormtide_loc=None)
    else:
        basin_data = combine_driver_data(basin=basin, fld_extent_df=fld_extent_df, wind_df=wind_df,
                                         rain_df=rain_df, stormtide_df=stormtide_df, stormtide_loc=stormtide_pt[i])
    vars = basin_data.columns
    basin_data = pd.concat(objs=[basin_data, storm_weights], axis=1, ignore_index=False)

    for v in vars:
        # Sort the dataframe by flood area and calculate return period
        sorted = basin_data.sort_values(by=v, axis=0, ascending=True)
        ecdf = sorted['weight'].cumsum()
        eprob = 1 - ecdf
        sorted[f'{v}_rp'] = (1 / (1 - np.exp(-lambda_param*eprob)))
        basin_data = sorted

    basin_data.to_csv(os.path.join(outdir,f'{basin}_data_rp_future.csv'))


