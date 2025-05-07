import pandas as pd
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np

sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True


def cleanup_meteo_tables(wind_df: pd.DataFrame, rain_df: pd.DataFrame, aoi: str=None):
    # Combine the wind and rain into a single dataframe
    meteo_df = pd.concat(objs=[wind_df, rain_df], ignore_index=False, axis=1)
    # Transpose the columns and index
    df_list = [meteo_df[meteo_df.index == i].T  for i in meteo_df.index]
    meteo_df = pd.concat(objs=df_list,axis=1)
    # Get the basin associated with each row
    basins = [x.split('_')[0] for x in meteo_df.index]
    # Get the variable associated with each row
    vars = [''.join(x.split('_')[1:]) for x in meteo_df.index]
    # Create tuples of basin, variable that can be used to make a multiindex
    tuples = [(basins[i], vars[i]) for i in range(len(vars))]
    meteo_df.index = pd.MultiIndex.from_tuples(tuples, names=['Basin', 'Variable'])

    if aoi is not None:
        meteo_df = meteo_df.xs(aoi, level='Basin').T
        meteo_df.index = meteo_df.index.astype(int)

    return meteo_df


def cleanup_flood_tables1(tables_dir: str=None):

    csvfiles = [f for f in os.listdir(tables_dir) if f.endswith('.csv')]
    df = pd.concat((pd.read_csv(os.path.join(tables_dir, file), index_col=0) for file in csvfiles), ignore_index=False)
    df.index = df.index.str.replace(' ', '', regex=False)

    df['tc_id'] = [int(x.split('_')[0]) for x in df.index]
    df['basin'] = [str(x.split('_')[1]) for x in df.index]
    df['attr'] = [str(x.split('_')[-1]) for x in df.index]
    mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
    df['attr'] = df['attr'].map(lambda x: mapping.get(x, 'Total'))

    # Reset index
    df.reset_index(inplace=True, drop=True)

    # columns to aggregate
    value_cols = [col for col in df.columns if col not in ['tc_id', 'basin', 'attr']]

    # Group by the row identifiers and aggregate
    df_grouped = df.groupby(['basin', 'attr', 'tc_id'])[value_cols].mean().reset_index()

    # Set index and unstack tc_id
    df_pivoted = df_grouped.set_index(['basin', 'attr', 'tc_id']).unstack('tc_id')

    # sort index and columns
    df_pivoted = df_pivoted.sort_index(axis=0).sort_index(axis=1)

    return df_pivoted


def cleanup_flood_tables2(tables_dir: str=None):
    csvfiles = [f for f in os.listdir(tables_dir) if f.endswith('.csv')]
    df = pd.concat((pd.read_csv(os.path.join(tables_dir, file), index_col=0) for file in csvfiles), ignore_index=False)
    df.index = df.index.str.replace(' ', '', regex=False)
    df.reset_index(inplace=True)
    df['col_numeric'] = pd.to_numeric(df['index'], errors='coerce')
    df = df[(df['col_numeric'] < 4) | (pd.isna(df['col_numeric']))]
    df.reset_index(inplace=True, drop=True)
    # Find transition indices: current is NaN, previous is integer
    is_nan_now = df['col_numeric'].isna()
    was_integer_before = df['col_numeric'].shift(1).notna() & (df['col_numeric'].shift(1) % 1 == 0)
    transition_indices = df.index[is_nan_now & was_integer_before]

    # Replace the 4 rows before each transition index
    for idx in transition_indices:
        tc_id = df['index'][idx].split('_')[0]
        for i in range(idx - 4, idx):
            current_value = df.at[i, 'col_numeric']
            if current_value == 0:
                new_value = f'{tc_id}_Domain'
            else:
                new_value = f'{tc_id}_Domain_attr{int(current_value)}'
            df.at[i, 'index'] = new_value

    df.drop(columns='col_numeric', inplace=True)
    df.set_index('index', inplace=True, drop=True)

    # Now back to normal
    df['tc_id'] = [int(x.split('_')[0]) for x in df.index]
    df['basin'] = [str(x.split('_')[1]) for x in df.index]
    df['attr'] = [str(x.split('_')[-1]) for x in df.index]
    mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
    df['attr'] = df['attr'].map(lambda x: mapping.get(x, 'Total'))

    # Reset index
    df.reset_index(inplace=True, drop=True)
    #df.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\cleaned_tables.csv')
    # columns to aggregate
    value_cols = [col for col in df.columns if col not in ['tc_id', 'basin', 'attr']]

    # Group by the row identifiers and aggregate
    df_grouped = df.groupby(['basin', 'attr', 'tc_id'])[value_cols].mean().reset_index()

    # Set index and unstack tc_id
    df_pivoted = df_grouped.set_index(['basin', 'attr', 'tc_id']).unstack('tc_id')

    # sort index and columns
    df_pivoted = df_pivoted.sort_index(axis=0).sort_index(axis=1)

    return df_pivoted


def calc_exceedance_probabilities(data: pd.DataFrame, vars: list=None, n_storms: int=None,
                                  lambda_param: float=None, weights_col: str=None) -> pd.DataFrame():
    if vars is None:
        vars = data.columns
    print(f'Calculating the return period for these varibles: {vars}')

    if weights_col is None:
        for v in vars:
            # Sort the dataframe by flood area and calculate return period
            sorted = data.sort_values(by=v, axis=0, ascending=True)
            ecdf = np.arange(1, n_storms + 1) / n_storms
            eprob = 1 - ecdf
            eprob = np.clip(eprob, 1e-10, 1)  # avoid 0
            sorted[f'{v}_RP'] = (1 / (1 - np.exp(-lambda_param * eprob)))
            data = sorted
    else:
        for v in vars:
            if v != weights_col:
                print('Using weights to calculate return period...')
                # Sort the dataframe by flood area and calculate return period
                sorted = data.sort_values(by=v, axis=0, ascending=True)
                ecdf = sorted[weights_col].cumsum()
                eprob = 1 - ecdf
                sorted[f'{v}_RP'] = (1 / (1 - np.exp(-lambda_param * eprob)))
                data = sorted
        data.drop(columns=['vmax', 'weight'], inplace=True)

    sorted = sorted.round(2)

    return sorted



def get_peakStormTide_at_AOI(aoi: str=None, stormtide_csv: str=None):
    stormtide_df = pd.read_csv(stormtide_csv, index_col=0)

    stormtide_pt = [None, '188', '194', '198', '204', '206']
    stormtide_basin = ['Domain', 'LowerPeeDee', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico']
    map = pd.DataFrame([stormtide_pt, stormtide_basin]).T
    map.columns = ['Loc', 'Basin']

    stormtide_loc = map.loc[map['Basin'] == aoi, 'Loc']
    if stormtide_loc.item() is None:
        basin_st = stormtide_df.max(axis=1)
        basin_st.index = basin_st.index.astype(int)
        basin_st = basin_st.round(2)
        df = pd.DataFrame(basin_st, columns=['stormtide'])
    else:
        basin_st = stormtide_df[stormtide_loc]
        basin_st.index = basin_st.index.astype(int)
        df = basin_st.round(2)
        df.columns = ['stormtide']

    return df



os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

''' 
HISTORIC PERIOD DATA 
'''
outdir = r'.\05_ANALYSIS\return_period_tables'

n_basin_storms = 5018  # Number of TCs in the Atlantic Basin for the historic period
n_storms = 1309 # Number of TCs within 200km of study area
lambda_param = 3.38 * (n_storms / n_basin_storms)

# Load meteo information
wind_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\rain\basin_tc_precipitation_stats.csv', index_col=0)
stormtide_csv = r'.\02_DATA\NCEP_Reanalysis\stormTide\gage_peaks_ZerosRemoved_ncep.csv'
fld_df = cleanup_flood_tables1(tables_dir=r'.\04_MODEL_OUTPUTS\ncep\flood_hazard_tables')

aois = np.unique(fld_df.index.get_level_values('basin'))
for aoi in aois:
    # if aoi == 'Domain':
    #     continue
    # else:
        # Calculate the RP for the meteo variables
        meteo_df = cleanup_meteo_tables(wind_df=wind_df, rain_df=rain_df, aoi=aoi)
        meteo_df = calc_exceedance_probabilities(data=meteo_df, vars=None, n_storms=n_storms, lambda_param=lambda_param,
                                                 weights_col=None)

        # Calculate the RP for the storm tide
        st_basin = get_peakStormTide_at_AOI(aoi=aoi, stormtide_csv=stormtide_csv)
        st_basin = calc_exceedance_probabilities(data=st_basin, vars=None, n_storms=n_storms, lambda_param=lambda_param,
                                                 weights_col=None)

        # Calculate the RP for the flood hazard - depth and extents

        fld_basin = fld_df.xs(aoi, level='basin').T
        fld_basin.index.rename(['stat', 'tc_id'], inplace=True)

        print(np.unique(fld_basin.index.get_level_values('stat')))
        select_stats = ['mean', '50%', '90%', 'Area_sqkm']
        fld_basin_rp = pd.DataFrame()
        for stat in select_stats:
            data = fld_basin.xs(stat, level='stat')
            data.columns = [f'{x}_{stat}' for x in data.columns]
            data = calc_exceedance_probabilities(data=data, vars=None, n_storms=n_storms, lambda_param=lambda_param,
                                                 weights_col=None)
            fld_basin_rp = pd.concat(objs=[fld_basin_rp, data], axis=1, ignore_index=False)

        # Combine into a single dataframe and write out
        combined = pd.concat(objs=[meteo_df, st_basin, fld_basin_rp], axis=1, ignore_index=False)
        combined = combined.rename_axis('tc_id').round(2)
        combined['basin'] = aoi
        combined.to_csv(os.path.join(outdir,f'{aoi}_data_rp_ncep.csv'))



'''
FUTURE PERIOD DATA
'''
n_storms = 1543
n_basin_storms = 6200
lambda_param = 3.38 * (n_storms / n_basin_storms)

wind_df = pd.read_csv(r'.\02_DATA\CMIP6_585\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df = pd.read_csv(r'.\02_DATA\CMIP6_585\rain\basin_tc_precipitation_stats.csv', index_col=0)
stormtide_csv = r'.\02_DATA\CMIP6_585\stormTide\gage_peaks_ZerosRemoved_canesm_ssp585.csv'
fld_df = cleanup_flood_tables1(tables_dir=r'.\04_MODEL_OUTPUTS\canesm_ssp585\flood_hazard_tables')
storm_weights = pd.read_csv(r'.\02_DATA\BiasCorrection\canesm_ssp585_weighted.csv', index_col=0, header=None)
storm_weights.columns = ['vmax','weight']

aois = np.unique(fld_df.index.get_level_values('basin'))

for aoi in aois:
    # if aoi == 'Domain':
    #     continue
    # else:
        # Calculate the RP for the meteo variables
        meteo_df = cleanup_meteo_tables(wind_df=wind_df, rain_df=rain_df, aoi=aoi)
        meteo_df = pd.concat(objs=[meteo_df, storm_weights], axis=1, ignore_index=False)
        meteo_df = calc_exceedance_probabilities(data=meteo_df, vars=None, n_storms=n_storms, lambda_param=lambda_param,
                                                 weights_col='weight')

        # Calculate the RP for the storm tide
        st_basin = get_peakStormTide_at_AOI(aoi=aoi, stormtide_csv=stormtide_csv)
        st_basin = pd.concat(objs=[st_basin, storm_weights], axis=1, ignore_index=False)
        st_basin = calc_exceedance_probabilities(data=st_basin, vars=['stormtide'], n_storms=n_storms, lambda_param=lambda_param,
                                                 weights_col='weight')

        # Calculate the RP for the flood hazard - depth and extents
        fld_basin = fld_df.xs(aoi, level='basin').T
        fld_basin.index.rename(['stat', 'tc_id'], inplace=True)
        print(np.unique(fld_basin.index.get_level_values('stat')))
        select_stats = ['mean', '50%', '90%', 'Area_sqkm']
        fld_basin_rp = pd.DataFrame()
        for stat in select_stats:
            data = fld_basin.xs(stat, level='stat')
            cols = [f'{x}_{stat}' for x in data.columns]
            data.columns = cols
            data = pd.concat(objs=[data, storm_weights], axis=1, ignore_index=False)

            data = calc_exceedance_probabilities(data=data, vars=cols, n_storms=n_storms, lambda_param=lambda_param,
                                                 weights_col='weight')
            fld_basin_rp = pd.concat(objs=[fld_basin_rp, data], axis=1, ignore_index=False)

        # Combine into a single dataframe and write out
        combined = pd.concat(objs=[meteo_df, st_basin, fld_basin_rp], axis=1, ignore_index=False)
        df_rounded = combined.rename_axis('tc_id').round(2)
        df_rounded['basin'] = aoi
        df_rounded = pd.concat(objs=[df_rounded, storm_weights],axis=1, ignore_index=False)
        df_rounded.to_csv(os.path.join(outdir,f'{aoi}_data_rp_canesm.csv'))




