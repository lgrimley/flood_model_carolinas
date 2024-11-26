import os
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')
# Reanalysis data from ADCIRC EDS v2
reanalysis = xr.open_dataset(r'.\EDSReanalysis_data\EDSReanalysis_V2_1992_2022.nc', engine='netcdf4')

# Mapping the stormTide ADCIRC to the reanalysis extraction locations
reanalysis_locs = pd.read_csv(r'.\EDSReanalysis_data\full_data_meta.csv')
stormTide_locs = pd.read_csv(r'.\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')
stormTide_locs['gage_id'] = stormTide_locs['gage_id'].astype(int)

reanalysis_locs = gpd.GeoDataFrame(reanalysis_locs,
                                   geometry=gpd.points_from_xy(x=reanalysis_locs['LON'],y=reanalysis_locs['LAT'],
                                                               crs=4326)).to_crs(32617)
stormTide_locs = gpd.GeoDataFrame(stormTide_locs,
                                   geometry=gpd.points_from_xy(x=stormTide_locs['x'],y=stormTide_locs['y'],
                                                               crs=4326)).to_crs(32617)

match = stormTide_locs.sjoin_nearest(reanalysis_locs, how='left', max_distance=2000)
match['gage_id'] = match['gage_id'].astype(int)

# Peak water level for each storm at each gage
storm_peaks = pd.read_csv(r'.\NCEP_Reanalysis\stormTide\gage_peaks_ZerosRemoved.csv', index_col=0)

stormTide_dir = r'.\NCEP_Reanalysis\stormTide\adcirc_waterlevel_netcdf'

outdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\stormTide\ADCIRC_storm_vs_reanalysis'
for tc_id in [1234, 2645, 2773, 3429]:
    gage_id_stormTide = storm_peaks.columns[~storm_peaks[storm_peaks.index == tc_id].isna().any()].astype(int).tolist()
    gage_id_noStormTide = storm_peaks.columns[storm_peaks[storm_peaks.index == tc_id].isna().any()].astype(int).tolist()

    # Open the netcdf of water level time series for the TC
    file = f'{str(tc_id).zfill(4)}.nc'
    st_da = xr.open_dataset(os.path.join(stormTide_dir, file), engine='netcdf4')

    # convert to a dataframe
    st_df = st_da.to_dataframe()

    for ind in st_da.index.values:
        # Get the ADCIRC TC storm tide output
        df = st_df.xs(key=ind, level='index')
        mask = [df['waterlevel']  == 0.0000]
        dfwl = df.copy()
        dfwl[mask] = np.nan
        df_stormTide = dfwl.dropna(axis=0, how='any')

        # Get the reanalysis
        station_re = match[match['gage_id'] == ind]['Point'].item()
        tstart = df_stormTide.index.min()
        tend = df_stormTide.index.max()
        st_re = reanalysis.sel(index=station_re).sel(time=slice(tstart, tend))
        st_re_df = st_re.to_dataframe()

        # Plot water level
        compare = pd.concat(objs=[df_stormTide, st_re_df], axis=1, ignore_index=False)
        dfp = compare['waterlevel']
        dfp.columns = ['stormTide', 'reanalysis']

        fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
        dfp.plot(ax=ax, legend=True)
        ax.set_xlabel('')
        ax.set_ylabel('Water Level (m+MSL)')
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'TC: {tc_id}; Gage: {ind}')
        plt.savefig(os.path.join(outdir, f'{str(tc_id).zfill(4)}_waterlevel_{ind}.png'))
        plt.close()


