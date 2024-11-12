import os
import xarray as xr
import pandas as pd
import geopandas as gpd


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

match = stormTide_locs.sjoin_nearest(reanalysis_locs, how='left', max_distance=5000)

# Peak water level for each storm at each gage
storm_peaks = pd.read_csv(r'.\NCEP_Reanalysis\stormTide\gage_peaks_ZerosRemoved.csv', index_col=0)

stormTide_dir = r'.\NCEP_Reanalysis\stormTide\adcirc_waterlevel_netcdf'
for tc_id in storm_peaks.index.values:
    gage_id_stormTide = storm_peaks.columns[~storm_peaks[storm_peaks.index == tc_id].isna().any()].astype(int).tolist()
    gage_id_noStormTide = storm_peaks.columns[storm_peaks[storm_peaks.index == tc_id].isna().any()].astype(int).tolist()
    file = f'{str(tc_id).zfill(4)}.nc'
    st_da = xr.open_dataset(os.path.join(stormTide_dir, file), engine='netcdf4')

    st_df = st_da.sel(index=gage_id_stormTide).to_dataframe()

    newIndex = match[match['gage_id'].isin(gage_id_noStormTide)]['Point']




