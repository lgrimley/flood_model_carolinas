import os
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs


# This script calculates water level stats using SFINCS map (sfincs_map.nc) output extracted at gage locations.
# Use createPlots = True/False to get the script to create figures of the model vs. observed water levels


def calculate_stats(station_id, df, tstart=None, tend=None):
    if tstart:
        df = df[df.index > tstart]
    if tend:
        df = df[df.index < tend]

    n = len(df)
    # Mean Absolute Error
    mae = sum(abs(df.Modeled - df.Observed)) / n
    # Mean Error or Bias
    bias = sum(df.Modeled - df.Observed) / n
    # Root Mean Squared Error
    rmse = sum(((df.Observed - df.Modeled) ** 2) / n) ** 0.5
    # Peak Error
    pe = df.Modeled.max() - df.Observed.max()
    # Time to Peak - Error
    tpe = df.Modeled.idxmax() - df.Observed.idxmax()
    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (sum((df.Modeled - df.Observed) ** 2) / sum((df.Observed - df.Observed.mean()) ** 2))
    # Correlation Coefficient (Pearson)
    try:
        r = sum(((df.Modeled - df.Modeled.mean()) * (df.Observed - df.Observed.mean()))) / (
                sum((df.Modeled - df.Modeled.mean()) ** 2) * sum((df.Observed - df.Observed.mean()) ** 2)) ** 0.5
    except:
        print('Correlation Coefficient problem')
        r = 0

    # Coefficient of Determination
    r2 = r ** 2

    # Save stats in a dataframe for output
    stats = pd.DataFrame(data={'station_id': station_id,
                               'mae': round(mae, 2),
                               'rmse': round(rmse, 2),
                               'nse': round(nse, 2),
                               'bias': round(bias, 2),
                               'r': round(r, 2),
                               'r2': round(r2, 2),
                               'pe': round(pe, 2),
                               'tpe': round(tpe.seconds, 1),
                               'mod_peak_wl': round(df.Modeled.max(), 2),
                               'obs_peak_wl': round(df.Observed.max(), 2)
                               },
                         index=[0]
                         )
    peak_dt = [df.Observed.idxmax(), df.Modeled.idxmax()]

    return stats, peak_dt


def clean_obs_coords(obs_df, source_crs, target_crs):
    # Clean up the observation data and the coordinates
    if 'geometry' in list(obs_df.coords):
        pts = gpd.GeoDataFrame(obs_df.index.values,
                               geometry=obs_df.geometry.values,
                               crs=source_crs)
        pts.to_crs(target_crs, inplace=True)
        pts.columns = ['site_no', 'geometry']
        obs_df.geometry.values = pts.geometry
    else:
        pts = gpd.GeoDataFrame(obs_df.index,
                               geometry=gpd.points_from_xy(x=obs_df.x.values,
                                                           y=obs_df.y.values,
                                                           crs=source_crs))
        pts.to_crs(target_crs, inplace=True)
        pts.columns = ['site_no', 'geometry']
        obs_df.x.values = pts.geometry.x
        obs_df.y.values = pts.geometry.y

    return pts, obs_df


# Load data catalog and model results
cat_dir = r'Z:\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\final_model')
model_roots = ['ENC_200m_sbg5m_maxN',
               # 'ENC_200m_sbg5m_avgN_adv1_eff25',
               # 'ENC_200m_sbg5m_avgN_adv1_eff75',
               # 'ENC_200m_sbg5m_avgN_adv1_eff50_compound',
               # 'ENC_200m_sbg5m_avgN_LPD1m',
               # 'ENC_200m_sbg5m_noChannels_avgN'
               ]

for model_root in model_roots:
    mod = SfincsModel(root=model_root, mode='r', data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
    cat = mod.data_catalog
    mod.read_results()

    # Get the station data for querying SFINCS results
    mod_zs_da = mod.results['point_zs']
    mod_zs_lookup = pd.DataFrame()
    mod_zs_lookup['station_id'] = mod_zs_da['station_id'].values
    mod_zs_lookup['station_name'] = [x.decode('utf-8').strip() for x in mod_zs_da['station_name'].values]
    mod_zs_lookup['data_source'] = [x.rsplit('_', 1)[0] for x in mod_zs_lookup['station_name']]
    mod_zs_lookup['data_source_id'] = [x.split('_')[-1] for x in mod_zs_lookup['station_name']]

    # Create directory to save the output
    out_dir = os.path.join(os.getcwd(), model_root, 'validation', 'waterlevel')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # LOAD THE OBSERVED WATER LEVEL TIMESERIES
    data_sources = ['USGS', 'NOAA']
    obs_sources = ['usgs_waterlevel_florence', 'noaa_waterlevel_florence']
    station_stats_mdf = pd.DataFrame()
    for counter in range(len(obs_sources)):
        dataset_name = obs_sources[counter]
        agency = data_sources[counter]
        # Load the observation data from the data catalog for the model region and time
        obs_da = cat.get_geodataset(dataset_name, geom=mod.region, variables=["waterlevel"],
                                    time_tuple=mod.get_model_time())
        pts, obs = clean_obs_coords(obs_df=obs_da, source_crs=4326, target_crs=mod.crs.to_epsg())

        # Loop through the observation locations and extract model data
        mod_zs_lookup_sub = mod_zs_lookup[mod_zs_lookup['data_source'] == agency]

        # Create empty lists/df to save information to when looping through the observation gages
        calculate_gage_stats = True
        if calculate_gage_stats is True:
            station_stats = pd.DataFrame()
            invalid_obs, valid_obs = [], []
            for index, row in mod_zs_lookup_sub.iterrows():
                data_source_id = int(row['data_source_id'])
                if data_source_id in obs_da.index.values.tolist():
                    obs_zs = obs.sel(index=data_source_id)
                    mod_zs = mod_zs_da.sel(stations=index)

                    # Add observed and modeled data into a single dataframe
                    obs_df = pd.DataFrame(data=obs_zs.values, index=obs_zs.time.values, columns=['Observed'])
                    mod_df = pd.DataFrame(data=mod_zs.values, index=mod_zs.time.values, columns=['Modeled'])
                    merged_df = pd.concat([obs_df, mod_df], axis=1)
                    merged_df.dropna(inplace=True)

                    # If the dataframe is empty or there are fewer than 20 observation points,
                    # append the gage ID to the list of "invalid_obs"
                    if merged_df.empty or len(merged_df) < 50:
                        print(f'No data for gage: {data_source_id}')
                        invalid_obs.append(data_source_id)
                    else:
                        valid_obs.append(data_source_id)

                    # Calculate the hydrograph stats at the station and add to master dataframe
                    ss, _ = calculate_stats(station_id=data_source_id, df=merged_df, tstart=None, tend=None)
                    ss['source'] = agency
                    station_stats = pd.concat([station_stats, ss], ignore_index=True)

        station_stats.set_index('station_id', drop=True, inplace=True)
        pts.set_index('site_no', drop=True, inplace=True)
        stats_out = pd.concat([pts, station_stats], axis=1, ignore_index=False)
        stats_out = gpd.GeoDataFrame(stats_out, geometry='geometry', crs=mod.crs)
        stats_out['x'] = stats_out.geometry.x
        stats_out['y'] = stats_out.geometry.y

        station_stats_mdf = pd.concat([station_stats_mdf, stats_out], axis=0)

    # Save the hydrograph stats at each station
    station_stats_mdf = station_stats_mdf.dropna(how='any')
    station_stats_mdf.drop(columns='geometry').to_csv(os.path.join(out_dir, 'hydrograph_stats_by_gageID.csv'))

    ''' Gage stats by Zone '''
    # BY STATE
    states = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
    states = states[states['NAME'].isin(['South Carolina', 'North Carolina'])]
    states.to_crs(epsg=32617, inplace=True)
    states = states[['STUSPS', 'geometry']]
    stats_out_zone = gpd.tools.sjoin(left_df=station_stats_mdf, right_df=states, how='left')
    stats_out_zone.drop(columns='index_right', inplace=True)

    # BY HUC6
    huc6 = gpd.read_file(
        r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
    huc6.to_crs(epsg=32617, inplace=True)
    huc6 = huc6[['Name', 'geometry']]
    huc6.columns = ['HUC6', 'geometry']
    stats_out_zone = gpd.tools.sjoin(left_df=stats_out_zone, right_df=huc6, how='left')
    stats_out_zone.drop(columns='index_right', inplace=True)
    stats_out_zone.drop(columns='geometry').to_csv(os.path.join(out_dir, 'hydrograph_stats_by_gageID.csv'))

    # Calculate zonal stats
    stats_out_zone['domain'] = 'domain'
    grp_stats_df = pd.DataFrame()
    for group in ['STUSPS', 'HUC6', 'domain']:
        df = stats_out_zone[['mae', 'rmse', 'nse', 'bias', 'r', 'r2', 'pe', 'tpe', group]]
        grp_stats = df.groupby([group]).describe(percentiles=[0.1, 0.5, 0.9]).round(2)
        grp_stats_df = pd.concat([grp_stats_df, grp_stats], axis=0)
    grp_stats_df.to_csv(os.path.join(out_dir, 'stats_by_zone.csv'))
