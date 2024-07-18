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


def get_observed_wl(model_root, catalog, dataset_name, buffer):
    mod = SfincsModel(root=model_root, mode='r')
    cat = hydromt.DataCatalog(catalog)
    tstart, tstop = mod.get_model_time()
    obs = cat.get_geodataset(dataset_name,
                             geom=mod.region,
                             buffer=buffer,
                             variables=["waterlevel"],
                             time_tuple=(tstart, tstop))
    if 'geometry' in list(obs.coords):
        pts = gpd.GeoDataFrame(geometry=obs.geometry.values, crs=4326)
        pts = pts.to_crs(mod.crs.to_epsg())
        obs.geometry.values = pts.geometry.values
    else:
        pts = gpd.points_from_xy(x=obs.x.values, y=obs.y.values, crs=4326)
        pts = pts.to_crs(mod.crs.to_epsg())
        obs.x.values = pts.x
        obs.y.values = pts.y
    return obs


def match_usgs_to_model_station(mod_station_df, obs_data):
    # Compute the difference between the x-y coordinates of the model station vs. all observation stations
    if 'x' in list(obs_data.coords.keys()):
        absx = np.abs(mod_station_df['station_x'].values - obs_data['x'].values)
        absy = np.abs(mod_station_df['station_y'].values - obs_data['y'].values)
    else:
        xx = obs_data.vector.geometry.x.values
        yy = obs_data.vector.geometry.y.values
        absx = np.abs(mod_station_df['station_x'].values - xx)
        absy = np.abs(mod_station_df['station_y'].values - yy)

    # Select the observation station that is closest to the x-y of the model station
    c = np.maximum(absx, absy)
    min_distance = np.min(c)

    # Get the index of the observation station and pull the data
    obs_index = np.where(c == min_distance)[0][0]

    return [obs_index, min_distance]


def create_merged_df(obs_ind, obs_data, mod_ind, mod_data):
    obs_df = obs_data.isel(index=obs_ind)
    obs_df2 = pd.DataFrame(data=obs_df.values,
                           index=obs_df.time.values,
                           columns=['Observed'])
    mod_df = mod_data.isel(stations=mod_ind)
    mod_df2 = pd.DataFrame(data=mod_df['point_zs'].values,
                           index=mod_df.time.values,
                           columns=['Modeled'])
    merged_df = pd.concat([obs_df2, mod_df2], axis=1)
    merged_df.dropna(inplace=True)
    return merged_df


def calculate_stats(df, tstart=None, tend=None):
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
        print('CC problem')
        r = 0

    # Coefficient of Determination
    r2 = r ** 2

    # Save stats in a dataframe for output
    stats = pd.DataFrame(data={'mae': round(mae, 2),
                               'rmse': round(rmse, 2),
                               'nse': round(nse, 2),
                               'bias': round(bias, 2),
                               'r': round(r, 2),
                               'r2': round(r2, 2),
                               'pe': round(pe, 2),
                               'tpe': round(tpe.seconds, 1),
                               'obs_peaktime': df.Observed.idxmax().strftime("%m/%d/%Y, %H:%M:%S"),
                               'mod_peaktime': df.Modeled.idxmax().strftime("%m/%d/%Y, %H:%M:%S")
                               }, index=[0])
    return stats


def get_station_info(obs_index, obs_data, mod_index, mod_data, station_info_df, match_distance, flag):
    obs_df = obs_data.isel(index=obs_index)
    mod_df = mod_data.isel(stations=mod_index)

    if 'geometry' in obs_df.coords:
        x_val = obs_df.geometry.values.item().x
        y_val = obs_df.geometry.values.item().y
    else:
        x_val = obs_df.x.values.item()
        y_val = obs_df.y.values.item()

    sta = pd.DataFrame(data={'mod_station_ID': int(mod_df.station_id.values.item()),
                             'mod_x': round(mod_df.station_x.values.item()),
                             'mod_y': round(mod_df.station_y.values.item()),
                             'obs_station_ID': int(obs_df.index.values.item()),
                             'obs_x': round(x_val),
                             'obs_y': round(y_val),
                             'distance': round(int(match_distance)),
                             'boundary': flag},
                       index=[0])
    out = pd.concat([station_info_df, sta], ignore_index=True)
    return out.round()


def plot_hydrograph(merged_df, model_station_id, obs_station_id, flag, out_dir, group=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    plt.rcParams.update({'font.size': 9})
    ax.plot(merged_df['Observed'], linewidth=0.5, color='black', alpha=0.6)
    ax.plot(merged_df['Modeled'], linewidth=1.5, color='blue')
    plt.legend(['Observed', 'Modeled'])
    d1 = merged_df.drop(axis=1, columns=["Modeled"])
    d1['Datetime'] = d1.index
    d1.plot.scatter(x='Datetime', y='Observed',
                    c='black', style='.', s=3,
                    ax=ax)
    ax.set_ylabel('Water Level (m, NAVD88)')
    ax.set_xlabel('')
    fig.autofmt_xdate()
    ax.grid(axis='y', color='gray', linewidth=0.5, linestyle='dashed', alpha=0.5)
    plt.title('Model Station ' + str(model_station_id))
    plt.tight_layout()
    if flag == 1:
        filename = 'BND_Station_mod_' + str(model_station_id) + '_obs_' + str(obs_station_id) + '.png'
    else:
        filename = 'Station_mod_' + str(model_station_id) + '_obs_' + str(obs_station_id) + '.png'
    if group:
        filename = str(group) + '_' + filename
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


# Load Data Catalog and read in SFINCS model
yml = r'Z:\users\lelise\data\data_catalog.yml'
os.chdir(r'Z:\users\lelise\projects\Carolinas\sfincs\model_improvments')
model_root = 'carolinas_200m_sbg5m'
mod = SfincsModel(root=model_root, mode='r')
mod.read()
mod_data = xr.open_dataset('./' + model_root + '/sfincs_his.nc')
mod_bnd = mod.forcing['bzs']

ds = 'usgs_rpd'
dataset_name = [
    #'usgs_waterlevel_florence_rapid_deployment',
    #'ncem_waterlevel_florence',
    #'noaa_waterlevel_florence',
    'usgs_waterlevel_florence'
    ]
dataset_name = dataset_name[0]
dist = 10
bnd_dst = 100
obs_data = get_observed_wl(model_root=model_root,
                           catalog=yml,
                           dataset_name=dataset_name,
                           buffer=0)

# Load in HUC6 boundary
huc6_boundary = gpd.read_file(r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape'
                              r'\WBDHU6.shp')
huc6_boundary.to_crs(mod.crs, inplace=True)
huc6_boundary = huc6_boundary[["HUC6", "Name", "geometry"]]

mod_stations = mod_data.station_id.to_dataframe()
mod_stations = mod_stations.loc[:, ~mod_stations.columns.duplicated()].copy()
mod_stations = gpd.GeoDataFrame(
    mod_stations,
    geometry=gpd.points_from_xy(mod_stations.point_x, mod_stations.point_y),
    crs=mod_data.crs.epsg_code)
mod_stations = gpd.tools.sjoin(left_df=mod_stations,
                               right_df=huc6_boundary,
                               how='left')

# Setup space to store data
station_info_df = pd.DataFrame()
station_stats = pd.DataFrame()
notmatched = []
notmatched_dist = []

# Create directories
out_dir1 = os.path.join(r'./' + model_root + '/validation')
if not os.path.exists(out_dir1):
    os.mkdir(out_dir1)
out_dir2 = os.path.join(r'./' + model_root + '/validation/waterlevel_gages')
if not os.path.exists(out_dir2):
    os.mkdir(out_dir2)
out_dir3 = os.path.join(r'./' + model_root + '/validation/waterlevel_gages/hydrographs_' + ds)
if not os.path.exists(out_dir3):
    os.mkdir(out_dir3)

# Loop through and do some magic
for mod_index in range(len(mod_data['station_id'])):

    # Get the data for the model station
    mod_station_df = mod_data.isel(stations=mod_index)

    _, bnd_distance = match_usgs_to_model_station(mod_station_df, obs_data=mod_bnd)
    if bnd_distance < bnd_dst:
        flag = 1
        print('Found boundary gage.')
    else:
        flag = 0

    obs_index, distance = match_usgs_to_model_station(mod_station_df, obs_data)

    if distance > dist:
        print('No match for model station: ' + str(mod_station_df['station_id'].values.item()))
        notmatched.append(mod_station_df['station_id'].values.item())
        notmatched_dist.append(distance)
        print('The closest USGS gage is: ' + str(distance))
        continue

    obs_station_df = obs_data.isel(index=obs_index)

    # Create a merged and cleaned DF for the modeled and observed data at this location
    merged_df = create_merged_df(obs_ind=obs_index, obs_data=obs_data,
                                 mod_ind=mod_index, mod_data=mod_data)

    if merged_df.empty:
        continue

    # The model station index is not the Station ID used by SFINCS. It is the Station ID minus 1.
    mod_station_id = int(mod_station_df['station_id'].values.item())
    obs_usgs_id = int(obs_station_df['index'].values.item())
    h = mod_stations[mod_stations['station_id'] == mod_station_df['station_id'].values.item()]['Name'].iloc[0]

    plot_hydrograph(merged_df=merged_df,
                    model_station_id=mod_station_id,
                    obs_station_id=obs_usgs_id,
                    out_dir=out_dir3,
                    flag=flag,
                    group=h)

    station_info_df = get_station_info(obs_index=obs_index, obs_data=obs_data,
                                       mod_index=mod_index, mod_data=mod_data,
                                       station_info_df=station_info_df,
                                       match_distance=distance,
                                       flag=flag)

    station_stats = pd.concat([station_stats,
                               calculate_stats(df=merged_df, tstart=None, tend=None)],
                              ignore_index=True)

# Write out the results
write = True
if write is True:
    joined = station_info_df.merge(station_stats,
                                   left_index=True,
                                   right_index=True)
    joined = joined.merge(mod_stations,
                          left_on='mod_station_ID',
                          right_on='station_id',
                          left_index=False,
                          right_index=False)
    joined.drop(mod_stations.columns.to_list()[0:7], axis=1, inplace=True)

    # Subset data
    matched_out = joined[(joined['distance'] < dist) & (joined['boundary'] == 0)]
    boundary_pts = joined[joined['boundary'] == 1]

    # Output to CSV
    matched_out.to_csv(os.path.join(out_dir2, ('gage_stats_' + ds + '.csv')),
                       index_label=False, index=False)
    boundary_pts.to_csv(os.path.join(out_dir2, ('gage_stats_' + ds + '_at_boundary.csv')),
                        index_label=False,
                        index=False)
    stat_summary = matched_out.describe()

    # Output to Shapefile
    df = gpd.GeoDataFrame(matched_out,
                          geometry=gpd.points_from_xy(x=matched_out['mod_x'].values,
                                                      y=matched_out['mod_y'].values), crs=32618)
    df.to_file(os.path.join(out_dir2, (ds + '_matched.shp')))

    # No matched
    nogo = mod_data.where(mod_data['station_id'].isin(notmatched),
                          drop=True)
    df = gpd.GeoDataFrame(nogo['station_id'].values,
                          geometry=gpd.points_from_xy(x=nogo.point_x.values,
                                                      y=nogo.point_y.values), crs=32618)
    df.columns = ['name', 'geometry']
    df['distance'] = notmatched_dist
    df.to_file(os.path.join(out_dir2, (ds + '_mod_notmatched.shp')))
