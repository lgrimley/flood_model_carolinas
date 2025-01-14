from shapely.geometry import LineString
import datetime
import os
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd

# Processing TC tracks

def get_track_datetime(df):
    datetime_list = []
    for i in range(len(df)):
        year = df['year100'][1]
        month = df['month100'][i]
        day = df['day100'][i]
        hour = df['hour100'][i]
        try:
            x = pd.to_datetime(datetime.datetime(year, month, day, hour))
        except:
            x = 0
        datetime_list.append(x)
    df['datetime'] = datetime_list
    return df


def get_track_info_in_df(tc_id, tc_tracks):
    df = pd.DataFrame()
    py_index = tc_id - 1  # TC_ID corresponds to row 0 in python
    for v in tc_tracks.keys():
        if v in ['__header__', '__version__', '__globals__', 'freq']:
            continue
        else:
            if len(tc_tracks[v][py_index]) == 1:
                df[v] = tc_tracks[v][py_index].item()
            else:
                df[v] = tc_tracks[v][py_index, :]

    df = get_track_datetime(df=df)

    return df


def track_as_gdf(df):
    df = df[df['datetime'] != 0]
    track_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['lon100'], y=df['lat100'], crs=4326))
    return track_gdf


def track_points_to_linestring(tc_tracks, output_shpfile=None):
    lat = tc_tracks['lat100']
    lon = tc_tracks['lon100']
    tc_ids = np.arange(start=1, stop=len(lat) + 1, step=1)

    tc_tracks_gdf = pd.DataFrame()
    for tc_id in tc_ids:
        py_index_loc = tc_id - 1  # Python index starts from 0 instead of 1 like matlab
        stm1_lat = lat[py_index_loc, :]
        stm1_lon = lon[py_index_loc, :]

        # Remove the zero points
        stm1_lat = stm1_lat[stm1_lat != 0]
        stm1_lon = stm1_lon[stm1_lon != 0]

        # Create a geospatial object that can be compared with the study area
        track = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=stm1_lon, y=stm1_lat, crs=4326))
        track['tc_id'] = tc_id
        track_line = track.groupby('tc_id')['geometry'].apply(lambda x: LineString(x.tolist()))
        track_line = gpd.GeoDataFrame(track_line, geometry='geometry', crs=4326)
        tc_tracks_gdf = pd.concat(objs=[tc_tracks_gdf, track_line], axis=0)

    tc_tracks_gdf['tc_id'] = tc_tracks_gdf.index
    if output_shpfile is not None:
        tc_tracks_gdf.to_file(output_shpfile)

    return tc_tracks_gdf


def select_TCs_by_AOI_intersect(aoi, tc_tracks_gdf, buffer_km=0):
    # Buffer the area of interest polygon
    aoi_buff = aoi.buffer(distance=buffer_km * 1000).to_crs(crs=4326)

    # See if each TC track (polyline) intersects with the polygon, returns T/F
    select_tc = tc_tracks_gdf.geometry.apply(lambda g: aoi_buff.intersects(g))
    values, counts = np.unique(select_tc, return_counts=True)
    print(f'{counts[1]} of {len(tc_tracks_gdf)} TC tracks intersect with the study area')

    # Add column to the TC track gdf
    tc_tracks_gdf['aoi_sel'] = select_tc

    return aoi_buff, tc_tracks_gdf


def TCR_precip_stats_to_netcdf(tc_ids, inputdir, outputdir):
    precip_sum_list = []
    precip_maxRR_list = []
    counter = 0
    for tc_id in tc_ids:
        filename = f'{str(tc_id).zfill(4)}.nc'
        d = xr.open_dataset(os.path.join(inputdir, filename))

        # Cumulative precipitation
        precip_sum = d.sum(dim='time')
        precip_sum_list.append(precip_sum)

        # Max rain rate across the grid
        max_rain_rate = d.max(dim='time')
        precip_maxRR_list.append(max_rain_rate)

        print(f'Processed {counter} out of {len(tc_ids)}')
        counter += 1

    print('Writing netcdfs...')
    da_sum = xr.concat(precip_sum_list, dim='run')
    da_sum['run'] = xr.IndexVariable('run', tc_ids)
    da_sum.to_netcdf(os.path.join(outputdir, 'precip_TC_AvgTotPrecip.nc'))

    da_max = xr.concat(precip_maxRR_list, dim='run')
    da_max['run'] = xr.IndexVariable('run', tc_ids)
    da_max.to_netcdf(os.path.join(outputdir, 'precip_TC_maxRainRate.nc'))

    return da_sum, da_max


def get_TCs_Vmax(tc_ids, inputdir, outputdir):
    Vmax_list = []
    counter = 0
    for tc_id in tc_ids:
        filename = f'{str(tc_id).zfill(4)}.nc'
        d = xr.open_dataset(os.path.join(inputdir, filename))

        # Max wind speed across the grid
        max_wndspd = d['wind_speed'].max(dim='time')
        Vmax_list.append(max_wndspd)

        print(f'Processed {counter} out of {len(tc_ids)}')
        counter += 1

    print('Writing netcdfs...')
    da = xr.concat(Vmax_list, dim='run')
    da['run'] = xr.IndexVariable('run', tc_ids)
    da.to_netcdf(os.path.join(outputdir, 'TC_maxWindSpeeds.nc'))

    return da


def find_nearest_non_nan_index(df, col_name, index):
    """Finds the nearest non-NaN index value in the specified column.

    Args:
        df (pd.DataFrame): The DataFrame.
        col_name (str): The name of the column to search in.
        index (int): The index value from which to search.

    Returns:
        int: The index of the nearest non-NaN value, or None if no such value exists.
    """

    # Get the non-NaN indices in the column
    non_nan_indices = df[col_name].dropna().index

    # Find the nearest non-NaN index
    if non_nan_indices.empty:
        return None  # No non-NaN values in the column
    else:
        return non_nan_indices[np.abs(non_nan_indices - index).argmin()]


def calculate_station_highTide_time(da: xr.Dataset,
                                    tref: pd.DatetimeIndex,
                                    station_names: list = None) -> pd.DataFrame:
    if station_names is None:
        station_names = da.index.values

    highTide_times = []
    sta_keep = []
    for station in station_names:
        # Get the high tide time lag of the ADCIRC Storm Tide gages
        # Select the first 12 hours of the data to search
        data = da.sel(index=station).sel(time=slice(tref, tref + pd.to_timedelta('12h')))
        df = data.to_dataframe()

        df['delta_waterlevel'] = df['waterlevel'].diff()  # Calculate the difference in water levels across the 12 hours
        increasing_trend = df[df['delta_waterlevel'] > 0]
        try:
            if len(increasing_trend) > 0:
                # Subset to where water level is increasing (e.g., rising tide)
                # Pick the datetime of the max water level in the 12 hours window
                ind = np.argmax(increasing_trend['waterlevel'])
                station_highTide_time = increasing_trend.index[ind]
            else:
                # Otherwise just pick the datetime of the max water level in the 12 hours window
                ind = np.argmax(df['waterlevel'])
                station_highTide_time = df.index[ind]
            highTide_times.append(station_highTide_time)
            sta_keep.append(station)
        except:
            print(f"Issue with {station}")
            pass

    highTide_df = pd.DataFrame()
    highTide_df['station'] = sta_keep
    highTide_df['highTide_time'] = highTide_times
    highTide_df.set_index(keys='station', drop=True, inplace=True)
    return highTide_df


def process_tmax_in_hours(tmax: xr.DataArray, time_min_hr: int = 0) -> xr.DataArray:
    # Get the time of inundation above twet_threshold
    #tmax = da['tmax'].max(dim='timemax')

    # Create a mask of the NaT values
    #mask = np.isnat(tmax.values)
    mask = np.isnan(tmax.values)

    # Convert timedelta64[ns] to float
    tmax = tmax.astype(float)

    # Mask out the NaT values
    tmax = tmax.where(~mask, np.nan)

    # Convert nanoseconds to hours
    tmax = tmax / (3.6 * 10 ** 12)

    # Subset the data futher with a minimum time threshold of interest
    tmax = xr.where(tmax >= time_min_hr, tmax, np.nan)

    return tmax


def calculate_flooded_area_by_process(da: xr.DataArray, tc_index: int):
    unique_codes, cell_counts = np.unique(da.data, return_counts=True)
    fld_area1 = cell_counts.copy()
    res = 200  # grid cell resolution in meters
    fld_area1 = fld_area1 * (res * res) / (1000 ** 2)  # square km

    # Cleanup the dataframe
    fld_area1 = pd.DataFrame(fld_area1).T
    fld_area1.columns = unique_codes
    expected = pd.DataFrame(columns=[0.0, 1.0, 2.0, 3.0, 4.0, np.NAN])
    fld_area = pd.concat(objs=[expected, fld_area1], axis=0, ignore_index=True)

    fld_area.columns = ['NoFlood', 'Coastal', 'Coastal-Compound', 'Runoff', 'Runoff-Compound', 'Inactive']
    fld_area['Total_Area'] = fld_area.sum(axis=1)  # calculate the total area of flooding
    fld_area['Compound'] = fld_area['Coastal-Compound'] + fld_area['Runoff-Compound']
    fld_area['Total_Flooded'] = fld_area[['Coastal', 'Runoff', 'Compound']].sum(axis=1)
    fld_area['tc_index'] = tc_index
    fld_area.set_index('tc_index', inplace=True, drop=True)

    return fld_area


def get_landfall_info(tc_df, gage_locs, clip_gdf):
    tc_gdf = gpd.GeoDataFrame(tc_df, geometry=gpd.points_from_xy(x=tc_df['lon100'], y=tc_df['lat100'], crs=4326))
    tc_gdf = tc_gdf.clip(clip_gdf)
    gage_locs_gdf = gpd.GeoDataFrame(gage_locs,
                                     geometry=gpd.points_from_xy(x=gage_locs['x'], y=gage_locs['y'], crs=4326))

    # Create an empty list to store the results
    min_distances = []

    # Loop through each point in df1
    for i, point1 in tc_gdf.iterrows():
        # Calculate the distance from the current point in df1 to all points in df2
        distances = gage_locs_gdf.geometry.apply(lambda point2: point1.geometry.distance(point2))

        # Find the smallest distance and the corresponding point from df2
        min_distance = np.round(distances.min(), 5)
        closest_point = gage_locs_gdf.loc[distances.idxmin()]

        # Store the result for the current point in df1
        min_distances.append((i, min_distance, closest_point['gage_id']))

    # Create a DataFrame with the results
    result_df = gpd.GeoDataFrame(min_distances, columns=["Index_in_df1", "Min_Distance", "Closest_Point_in_df2"])

    lf_idx = result_df.loc[result_df['Min_Distance'].idxmin()]
    landfall_info = tc_gdf[tc_gdf.index == lf_idx['Index_in_df1']]
    landfall_info['Min_Distance'] = lf_idx['Min_Distance']
    landfall_info['Closest_Point_in_df2'] = lf_idx['Closest_Point_in_df2']

    return landfall_info





