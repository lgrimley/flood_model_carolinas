from shapely.geometry import LineString
import datetime
import os
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path


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


def TCR_precip_stats2netcdf(tc_ids: list, inputdir: Path, outputdir: Path,
                            rr_threshold: int=5) -> xr.Dataset:
    cumP_list = []
    maxRR_list = []
    meanRR_list = []
    meanRR_thresh_list = []
    counter = 0
    for tc_id in tc_ids:
        filename = f'{str(tc_id).zfill(4)}.nc'
        d = xr.open_dataset(os.path.join(inputdir, filename))

        # Cumulative precipitation
        precip_sum = d.sum(dim='time')
        cumP_list.append(precip_sum)

        # Max rain rate across the grid
        max_rain_rate = d.max(dim='time')
        maxRR_list.append(max_rain_rate)

        # Max rain rate across the grid
        mean_rain_rate = d.mean(dim='time')
        meanRR_list.append(mean_rain_rate)

        mask = d > rr_threshold
        mean_rain_rate_thresh = d.where(mask).mean(dim='time')
        meanRR_thresh_list.append(mean_rain_rate_thresh)

        print(f'Processed {counter} out of {len(tc_ids)}')
        counter += 1

    ds_sum = xr.concat(cumP_list, dim='tc_id')
    ds_sum['tc_id'] = xr.IndexVariable('tc_id', tc_ids)
    ds_sum = ds_sum.rename({'precip':'total_precip'})

    ds_max = xr.concat(maxRR_list, dim='tc_id')
    ds_max['tc_id'] = xr.IndexVariable('tc_id', tc_ids)
    ds_max = ds_max.rename({'precip':'max_RR'})

    ds_mean = xr.concat(meanRR_list, dim='tc_id')
    ds_mean['tc_id'] = xr.IndexVariable('tc_id', tc_ids)
    ds_mean = ds_mean.rename({'precip':'mean_RR'})

    ds_mean_thresh = xr.concat(meanRR_thresh_list, dim='tc_id')
    ds_mean_thresh['tc_id'] = xr.IndexVariable('tc_id', tc_ids)
    ds_mean_thresh = ds_mean_thresh.rename({'precip':'mean_RR_thresh'})

    ds = xr.merge([ds_sum, ds_max, ds_mean, ds_mean_thresh])
    ds.assign_attrs(RR_threshold=rr_threshold)
    outfile = os.path.join(outputdir, 'tc_precipitation_stats.nc')
    ds.to_netcdf(outfile)
    print(f'Created {outfile}')

    return ds


def TC_windspd_stats2netcdf(tc_ids: list, inputdir: Path,
                            outputdir: Path) -> xr.Dataset:
    vmax_list = []
    vmean_list = []
    wnd_direction_list = []
    mean_wndspd_thresh_list = []
    counter = 0
    for tc_id in tc_ids:
        filename = f'{str(tc_id).zfill(4)}.nc'
        d = xr.open_dataset(os.path.join(inputdir, filename))

        # Calculate the wind speed (magnitude) and direction (angle)
        wind_direction_rad = np.arctan2(d['wind10_v'], d['wind10_u'])  # Direction in radians
        wind_direction_deg = np.degrees(wind_direction_rad)
        # Wind conventions: The direction represents the origin of the wind.
        # 0°: Wind coming from the north (i.e., blowing southward).
        # 90°: Wind coming from the east (i.e., blowing westward).
        # 180°: Wind coming from the south (i.e., blowing northward).
        # 270°: Wind coming from the west (i.e., blowing eastward).
        # Normalize wind direction to be between 0° and 360°
        normalized_direction = (wind_direction_deg + 360) % 360
        wnd_direction = normalized_direction.mean(dim='time')
        wnd_direction_list.append(wnd_direction)

        # Max wind speed across the grid
        max_wndspd = d['wind_speed'].max(dim='time')
        vmax_list.append(max_wndspd)

        # Mean wind speed across the grid
        mean_wndspd = d['wind_speed'].mean(dim='time')
        vmean_list.append(mean_wndspd)

        # Mean wind speed with threshold
        mask = d['wind_speed'] > 5
        mean_wndspd_thresh = d['wind_speed'].where(mask).mean(dim='time')
        mean_wndspd_thresh_list.append(mean_wndspd_thresh)

        print(f'Processed {counter} out of {len(tc_ids)}')
        counter += 1

    ds1 = xr.concat(vmax_list, dim='tc_id')
    ds1['tc_id'] = xr.IndexVariable('tc_id', tc_ids)
    ds1 = ds1.to_dataset()
    ds1 = ds1.rename({'wind_speed':'max_windspd'})

    ds2 = xr.concat(vmean_list, dim='tc_id')
    ds2['tc_id'] = xr.IndexVariable('tc_id', tc_ids)
    ds2 = ds2.to_dataset()
    ds2 = ds2.rename({'wind_speed':'mean_windspd'})

    ds4 = xr.concat(mean_wndspd_thresh_list, dim='tc_id')
    ds4['tc_id'] = xr.IndexVariable('tc_id', tc_ids)
    ds4 = ds4.to_dataset()
    ds4 = ds4.rename({'wind_speed':'mean_windspd_threshold'})

    ds3 = xr.concat(wnd_direction_list, dim='tc_id')
    ds3['tc_id'] = xr.IndexVariable('tc_id', tc_ids)
    ds3 = ds3.to_dataset()
    ds3 = ds3.rename({'wind10_v':'mean_direction_deg'})

    ds = xr.merge([ds1, ds2, ds4, ds3])
    outfile = os.path.join(outputdir, 'tc_windspeed_stats.nc')
    ds.to_netcdf(outfile)
    print(f'Created {outfile}')

    return ds


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


def calculate_flooded_area_by_process(da: xr.DataArray, tc_index: int=0):
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
    tc_gdf = tc_gdf.to_crs(32617)
    gage_locs_gdf = gage_locs_gdf.to_crs(32617)

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



def classify_zsmax_by_process(da: xr.DataArray, da_diff: xr.DataArray, hmin: float=0.1) -> xr.Dataset:
    # Outputs a data array with the zsmax attributed to processes (codes 0 to 4)
    # Create masks based on the driver that caused the max water level given a depth threshold hmin
    compound_mask = da_diff > hmin
    coastal_mask = da.sel(scenario='coastal').fillna(0) > da.sel(scenario=['runoff']).fillna(0).max('scenario')
    runoff_mask = da.sel(scenario='runoff').fillna(0) > da.sel(scenario=['coastal']).fillna(0).max('scenario')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    da_classified = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
                     + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()

    da_classified.name = 'zsmax_classified'
    da_classified = da_classified.assign_attrs(hmin=hmin,
                                               no_class=0, coast_class=1, coast_compound_class=2,
                                               runoff_class=3, runoff_compound_class=4)
    da_classified = da_classified.astype(int)

    ds = xr.Dataset({'zsmax_diff': da_diff,
                     'zsmax_attr': da_classified})
    return ds


def calc_diff_in_zsmax_compound_minus_max_individual(da: xr.DataArray) -> xr.DataArray:
    # Outputs a data array of the diff in water level compound minus max. single driver
    # Calculate the max water level at each cell across the coastal and runoff drivers
    da_single_max = da.sel(scenario=['runoff', 'coastal']).max('scenario')

    # Calculate the difference between the max water level of the compound and the max of the individual drivers
    da_diff = (da.sel(scenario='compound') - da_single_max).compute()
    da_diff.name = 'zsmax_diff'
    da_diff.attrs = da.attrs
    da_diff.assign_attrs(units='m')
    da_diff.assign_attrs(description = 'diff in waterlevel compound minus max. single driver')

    return da_diff