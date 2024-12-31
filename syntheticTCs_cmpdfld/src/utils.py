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


