import os
import datetime
import numpy as np
import hydromt
import pandas as pd
import geopandas as gpd
import xarray as xr
import scipy.io as sio
from shapely.geometry import LineString
from pathlib import Path
from typing import Self



def calculate_station_highTide_time(da: xr.Dataset,
                                    tref: pd.DatetimeIndex, station_names: list = None) -> pd.DataFrame:
    if station_names is None:
        station_names = da.index.values

    highTide_times = []
    for station in station_names:
        # Get the high tide time lag of the ADCIRC Storm Tide gages
        data = da.sel(index=station).sel(time=slice(tref, tref + pd.to_timedelta('12h')))
        df = data.to_dataframe()
        df['delta_waterlevel'] = df['waterlevel'].diff()  # Calculate the difference in water levels across time
        increasing_trend = df[df['delta_waterlevel'] > 0]  # Identify where water level is increasing
        station_highTide_time = increasing_trend['waterlevel'].idxmax()
        highTide_times.append(station_highTide_time)

    highTide_df = pd.DataFrame()
    highTide_df['station'] = station_names
    highTide_df['highTide_time'] = highTide_times
    highTide_df.set_index(keys='station', drop=True, inplace=True)
    return highTide_df


class DataPaths:
    adcirc_reanalysis_locs_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\EDSReanalysis_data\full_data_meta.csv')
    adcirc_reanalysis_data_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\EDSReanalysis_data\EDSReanalysis_V2_1992_2022.nc')
    adcirc_stormTide_locs_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')
    adcirc_loc_mapping_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\stormTide\map_Reanalysis_to_stormTideLocs.csv')

    def __init__(
            self: Self,
            root: Path = None,
            tracks_filepath: Path = None,
            adcirc_stormTide_dir: Path = None,
            wind_dir: Path = None,
            precip_dir: Path = None
    ):
        self.root = root
        self.tracks_filepath = os.path.join(self.root, tracks_filepath)
        self.adcirc_stormTide_dir = os.path.join(self.root, adcirc_stormTide_dir)
        self.wind_dir = os.path.join(self.root, wind_dir)
        self.precip_dir = os.path.join(self.root, precip_dir)


class SyntheticTrack:
    def __init__(
            self: Self,
            DataPaths,
            DatCat: hydromt.DataCatalog = None,
            tc_index: int = None,
    ):

        self.DataPaths = DataPaths
        self.DatCat = DatCat
        self.tc_index = tc_index
        self.filename = f'{str(tc_index).zfill(4)}.nc'

        # Read in a geodataframe of the track information from the MAT file
        self.track_information = self.prepare_track_information(tracks_file=DataPaths.tracks_filepath,
                                                                tc_index=tc_index)
        self.track_time = self.get_track_time(track_information=self.track_information)
        self.stormTide = self.get_stormTide_data(DatCat=DatCat, dataName=f'stormTide_{tc_index}')
        self.adcirc_time = self.get_stormTide_time(stormTide=self.stormTide)

        tmin = min(self.track_time[0], self.adcirc_time[0])
        tmax = max(self.track_time[1], self.adcirc_time[1])
        self.time_range = (tmin, tmax)
        print(self.time_range)

        # Create a dataframe for mapping the ADCIRC reanalysis and stormtide stations because both are used
        # for creating the coastal water level boundary conditions
        self.coastal_locations_mapped = self.coastal_boundary_condition_locations(
            adcirc_reanalysis_locs_filepath=DataPaths.adcirc_reanalysis_locs_filepath,
            adcirc_stormTide_locs_filepath=DataPaths.adcirc_stormTide_locs_filepath)

        # Read in all ADCIRC reanalysis data for the TC time period, buffer 15 days on either end of the storm
        buff_time = (self.time_range[0] - pd.Timedelta(days=15), self.time_range[1] + pd.Timedelta(days=15))
        self.reanalysis_data = self.DatCat.get_geodataset(data_like=f'tide_reanalysis',
                                                          time_tuple=buff_time)


        if self.DataPaths.wind_dir is not None:
            self.wind = self.prepare_gridded_data(DatCat=DatCat,
                                                  dataName=f'wind_{tc_index}', track_time=self.time_range)
        if self.DataPaths.precip_dir is not None:
            self.precip = self.prepare_gridded_data(DatCat=DatCat,
                                                    dataName=f'precip_{tc_index}', track_time=self.time_range)
        print('Track geometry and boundary condition files loaded.')

        self.tide_offset_table = self.calculate_reanalysis_tide_offset(mapped_stations=self.coastal_locations_mapped,
                                                                       stormTide=self.stormTide,
                                                                       reanalysis=self.reanalysis_data,
                                                                       tref=self.adcirc_time[0])


    @staticmethod
    def get_stormTide_data(DatCat: hydromt.DataCatalog, dataName: str) -> xr.Dataset:
        stormTide = DatCat.get_geodataset(data_like=dataName)
        indices_to_remove = stormTide.where(abs(stormTide) > 100).dropna(dim='index').coords['index'].values
        print(f'Index of StormTide Stations Removed: {indices_to_remove}')
        stormTide = stormTide.drop_sel(index=indices_to_remove)
        return stormTide

    @staticmethod
    def prepare_gridded_data(DatCat: hydromt.DataCatalog, dataName: str, track_time: tuple) -> xr.Dataset:
        da = DatCat.get_rasterdataset(data_like=dataName, time_tuple=track_time)
        time_pd = pd.to_datetime(da.coords['time'].values)
        freq = pd.infer_freq(time_pd)
        if freq != 'h':
            da = da.resample(time='1H').ffill()
        time_target = pd.date_range(start=track_time[0], end=track_time[1], freq='h')
        da = da.reindex(time=time_target).fillna(0)
        return da

    @staticmethod
    def prepare_track_information(tracks_file: Path, tc_index) -> gpd.GeoDataFrame:
        tc_tracks = sio.loadmat(str(tracks_file))
        df = pd.DataFrame()
        py_index = tc_index - 1  # TC_ID corresponds to row 0 in python
        for v in tc_tracks.keys():
            if v in ['__header__', '__version__', '__globals__', 'freq']:
                continue
            else:
                if len(tc_tracks[v][py_index]) == 1:
                    df[v] = tc_tracks[v][py_index].item()
                else:
                    df[v] = tc_tracks[v][py_index, :]
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

        track_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['lon100'], y=df['lat100'], crs=4326))
        track_gdf = track_gdf.loc[track_gdf['vstore100'] != 0]

        return track_gdf

    @staticmethod
    def get_track_time(track_information: gpd.GeoDataFrame, ) -> tuple:
        track_time = (track_information['datetime'].min(), track_information['datetime'].max())
        return track_time

    @staticmethod
    def get_stormTide_time(stormTide: xr.Dataset, ) -> tuple:
        tmin = pd.to_datetime(stormTide.time.min().values)
        tmax = pd.to_datetime(stormTide.time.max().values)
        return (tmin, tmax)

    @property
    def track_points_to_linestring(self) -> gpd.GeoDataFrame:
        self.track_information['tc_id'] = self.tc_index
        track_line = self.track_information.groupby(['tc_id'])['geometry'].apply(lambda x: LineString(x.tolist()))
        track_line = gpd.GeoDataFrame(track_line, geometry='geometry', crs=4326)
        return track_line

    @staticmethod
    def coastal_boundary_condition_locations(adcirc_reanalysis_locs_filepath: Path,
                                             adcirc_stormTide_locs_filepath: Path) -> list:

        # Mapping the stormTide ADCIRC to the reanalysis extraction locations
        reanalysis_locs = pd.read_csv(str(adcirc_reanalysis_locs_filepath))
        stormTide_locs = pd.read_csv(str(adcirc_stormTide_locs_filepath))
        stormTide_locs['gage_id'] = stormTide_locs['gage_id'].astype(int)

        reanalysis_locs = gpd.GeoDataFrame(reanalysis_locs,
                                           geometry=gpd.points_from_xy(x=reanalysis_locs['LON'],
                                                                       y=reanalysis_locs['LAT'],
                                                                       crs=4326)).to_crs(32617)
        stormTide_locs = gpd.GeoDataFrame(stormTide_locs,
                                          geometry=gpd.points_from_xy(x=stormTide_locs['x'], y=stormTide_locs['y'],
                                                                      crs=4326)).to_crs(32617)
        locs_map = stormTide_locs.sjoin_nearest(reanalysis_locs, how='left', max_distance=1000)
        locs_map['gage_id'] = locs_map['gage_id'].astype(int)

        return locs_map

    @property
    def calculate_stormTide_peaks(self) -> pd.DataFrame:
        peaks = self.stormTide.max(dim=['time']).to_dataframe()
        return peaks

    @staticmethod
    def calculate_reanalysis_tide_offset(mapped_stations: gpd.GeoDataFrame,
                                         stormTide: xr.Dataset,
                                         reanalysis: xr.Dataset,
                                         tref: pd.DatetimeIndex) -> pd.DataFrame:
        print('Calculating high tide offset between ADCIRC StormTide and Reanalysis data...')
        sta_df1 = calculate_station_highTide_time(da=stormTide, tref=tref, station_names=None)
        sta_df1 = pd.concat(objs=[sta_df1, mapped_stations.set_index('gage_id')['Point']], axis=1)

        sta_df2 = calculate_station_highTide_time(da=reanalysis, tref=tref,
                                                  station_names=mapped_stations['Point'].tolist())
        sta_df2 = pd.concat(objs=[sta_df2, mapped_stations.set_index('Point')['gage_id']], axis=1)

        sta_df = pd.concat(objs=[sta_df1, sta_df2.set_index('gage_id')], axis=1)
        sta_df.columns = ['st_ht_dt', 'Point', 're_ht_dt']
        sta_df['offset'] = sta_df['st_ht_dt'] - sta_df['re_ht_dt']

        return sta_df

    @property
    def shift_reanalysis_tides(self) -> xr.Dataset:
        print('Applying temporal shift to Reanalysis data...')
        da = self.reanalysis_data.sel(index=self.coastal_locations_mapped['Point'].tolist())
        for sta in self.tide_offset_table['Point']:
            time_offset = self.tide_offset_table[self.tide_offset_table['Point'] == sta]['offset'].item()
            if time_offset is not pd.NaT:
                if abs(time_offset) > datetime.timedelta(0):
                    data = da.sel(index=sta)
                    df = data.to_dataframe()

                    df_shift = df.copy()
                    df_shift['offset'] = data.time + time_offset
                    df_shift = df_shift[['waterlevel', 'offset']]
                    df_shift.set_index('offset', inplace=True, drop=True)

                    df_out = pd.concat(objs=[df['index'], df_shift], axis=1, ignore_index=False, join='outer')
                    df_out = df_out[df_out.index.isin(df.index)]
                    da.sel(index=sta).values = df_out['waterlevel'].to_numpy()
                else:
                    continue
        return da


NCEP_DataPaths = DataPaths(
    root=Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis'),
    tracks_filepath=Path(r'.\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100.mat'),
    adcirc_stormTide_dir=Path(r'.\stormTide\adcirc_waterlevel_netcdf'),
    wind_dir=Path(r'.\wind\02_CLE15_WindOutput_Gridded'),
    precip_dir=Path(r'.\rain\03_TCR_RainOutput_Gridded_hourly')
)

