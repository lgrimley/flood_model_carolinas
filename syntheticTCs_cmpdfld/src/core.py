import sys
import os
import datetime
import pandas as pd
import geopandas as gpd
import xarray as xr
import scipy.io as sio
import numpy as np
import pathlib
from shapely.geometry import LineString
from pathlib import Path
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld\src')

class DataPaths:
    adcirc_reanalysis_locs_filepath = Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\EDSReanalysis_data\full_data_meta.csv')
    adcirc_reanalysis_data_filepath = Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\EDSReanalysis_data\EDSReanalysis_V2_1992_2022.nc')
    adcirc_stormTide_locs_filepath = Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')
    adcirc_loc_mapping_filepath = Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\stormTide\map_Reanalysis_to_stormTideLocs.csv')

    def __init__(self, root: pathlib.Path = None, tracks_filepath: pathlib.Path = None,
                 adcirc_stormTide_dir: pathlib.Path = None,
                 wind_dir: pathlib.Path = None, precip_dir: pathlib.Path = None):

        self.root = root
        os.chdir(self.root)
        self.tracks_filepath = tracks_filepath
        self.adcirc_stormTide_dir = adcirc_stormTide_dir
        self.wind_dir = wind_dir
        self.precip_dir = precip_dir


class SyntheticTrack:
    def __init__(self,
                 DataPaths,
                 tc_id: int = None
                 ):
        self.tc_id = tc_id
        self.DataPaths = DataPaths
        os.chdir(self.DataPaths.root)

        # Calculated
        self.track_gdf = self.get_track_info_to_df()
        self.reanalysis_locs, self.stormTide_locs = self.get_adcirc_locs_as_gdfs()
        match = self.stormTide_locs.sjoin_nearest(self.reanalysis_locs, how='left', max_distance=1000)
        match['gage_id'] = match['gage_id'].astype(int)
        self.adcirc_loc_mapping_gdf = match
        self.filename = f'{str(tc_id).zfill(4)}.nc'

        # If a filepath is specified, read in the data
        if self.DataPaths.adcirc_stormTide_dir is not None:
            self.stormTide_data = xr.open_dataset(os.path.join(self.DataPaths.adcirc_stormTide_dir, self.filename), engine='netcdf4')
            self.adcirc_time = self.get_adcirc_time()
            if self.DataPaths.adcirc_reanalysis_data_filepath is not None:
                self.reanalysis_data=self.subset_reanalysis_data()
        if self.DataPaths.wind_dir is not None:
            self.wind_data = xr.open_dataset(os.path.join(self.DataPaths.wind_dir, self.filename), engine='netcdf4')
        if self.DataPaths.precip_dir is not None:
            self.precip_data = xr.open_dataset(os.path.join(self.DataPaths.precip_dir, self.filename), engine='netcdf4')


    def get_track_info_to_df(self) -> gpd.GeoDataFrame:
        tc_tracks = sio.loadmat(self.DataPaths.tracks_filepath)

        df = pd.DataFrame()
        py_index = self.tc_id - 1  # TC_ID corresponds to row 0 in python
        for v in tc_tracks.keys():
            if v in ['__header__', '__version__', '__globals__', 'freq']:
                continue
            else:
                if len(tc_tracks[v][py_index]) == 1:
                    df[v] = tc_tracks[v][py_index].item()
                else:
                    df[v] = tc_tracks[v][py_index, :]

        def get_track_datetime(df) -> gpd.GeoDataFrame:
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

        df = get_track_datetime(df=df)
        track_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['lon100'], y=df['lat100'], crs=4326))
        self.track_gdf = track_gdf.loc[track_gdf['vstore100'] != 0]

        return self.track_gdf

    @property
    def track_points_to_linestring(self) -> gpd.GeoDataFrame:
        self.track_gdf['tc_id'] = self.tc_id
        track_line = self.track_gdf.groupby(['tc_id'])['geometry'].apply(lambda x: LineString(x.tolist()))
        track_line = gpd.GeoDataFrame(track_line, geometry='geometry', crs=4326)
        return track_line

    def get_adcirc_locs_as_gdfs(self) -> list:
        # Mapping the stormTide ADCIRC to the reanalysis extraction locations
        reanalysis_locs = pd.read_csv(self.DataPaths.adcirc_reanalysis_locs_filepath)
        stormTide_locs = pd.read_csv(self.DataPaths.adcirc_stormTide_locs_filepath)
        stormTide_locs['gage_id'] = stormTide_locs['gage_id'].astype(int)

        reanalysis_locs = gpd.GeoDataFrame(reanalysis_locs,
                                           geometry=gpd.points_from_xy(x=reanalysis_locs['LON'],y=reanalysis_locs['LAT'],
                                                                       crs=4326))
        stormTide_locs = gpd.GeoDataFrame(stormTide_locs,
                                           geometry=gpd.points_from_xy(x=stormTide_locs['x'],y=stormTide_locs['y'],
                                                                       crs=4326))

        return [reanalysis_locs, stormTide_locs]


    def get_adcirc_time(self) -> tuple:
        """Return (tstart, tstop) tuple for the ADCIRC stormTide data"""
        tstart = pd.to_datetime(self.stormTide_data.time.min().values)
        tstop = pd.to_datetime(self.stormTide_data.time.max().values)
        return tstart, tstop


    def subset_reanalysis_data(self) -> xr.Dataset:
        re_da = xr.open_dataset(self.DataPaths.adcirc_reanalysis_data_filepath, engine='netcdf4')
        tstart_buff = self.adcirc_time[0] - datetime.timedelta(hours=36)
        tend_buff = self.adcirc_time[1] + datetime.timedelta(hours=36)
        reanalysis_data = re_da.sel(time=slice(tstart_buff, tend_buff))
        return reanalysis_data


NCEP_DataPaths = DataPaths(
    root = Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis'),
    tracks_filepath = Path(r'.\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100.mat'),
    adcirc_stormTide_dir = Path(r'.\stormTide\adcirc_waterlevel_netcdf'),
    wind_dir = Path(r'.\wind\02_CLE15_WindOutput_Gridded'),
    precip_dir = Path(r'.\rain\03_TCR_RainOutput_Gridded_hourly')
)

track = SyntheticTrack(tc_id=2645, DataPaths=NCEP_DataPaths)







