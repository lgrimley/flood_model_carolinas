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
from hydromt_sfincs import SfincsModel

from sfincs.post_processing.mask_zsmax import dep_sbg
from src.utils import process_tmax_in_hours, calculate_station_highTide_time
from scipy import ndimage
import time


class DataPaths:
    adcirc_reanalysis_locs_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\RENCI_EDSReanalysis\full_data_meta.csv')
    adcirc_stormTide_locs_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.csv')
    adcirc_loc_mapping_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\stormTide\map_Reanalysis_to_stormTideLocs.csv')

    def __init__(
            self: Self,
            root: Path = None,
            tracks_filepath: Path = None,
            adcirc_stormTide_dir: Path = None,
            wind_dir: Path = None,
            precip_dir: Path = None,
            adcirc_reanalysis_data_filepath: Path = None,
    ):
        self.adcirc_reanalysis_data_filepath = adcirc_reanalysis_data_filepath
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
            coastal_locations_mapped: pd.DataFrame = None,
    ):
        self.DataPaths = DataPaths
        self.DatCat = DatCat
        self.tc_index = tc_index
        self.tc_index_padded = str(tc_index).zfill(4)
        self.filename = f'{self.tc_index_padded}.nc'
        self.coastal_locations_mapped = coastal_locations_mapped

        # Read in a geodataframe of the track information from the MAT file
        try:
            self.track_information = self.prepare_track_information(tracks_file=DataPaths.tracks_filepath,
                                                                    tc_index=tc_index)
            self.track_time = self.get_track_time(track_information=self.track_information)
            print('Track information loaded from mat file.')
        except Exception as e:
            print('Issue reading track information from mat file.')
            print(e)
            #breakpoint()

        try:
            self.stormTide = self.DatCat.get_geodataset(data_like=f'stormTide_{self.tc_index_padded}')
            self.adcirc_time = self.get_stormTide_time(stormTide=self.stormTide)
            print('Stormtide data loaded from Hydromt Datacatalog.')
        except Exception as e:
            print('Issue reading stormtide netcdf.')
            print(e)
            #breakpoint()

        tmin = min(self.track_time[0], self.adcirc_time[0])
        tmax = max(self.track_time[1], self.adcirc_time[1]) + pd.to_timedelta('5d')
        self.time_range = (tmin, tmax)
        print(self.time_range)

        # Create a dataframe for mapping the ADCIRC reanalysis and stormtide stations because both are used
        # for creating the coastal water level boundary conditions
        if self.coastal_locations_mapped is None:
            try:
                self.coastal_locations_mapped = self.coastal_boundary_condition_locations(
                    adcirc_reanalysis_locs_filepath=DataPaths.adcirc_reanalysis_locs_filepath,
                    adcirc_stormTide_locs_filepath=DataPaths.adcirc_stormTide_locs_filepath)
                print('Loaded coastal gage locations and create mapping table.')
            except:
                print('Issue loading coastal gage locations and create mapping table.')
                breakpoint()

        # Read in all ADCIRC reanalysis data for the TC time period, buffer 15 days on either end of the storm
        buff_time = (self.time_range[0] - pd.Timedelta(days=15), self.time_range[1] + pd.Timedelta(days=15))
        self.reanalysis_data = self.DatCat.get_geodataset(data_like=f'tide_reanalysis', time_tuple=buff_time)
        print('Load reanalysis data for storm dates with 15 day buffer on either end.')

        if self.DataPaths.wind_dir is not None:
            self.wind = self.prepare_gridded_data(DatCat=DatCat,
                                                  dataName=f'wind_{self.tc_index_padded}', track_time=self.time_range)
            print('Load wind data from the Hydromt Datacatalog.')

        if self.DataPaths.precip_dir is not None:
            self.precip = self.prepare_gridded_data(DatCat=DatCat,
                                                    dataName=f'precip_{self.tc_index_padded}', track_time=self.time_range)
            print('Load precipitation data from the Hydromt Datacatalog.')

        self.stormTide = self.prepare_stormTide_data(stormTide = self.stormTide, track_time=self.time_range)
        print('Stormtide data cleaned.')
        print('Calculating high tide offset between ADCIRC StormTide and Reanalysis data...')
        self.tide_offset_table = self.calculate_reanalysis_tide_offset(mapped_stations=self.coastal_locations_mapped,
                                                                       stormTide=self.stormTide,
                                                                       reanalysis=self.reanalysis_data,
                                                                       tref=self.adcirc_time[0])
        print('Tide offset table created for reanalysis and stormtide gages.')

        self.reanalysis_data_offset = self.shift_reanalysis_tides(mapped_stations=self.coastal_locations_mapped,
                                                                  tide_offset_table=self.tide_offset_table,
                                                                  reanalysis=self.reanalysis_data)
        print('Reanalysis gage data offset to match tides from stormdata.')

        self.merged_waterlevel = self.merge_stormTide_with_shiftedReanalysis(stormTide=self.stormTide,
                                               tide_offset_table=self.tide_offset_table,
                                               reanalysis_data_offset=self.reanalysis_data_offset)

        print('Shifted reanalysis tides appended to cleaned Stormtide data for writing to SFINCS input.')

    @staticmethod
    def prepare_stormTide_data(stormTide: xr.Dataset, track_time: tuple) -> xr.Dataset:
        indices_to_remove = stormTide.where(abs(stormTide) > 100).dropna(dim='index').coords['index'].values
        print(f'Index of StormTide Stations Removed: {indices_to_remove}')
        stormTide = stormTide.drop_sel(index=indices_to_remove)
        time_target = pd.date_range(start=track_time[0], end=track_time[1], freq='h')
        da = stormTide.reindex(time=time_target).fillna(0)
        return da

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

        try:
            sta_df1 = calculate_station_highTide_time(da=stormTide, tref=tref, station_names=None)
            sta_df1 = pd.concat(objs=[sta_df1, mapped_stations.set_index('gage_id')['Point']], axis=1)
            print('High tide time calculated for stormtide gages.')
        except:
            print('Issue calculating station highTide time for stormtide data...')

        try:
            sta_df2 = calculate_station_highTide_time(da=reanalysis, tref=tref,
                                                      station_names=mapped_stations['Point'].tolist())
            sta_df2 = pd.concat(objs=[sta_df2, mapped_stations.set_index('Point')['gage_id']], axis=1)
            print('High tide time calculated for reanalysis gages.')
        except:
            print('Issue calculating station highTide time for reanalysis data...')

        sta_df = pd.concat(objs=[sta_df1, sta_df2.set_index('gage_id')], axis=1)
        sta_df.columns = ['st_ht_dt', 'Point', 're_ht_dt']
        sta_df['offset'] = sta_df['st_ht_dt'] - sta_df['re_ht_dt']
        sta_df.sort_index(axis=0, ascending=True, inplace=True)
        sta_df['offset_fill'] = sta_df['offset'].interpolate(method='pad')
        return sta_df

    @staticmethod
    def shift_reanalysis_tides(mapped_stations: gpd.GeoDataFrame,
                               tide_offset_table: pd.DataFrame,
                               reanalysis: xr.Dataset) -> xr.Dataset:
        print('Applying temporal shift to Reanalysis data...')
        da = reanalysis.sel(index=mapped_stations['Point'].tolist())

        mdf = pd.DataFrame()
        mdf['time'] = da.time.values
        mdf.set_index('time', inplace=True, drop=True)
        for sta in tide_offset_table['Point']:
            time_offset = tide_offset_table[tide_offset_table['Point'] == sta]['offset_fill'].item()
            data = da.sel(index=sta)
            df = data.to_dataframe()
            df_out = df
            if time_offset is not pd.NaT:
                if abs(time_offset) > datetime.timedelta(0):
                    df_shift = df.copy()
                    df_shift['offset'] = data.time + time_offset
                    df_shift = df_shift[['waterlevel', 'offset']]
                    df_shift.set_index('offset', inplace=True, drop=True)

                    df_out = pd.concat(objs=[df['index'], df_shift], axis=1, ignore_index=False, join='outer')
                    df_out = df_out[df_out.index.isin(df.index)]

            mdf = pd.concat(objs=[mdf, df_out['waterlevel']], axis=1, ignore_index=False)
        mdf.columns = mapped_stations['Point'].tolist()
        da.data = mdf
        return da

    @staticmethod
    def merge_stormTide_with_shiftedReanalysis(stormTide,
                                               tide_offset_table,
                                               reanalysis_data_offset) -> xr.Dataset:
        mdf = pd.DataFrame()
        mdf['time'] = stormTide.time
        mdf.set_index('time', inplace=True, drop=True)
        for sta in stormTide.index.values:
            # Load the station storm tide data
            wl = stormTide.sel(index=sta).to_dataframe()
            wl['delta_waterlevel'] = wl['waterlevel'].diff()  # Calculate the difference in water levels across time
            notrend_time = wl[wl['delta_waterlevel'] == 0].idxmin()  # Identify where water level is increasing

            # Get the time when model output zeroed
            t1 = notrend_time[0] - pd.Timedelta('1h')
            t2 = wl.index[-1]

            # Pull the shifted reanalysis data for this gage
            tide_gage = tide_offset_table[tide_offset_table.index == sta]['Point'].item()
            tide_data = reanalysis_data_offset.sel(index=tide_gage).to_dataframe()[['waterlevel']]
            tide_data.columns = ['tides']
            tide_subset = tide_data[tide_data.index >= t1]

            # Replace the zero stormTide data with the shifted, reanalysis tides
            new_ts = pd.concat([wl, tide_subset], axis=1)
            new_ts['waterlevel'][new_ts.index >= t1] = new_ts['tides'][new_ts.index >= t1]
            new_ts = new_ts[new_ts.index <= t2]
            new_ts = new_ts[['waterlevel']]

            mdf = pd.concat(objs=[mdf,new_ts], axis=1, ignore_index=False)
        mdf.columns = stormTide.index.values
        stormTide.data = mdf
        return stormTide


# need to remove masking function, possible add downscaling
class TCFloodHazard_old:
    def __init__(
            self: Self,
            tc_root: Path = None,
            tracks: str = None,
            tc_index: int = None,
            sfincs_mod: SfincsModel = None,
            zsmax_threshold: float = None,
    ):
        self.tc_root = tc_root
        self.tracks = tracks
        self.tc_index = tc_index
        self.sfincs_mod = sfincs_mod
        self.tc_name = f'TC_{str(tc_index).zfill(4)}'
        self.input_dir = Path(os.path.join(self.tc_root, 'sfincs_bc_inputs'))
        self.output_dir = Path(os.path.join(self.tc_root, 'sfincs_outputs'))
        self.zsmax_threshold = zsmax_threshold
        self.watermask = sfincs_mod.data_catalog.get_rasterdataset(
            r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\waterbody_mask.nc')

        try:
            self.sfincs_results = self.get_sfincs_tc_results(sfincs_mod=self.sfincs_mod, tc_dir = self.tc_root)
        except Exception as e:
            print(e)
            breakpoint()

        # Get the maximum outputs for the variables of interest
        self.varmax = self.process_maximum_outputs(tc_index=self.tc_index, tc_name=self.tc_name, tracks=self.tracks,
                                                    sfincs_results=self.sfincs_results, watermask=self.watermask)

        # Get a combined data array of the water level max for each scenario and the difference in
        # maximum water levels between compound and max individual
        self.da_zsmax, self.zsmax_diff = self.calc_diff_in_zsmax_compound_minus_max_individual(da_dict=self.varmax)

        # Attribute the peak water level in each cell to runoff, coastal, or compound processes using hmin threshold
        self.da_classified = self.classify_zsmax_by_process(da=self.da_zsmax, da_diff=self.zsmax_diff,
                                                            hmin = self.zsmax_threshold)

        self.da_extents = self.get_binary_flood_extents(da_classified=self.da_classified)
        #self.flooded_area_table = self.calculate_flooded_area_by_process(da=self.da_classified, tc_index=self.tc_index)
        self.precip, self.wind = self.process_bc_inputs(input_dir=self.input_dir)

    @staticmethod
    def get_sfincs_tc_results(sfincs_mod: SfincsModel, tc_dir: Path) -> dict:
        results = {}
        scenarios = ['compound', 'runoff', 'coastal']
        for scenario in scenarios:
            # Read the SFINCS model config file and outputs
            sfincs_mod.read_config(os.path.join(tc_dir, scenario, 'sfincs.inp'))
            sfincs_mod.read_results(
                fn_map=os.path.join(tc_dir, scenario, 'sfincs_map.nc'),
                fn_his=os.path.join(tc_dir, scenario, 'sfincs_his.nc'),
                decode_times=False)
            res = sfincs_mod.results.copy()
            results[scenario] = res
        return results

    @staticmethod
    def process_maximum_outputs(tc_index: int, tc_name: str, tracks: str,
                                sfincs_results: dict, watermask: xr.DataArray) -> dict:
        variables = ['zsmax', 'vmax', 'tmax']
        da_dict = {}
        for scenario in sfincs_results.keys():
            ds = sfincs_results[scenario]
            scen_dict = {}
            for var in variables:
                da = ds[var].max(dim='timemax')
                da = da.where(watermask == 0.0)
                if var == 'tmax':
                    da = process_tmax_in_hours(tmax = da, time_min_hr = 0)
                da = da.assign_attrs(variable = var, scenario=scenario,
                                     tc_index=tc_index, tc_name=tc_name, tracks=tracks)
                scen_dict[var] = da
            da_dict[scenario] = scen_dict
        return da_dict

    @staticmethod
    def calc_diff_in_zsmax_compound_minus_max_individual(da_dict: dict) -> [xr.DataArray, xr.DataArray]:
        da = xr.concat(objs=[da_dict['compound']['zsmax'],
                             da_dict['runoff']['zsmax'],
                             da_dict['coastal']['zsmax']],
                       dim='scenario')
        da['scenario'] = xr.IndexVariable(dims='scenario', data=['compound', 'runoff','coastal'])

        # Outputs a data array of the diff in water level compound minus max. single driver
        # Calculate the max water level at each cell across the coastal and runoff drivers
        da_single_max = da.sel(scenario=['runoff', 'coastal']).max('scenario')

        # Calculate the difference between the max water level of the compound and the max of the individual drivers
        da_diff = (da.sel(scenario='compound') - da_single_max).compute()
        da_diff.name = 'diff in waterlevel compound minus max. single driver'
        da_diff.attrs = da_dict['compound']['zsmax'].attrs

        return da, da_diff

    @staticmethod
    def classify_zsmax_by_process(da: xr.DataArray, da_diff: xr.DataArray, hmin: float=0.1) -> xr.DataArray:
        # Outputs a data array with the zsmax attributed to processes (codes 0 to 4)
        # Create masks based on the driver that caused the max water level given a depth threshold hmin
        compound_mask = da_diff > hmin
        coastal_mask = da.sel(scenario='coastal').fillna(0) > da.sel(scenario=['runoff']).fillna(0).max('scenario')
        runoff_mask = da.sel(scenario='runoff').fillna(0) > da.sel(scenario=['coastal']).fillna(0).max('scenario')
        assert ~np.logical_and(runoff_mask, coastal_mask).any()
        da_classified = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
                         + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()

        da_classified.name = 'zsmax_attribution'
        da_classified = da_classified.assign_attrs(hmin=hmin,
                                                   no_class=0, coast_class=1, coast_compound_class=2,
                                                   runoff_class=3, runoff_compound_class=4)
        return da_classified

    @staticmethod
    def get_binary_flood_extents(da_classified: xr.DataArray) -> xr.DataArray:
        # Compound
        mask = ((da_classified == 2) | (da_classified == 4))
        compound_extent = xr.where(da_classified.where(mask), x=1, y=0)
        compound_extent.name = 'compound_flood_extent'

        # Runoff
        mask = (da_classified == 3)
        runoff_extent = xr.where(da_classified.where(mask), x=1, y=0)
        runoff_extent.name = 'runoff_flood_extent'

        # Coastal
        mask = (da_classified == 1)
        coastal_extent = xr.where(da_classified.where(mask), x=1, y=0)
        coastal_extent.name = 'coastal_flood_extent'

        # Combined
        mask = (da_classified == 1)
        total_extent = xr.where(da_classified > 0, x=1, y=0)
        total_extent.name = 'total_flood_extent'

        da = xr.concat(objs=[compound_extent, runoff_extent, coastal_extent, total_extent], dim='scenario')
        da['scenario'] = xr.IndexVariable(dims='scenario', data=['compound', 'runoff','coastal', 'total'])

        return da

    @staticmethod
    def process_bc_inputs(input_dir: Path=None) -> [xr.DataArray, xr.DataArray]:

        bc = 'precip_2d'
        data = xr.open_dataset(os.path.join(input_dir, f'{bc}.nc'))
        total = data.sum(dim='time')['Precipitation']
        mean_rate = data.mean(dim='time')['Precipitation']
        max_rate = data.max(dim='time')['Precipitation']
        da_ls = [total, mean_rate, max_rate]
        da_precip = xr.concat(objs=da_ls, dim='name')
        da_precip['name'] = xr.IndexVariable(dims='name', data=['total', 'mean', 'max'])

        bc = 'wind_2d'
        data = xr.open_dataset(os.path.join(input_dir, f'{bc}.nc'))
        windspeed = np.sqrt((data['eastward_wind']**2) + (data['northward_wind']**2))
        windspeed.name = 'windspeed'
        mean_rate = windspeed.mean(dim='time')
        max_rate = windspeed.max(dim='time')
        da_ls = [mean_rate, max_rate]
        da_wind = xr.concat(objs=da_ls, dim='name')
        da_wind['name'] = xr.IndexVariable(dims='name', data=['mean', 'max'])

        return [da_precip, da_wind]



NCEP_DataPaths = DataPaths(
    root=Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis'),
    tracks_filepath=Path(r'.\tracks\UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100.mat'),
    adcirc_stormTide_dir=Path(r'.\stormTide\adcirc_waterlevel_netcdf'),
    wind_dir=Path(r'.\wind\02_CLE15_WindOutput_Gridded'),
    precip_dir=Path(r'.\rain\03_TCR_RainOutput_Gridded_hourly'),
    adcirc_reanalysis_data_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\RENCI_EDSReanalysis\EDSReanalysis_V2_1992_2022.nc')
)

cansem_ssp585_DataPaths = DataPaths(
    root=Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585'),
    tracks_filepath=Path(r'.\tracks\UScoast6_AL_canesm_ssp585cal_roEst1rmEst1_trk100.mat'),
    adcirc_stormTide_dir=Path(r'.\stormTide\adcirc_waterlevel_netcdf_canesm_ssp585'),
    wind_dir=Path(r'.\wind\CLE15_ReGridded_canesm_ssp585cal'),
    precip_dir=Path(r'.\rain\TCR_Gridded_canesm_ssp585cal_hourly'),
    adcirc_reanalysis_data_filepath = Path(
        r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\RENCI_EDSReanalysis\EDSReanalysis_V2_1992_2022_with90yrOffset.nc')
)


def get_binary_flood_extents(da_classified: xr.DataArray) -> xr.Dataset:
    # Compound
    mask = ((da_classified == 2) | (da_classified == 4))
    compound_extent = xr.where(da_classified.where(mask), x=1, y=0)
    compound_extent.name = 'flood_extent'

    # Runoff
    mask = (da_classified == 3)
    runoff_extent = xr.where(da_classified.where(mask), x=1, y=0)
    runoff_extent.name = 'flood_extent'

    # Coastal
    mask = (da_classified == 1)
    coastal_extent = xr.where(da_classified.where(mask), x=1, y=0)
    coastal_extent.name = 'flood_extent'

    # Combined
    mask = (da_classified == 1)
    total_extent = xr.where(da_classified > 0, x=1, y=0)
    total_extent.name = 'flood_extent'

    da_out = xr.concat(objs=[compound_extent, runoff_extent, coastal_extent, total_extent], dim='scenario')
    da_out['scenario'] = xr.IndexVariable(dims='scenario', data=['compound', 'runoff','coastal', 'total'])
    da_out = da_out.astype(int)

    return da_out


def process_bc_inputs(input_dir: Path=None) -> [xr.DataArray, xr.DataArray]:
    bc = 'precip_2d'
    data = xr.open_dataset(os.path.join(input_dir, f'{bc}.nc'))
    data = data.astype('float32')
    total = data.sum(dim='time')['Precipitation']
    mean_rate = data.mean(dim='time')['Precipitation']
    max_rate = data.max(dim='time')['Precipitation']
    da_ls = [total, mean_rate, max_rate]
    da_precip = xr.concat(objs=da_ls, dim='name')
    da_precip['name'] = xr.IndexVariable(dims='name', data=['total', 'mean', 'max'])

    bc = 'wind_2d'
    data = xr.open_dataset(os.path.join(input_dir, f'{bc}.nc'))
    data = data.astype('float32')
    windspeed = np.sqrt((data['eastward_wind']**2) + (data['northward_wind']**2))
    windspeed.name = 'windspeed'
    mean_rate = windspeed.mean(dim='time')
    max_rate = windspeed.max(dim='time')
    da_ls = [mean_rate, max_rate]
    da_wind = xr.concat(objs=da_ls, dim='name')
    da_wind['name'] = xr.IndexVariable(dims='name', data=['mean', 'max'])

    return [da_precip, da_wind]


def resized_gridded_output(ds: xr.Dataset, elevation_da: xr.DataArray, variables=None) -> xr.Dataset:
        if variables is None:
            variables = ['zsmax']

        start_time = time.time()
        target_shape = elevation_da.shape
        scenarios = ds.scenario.values

        rdas_dict = {}
        for var in variables:
            out_varname = f'resized_{var}'
            rdas = []
            for scen in scenarios:
                da = ds.sel(scenario=scen)[var]
                scaling_factors = [target_shape[i] / da.shape[i] for i in range(len(da.shape))]
                ra = ndimage.zoom(input=da, zoom=scaling_factors, order=1,
                                             output='float32', mode='grid-constant',
                                             cval=np.nan, prefilter=False, grid_mode=True)
                rda = xr.DataArray(ra,
                                   dims=da.dims,
                                   coords={dim: np.linspace(da.coords[dim].min(), da.coords[dim].max(),
                                                               target_shape[i]) for i, dim in enumerate(da.dims)},
                                   attrs=da.attrs)
                rda['spatial_ref'] = da['spatial_ref']
                rdas.append(rda)

            x = xr.concat(objs=rdas, dim='scenario')
            x['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)
            rdas_dict[out_varname] = x

        ds_resized = xr.Dataset(rdas_dict)
        ds_resized.attrs = ds.attrs

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        return ds_resized


class TCFloodHazard:
    def __init__(
            self: Self,
            tc_root: Path = None,
            tracks: str = None,
            tc_index: int = None,
            sfincs_mod: SfincsModel = None,
            zsmax_threshold: float = None,
    ):
        self.tc_root = tc_root
        self.tracks = tracks
        self.tc_index = tc_index
        self.sfincs_mod = sfincs_mod
        self.tc_name = f'TC_{str(tc_index).zfill(4)}'
        self.input_dir = Path(os.path.join(self.tc_root, 'sfincs_bc_inputs'))
        self.zsmax_threshold = zsmax_threshold
        self.elevation_da = self.sfincs_mod.grid['dep']

        try:
            self.sfincs_results = self.get_sfincs_tc_results(sfincs_mod=self.sfincs_mod, tc_dir = self.tc_root)
        except Exception as e:
            print(e)
            breakpoint()

        # Get the maximum outputs for the variables of interest
        self.scenario_results = self.process_maximum_map_outputs(tc_index=self.tc_index,
                                                                 tc_name=self.tc_name,
                                                                 tracks=self.tracks,
                                                                 sfincs_results=self.sfincs_results)

        self.scenario_results = self.calc_maximum_depth(ds =self.scenario_results,
                                                        zs_var= 'zsmax',
                                                        elevation_das=[self.elevation_da, self.sfincs_mod.subgrid['z_zmin']],
                                                        output_names=['hmax', 'hmax_z_zmin'])

        # Attribute the peak water level in each cell to runoff, coastal, or compound processes using hmin threshold
        da_diff = self.calc_diff_in_zsmax_compound_minus_max_individual(da=self.scenario_results['zsmax'])
        self.attribution_results = self.classify_zsmax_by_process(da=self.scenario_results['zsmax'],
                                                                  da_diff=da_diff,
                                                                  hmin = self.zsmax_threshold)

    @staticmethod
    def get_sfincs_tc_results(sfincs_mod: SfincsModel, tc_dir: Path) -> dict:
        results = {}
        scenarios = ['compound', 'runoff', 'coastal']
        for scenario in scenarios:
            # Read the SFINCS model config file and outputs
            sfincs_mod.read_config(os.path.join(tc_dir, scenario, 'sfincs.inp'))
            sfincs_mod.read_results(
                fn_map=os.path.join(tc_dir, scenario, 'sfincs_map.nc'),
                fn_his=os.path.join(tc_dir, scenario, 'sfincs_his.nc'),
                decode_times=False)
            res = sfincs_mod.results.copy()
            results[scenario] = res
        return results

    @staticmethod
    def process_maximum_map_outputs(tc_index: int, tc_name: str, tracks: str, sfincs_results: dict) -> xr.Dataset:
        da_dict = {}
        variables = ['zsmax', 'vmax', 'tmax']
        scenarios = ['compound', 'runoff', 'coastal']
        for var in variables:
            da_list = []
            for scen in scenarios:
                da = sfincs_results[scen][var].max(dim='timemax')
                if var == 'zsmax':
                    da.name = 'max_wse'
                    da.assign_attrs(units='m+NAVD88')
                elif var == 'vmax':
                    da.name = 'max_velocity'
                    da.assign_attrs(units='m3/s')
                elif var == 'tmax':
                    da = process_tmax_in_hours(tmax=da, time_min_hr=0)
                    da.name = 'time_of_inundation'
                    da.assign_attrs(units='hours', threshold='0.1 meters')
                da = da.astype('float32')
                da_list.append(da)

            x = xr.concat(objs=da_list, dim='scenario')
            x['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)
            da_dict[var] = x

        ds = xr.Dataset({'zsmax': da_dict['zsmax'],
                         'vmax': da_dict['vmax'],
                         'tmax': da_dict['tmax']})
        ds = ds.assign_attrs(tc_index=tc_index, tc_name=tc_name, tracks=tracks)

        return ds

    @staticmethod
    def calc_maximum_depth(ds: xr.Dataset, zs_var: str, elevation_das: list, output_names: list) -> xr.Dataset:
        da_zsmax = ds[zs_var]
        hmax_dict = {}
        for i in range(len(elevation_das)):
            dep = elevation_das[i]
            hmax = da_zsmax - dep
            hmax.name = 'hmax'
            hmax.assign_attrs(units='m', description=f'{zs_var} relative to {output_names[i]}')
            hmax_dict[output_names[i]] = hmax
        ds_hmax = xr.Dataset(hmax_dict)
        ds_out = xr.merge([ds, ds_hmax])
        ds_out.attrs = ds.attrs

        return ds_out

    @staticmethod
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

    @staticmethod
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

