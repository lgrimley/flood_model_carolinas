#!/usr/bin/env python
# coding: utf-8
import os
import re
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt
import xarray as xr
import matplotlib as mpl
import hydromt
from hydromt import DataCatalog
import hydromt_sfincs
from hydromt_sfincs import SfincsModel
mpl.use('TkAgg')
plt.ion()

'''' Load in the Building data '''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure\return_period_exposure')
#building_df = pd.read_csv(r'..\buildings_tc_exposure_rp_real.csv', index_col=0, low_memory=True)
building_geom = pd.read_csv(r'..\buildings_within_domain.csv', index_col=0, low_memory=False)

rps = [10, 25, 100, 250]
for T in rps:
    '''' Load in the Historic data '''
    # Load the water level data for the historical return periods
    mod_output_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep\aep'
    rp_zs_da = xr.open_dataarray(os.path.join(mod_output_dir, 'ncep_MaxWL_returnPeriods_compound.nc'))
    rp_attr_ds = xr.open_dataset(os.path.join(mod_output_dir, 'ncep_RP_attribution.nc'))
    rp_zs = rp_zs_da.sel(return_period=T)
    rp_attr = rp_attr_ds.sel(return_period=T)['zsmax_attr']
    rp_diff = rp_attr_ds.sel(return_period=T)['zsmax_diff']

    # Extract max water level
    zs = rp_zs.sel(x=building_geom['xcoords'].to_xarray(), y=building_geom['ycoords'].to_xarray(), method='nearest').values
    building_geom[f'hist_rp{T}_zsmax'] = zs.transpose()
    # Extract class
    attr = rp_attr.sel(x=building_geom['xcoords'].to_xarray(), y=building_geom['ycoords'].to_xarray(), method='nearest').values
    building_geom[f'hist_rp{T}_class'] = attr.transpose()
    # Extract diff between compound and max individual
    diff = rp_diff.sel(x=building_geom['xcoords'].to_xarray(), y=building_geom['ycoords'].to_xarray(), method='nearest').values
    building_geom[f'hist_rp{T}_zsDiff'] = diff.transpose()

    '''' Load in the Projected Future data '''
    mod_output_dir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\canesm_ssp585\aep'
    rp_zs_da = xr.open_dataarray(os.path.join(mod_output_dir, 'projected_MaxWL_returnPeriods_compound.nc'))
    rp_attr_ds = xr.open_dataset(os.path.join(mod_output_dir, 'gcm_RP_attribution.nc'))
    rp_zs = rp_zs_da.sel(return_period=T)
    rp_attr = rp_attr_ds.sel(return_period=T)['zsmax_attr']
    rp_diff = rp_attr_ds.sel(return_period=T)['zsmax_diff']

    zs = rp_zs.sel(x=building_geom['xcoords'].to_xarray(), y=building_geom['ycoords'].to_xarray(), method='nearest').values
    building_geom[f'fut_rp{T}_zsmax'] = zs.transpose()

    attr = rp_attr.sel(x=building_geom['xcoords'].to_xarray(), y=building_geom['ycoords'].to_xarray(), method='nearest').values
    building_geom[f'fut_rp{T}_class'] = attr.transpose()

    diff = rp_diff.sel(x=building_geom['xcoords'].to_xarray(), y=building_geom['ycoords'].to_xarray(), method='nearest').values
    building_geom[f'fut_rp{T}_zsDiff'] = diff.transpose()

    # Calculating depths
    building_geom[f'hist_rp{T}_depth'] = building_geom[f'hist_rp{T}_zsmax'] - building_geom['gnd_elev']
    building_geom[f'fut_rp{T}_depth'] = building_geom[f'fut_rp{T}_zsmax'] - building_geom['gnd_elev']
    building_geom[f'rp{T}_depth_diff'] = building_geom[f'fut_rp{T}_depth'] - building_geom[f'hist_rp{T}_depth']

    print(f'Done with {T}')

building_geom.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure\buildings_rps_exposure.csv')
