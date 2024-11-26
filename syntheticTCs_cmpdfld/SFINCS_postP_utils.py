import os
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils


def calc_diff_in_zsmax_compound_minus_max_individual(da_zsmax, compound_key, runoff_key, coastal_key):
    # Outputs a data array of the diff in water level compound minus max. single driver
    # Calculate the max water level at each cell across the coastal and runoff drivers
    da_single_max = da_zsmax.sel(run=[runoff_key, coastal_key]).max('run')

    # Calculate the difference between the max water level of the compound and the max of the individual drivers
    da_diff = (da_zsmax.sel(run=compound_key) - da_single_max).compute()
    if 'run' in da_diff.coords.keys():
        da_diff = da_diff.drop_vars('run')

    da_diff.name = 'diff in waterlevel compound minus max. single driver'
    da_diff.attrs = da_zsmax.attrs

    return da_diff


def classify_zsmax_by_process(da_zsmax, compound_key, runoff_key, coastal_key, hmin):

    da_diff = calc_diff_in_zsmax_compound_minus_max_individual(da_zsmax, compound_key, runoff_key, coastal_key)

    # Outputs a data array with the zsmax attributed to processes (codes 0 to 4)
    # Create masks based on the driver that caused the max water level given a depth threshold hmin
    compound_mask = da_diff > hmin
    coastal_mask = da_zsmax.sel(run=coastal_key).fillna(0) > da_zsmax.sel(run=[runoff_key]).fillna(0).max('run')
    runoff_mask = da_zsmax.sel(run=runoff_key).fillna(0) > da_zsmax.sel(run=[coastal_key]).fillna(0).max('run')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    da_classified = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
                     + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()
    da_classified.name = 'zsmax classification'
    da_classified = da_classified.assign_attrs(tc_id=da_zsmax.attrs['tc_id'], hmin=hmin,
                                               no_class = 0, coast_class = 1, coast_compound_class=2,
                                               runoff_class = 3, runoff_compound_class = 4)

    # Return compound only locations
    da_compound_extent = xr.where(compound_mask, x=1, y=0)
    da_compound_extent.name = 'compound extent'
    da_compound_extent = da_compound_extent.assign_attrs(tc_id=da_zsmax.attrs['tc_id'], hmin=hmin)

    return da_diff, da_classified, da_compound_extent