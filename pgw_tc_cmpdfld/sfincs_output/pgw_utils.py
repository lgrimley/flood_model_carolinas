import os
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import scipy.ndimage as sndi


def bilinear_downscale(elevs, subgridscale):
    """
    Use scipy's zoom interpolator to interpolate the grid elevations
    onto the subgrid via bilinear interpolation.
    Much faster than regular grid interpolator for our purposes.
    """
    # Use scipy's zoom function for a faster bilinear interpolation from
    # grid to subgrid
    downscaled = sndi.zoom(
        elevs, subgridscale, order=1, output=float, mode='constant',
        cval=np.nan, prefilter=False, grid_mode=False
    )

    return downscaled


def get_ensemble_runIDs(da_zsmax, storm=None, climate=None, scenario=None):
    # Get the list of runs in the da_zsmax so we can subset
    run_ids = da_zsmax.run.values.tolist()
    # Hardcoded, number of WRF ensemble members
    nruns = 7
    if storm == 'matt':
        nruns = 6
    run_ids = [i for i in run_ids if f'SF{(nruns + 1)}' not in i]

    # Subset the run IDs to the storm, climate, scenario we are interested in
    if storm is not None:
        run_ids = [i for i in run_ids if storm in i]
    if climate is not None:
        run_ids = [i for i in run_ids if climate in i]
    if scenario is not None:
        run_ids = [i for i in run_ids if scenario in i]

    return run_ids


def summarize_ensemble(da_zsmax, storm, climate, scenario, ntype='mean'):
    # Hardcoded, number of WRF ensemble members
    nruns = 7
    if storm == 'matt':
        nruns = 6

    # Get the list of runs in the da_zsmax so we can subset
    rids = da_zsmax.run.values.tolist()

    # Subset the run IDs to the storm, climate, scenario we are interested in
    sub = [i for i in rids if f'{storm}_{climate}' in i]
    sub2 = [i for i in sub if f'SF{(nruns + 1)}' not in i]
    selected_ids = [i for i in sub2 if scenario in i]

    # Pull the zsmax for these runs
    sel_zmax = da_zsmax.sel(run=selected_ids)
    if len(sel_zmax.run.values) % nruns != 0:
        print('Expected number of runs were not found...')
        print(selected_ids)
    else:
        if ntype == 'mean':
            da_ensemble = da_zsmax.sel(run=selected_ids).mean('run')
        elif ntype == 'max':
            da_ensemble = da_zsmax.sel(run=selected_ids).max('run')
    da_ensemble = da_ensemble['zsmax']
    da_ensemble.name = f'{storm}_{climate}_{scenario}_{ntype}'

    return da_ensemble, selected_ids


def calc_diff_in_zsmax_compound_minus_max_individual(da_zsmax, compound_key, runoff_key, coastal_key):
    # Outputs a data array of the diff in water level compound minus max. single driver
    # Calculate the max water level at each cell across the coastal and runoff drivers
    da_single_max = da_zsmax.sel(run=[runoff_key, coastal_key]).max('run')

    # Calculate the difference between the max water level of the compound and the max of the individual drivers
    da_diff = (da_zsmax.sel(run=compound_key) - da_single_max).compute()
    da_diff = da_diff['zsmax']
    da_diff.name = 'diff in waterlevel compound minus max. single driver'
    da_diff.attrs.update(unit='m')

    return da_diff


def classify_zsmax_by_process(da_zsmax, compound_key, runoff_key, coastal_key, name_out, hmin):
    # Outputs a data array with the zsmax attributed to processes (codes 0 to 4)
    da_diff = calc_diff_in_zsmax_compound_minus_max_individual(da_zsmax, compound_key, runoff_key, coastal_key)
    # Create masks based on the driver that caused the max water level given a depth threshold hmin
    compound_mask = da_diff > hmin
    coastal_mask = da_zsmax.sel(run=coastal_key).fillna(0) > da_zsmax.sel(run=[runoff_key]).fillna(0).max('run')
    runoff_mask = da_zsmax.sel(run=runoff_key).fillna(0) > da_zsmax.sel(run=[coastal_key]).fillna(0).max('run')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    da_classified = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
                     + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()
    da_classified = da_classified['zsmax']
    da_classified.name = name_out

    # Return compound only locations
    da_compound_extent = xr.where(compound_mask, x=1, y=0)
    da_compound_extent.name = da_classified.name

    return da_diff, da_classified, da_compound_extent


def calculate_flood_area_by_process(da_classified):
    unique_codes, cell_counts = np.unique(da_classified.data, return_counts=True)
    fld_area = cell_counts.copy()
    res = 200  # grid cell resolution in meters
    fld_area = fld_area * (res * res) / (1000 ** 2)  # square km
    fld_area = fld_area.T
    fld_area = pd.DataFrame(fld_area)
    return fld_area


def cleanup_flood_area_dataframe(fld_area_df):
    fld_area_df = fld_area_df.T

    # Combine compound flood runoff and coastal
    fld_area_df['compound'] = fld_area_df['compound_coastal'] + fld_area_df['compound_runoff']

    # Data organization
    fld_area_df.drop(['no_flood', 'compound_coastal', 'compound_runoff'], axis=1, inplace=True)

    # Add a column for total peak flood extent (sum of processes)
    fld_area_df['Total'] = fld_area_df.sum(axis=1)

    # Rename the columns
    fld_area_df.columns = ['Coastal', 'Runoff', 'Compound', 'Total']

    # Add some more information for grouping
    fld_area_df['storm'] = [i.split("_")[0] for i in fld_area_df.index]
    fld_area_df['climate'] = [i.split("_")[1] for i in fld_area_df.index]
    fld_area_df['group'] = fld_area_df['storm'] + ' ' + fld_area_df['climate']

    return fld_area_df

