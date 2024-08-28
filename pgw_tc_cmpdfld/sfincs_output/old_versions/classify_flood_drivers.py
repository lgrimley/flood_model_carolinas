import os
import glob
import hydromt
from hydromt import DataCatalog
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
from hydromt_sfincs import SfincsModel, utils


# This script classifies the driver of peak water levels for each PGW run as coastal, runoff, compound.
# It outputs a netcdf with the driver classification, compound areas, and a CSV with peak flood extent total areas


def classify_zsmax_by_driver(da, compound_key, runoff_key, coastal_key, name_out, hmin):
    # Calculate the max water level at each cell across the coastal and runoff drivers
    da_single_max = da.sel(run=[runoff_key, coastal_key]).max('run')
    # Calculate the difference between the max water level of the compound and the max of the individual drivers
    da_diff = (da.sel(run=compound_key) - da_single_max).compute()
    da_diff.name = 'diff in waterlevel compound minus max. single driver'
    da_diff.attrs.update(unit='m')

    # Create masks based on the driver that caused the max water level given a depth threshold hmin
    compound_mask = da_diff > hmin
    coastal_mask = da.sel(run=coastal_key).fillna(0) > da.sel(run=[runoff_key]).fillna(0).max('run')
    runoff_mask = da.sel(run=runoff_key).fillna(0) > da.sel(run=[coastal_key]).fillna(0).max('run')
    assert ~np.logical_and(runoff_mask, coastal_mask).any()
    da_classified = (xr.where(coastal_mask, x=compound_mask + 1, y=0)
                     + xr.where(runoff_mask, x=compound_mask + 3, y=0)).compute()
    da_classified.name = name_out

    # Calculate the number of cells that are attributed to the different drivers
    unique_codes, fld_area_by_driver = np.unique(da_classified.data, return_counts=True)

    # Return compound only locations
    da_compound = xr.where(compound_mask, x=1, y=0)
    da_compound.name = name_out

    return da_classified, fld_area_by_driver, da_compound, da_diff


work_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis'
out_dir = os.path.join(work_dir, 'driver_analysis')
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)
os.chdir(out_dir)

storms = ['flor', 'floy', 'matt']
climates = ['pres', 'presScaled']
hmin = 0.05  # minimum difference between the individual and compound drivers

''' Parse the drivers for each simulation '''
# Load the peak water levels for all simulations
zsmax_file = os.path.join(work_dir, 'zsmax', 'pgw_zsmax.nc')
da_zsmax = xr.open_dataarray(zsmax_file)

if os.path.exists('pgw_drivers_classified_all.nc') is False:
    fld_cells = pd.DataFrame()  # dataframe populated with total flooded area
    fld_da_compound = []  # populated with data arrays of the compound areas for each run
    fld_da_classified = []
    fld_da_diff = []
    cc_run_ids = []
    run_ids = []  # keep track of the run IDs and their order
    # Loop through and classify flooding by driver
    for storm in storms:
        for climate in climates:
            # Create a list of runs to loop through
            nruns = 8  # Florence and Floyd have 7 ensemble members
            if storm == 'matt':
                nruns = 7  # Matthew only has 6 ensemble members
            # Add the ensemble mean to the list of runs
            runs = [f'ens{i}' for i in np.arange(1, nruns, 1)] + ['ensmean']
            for run in runs:
                name_out = f'{storm}_{climate}_{run}'
                if climate == 'presScaled':
                    compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_{run}_compound',
                                                             f'{storm}_{climate}_{run}_runoff',
                                                             f'{storm}_pres_{run}_coastal']
                    cc_run_ids.append(name_out)
                elif climate == 'pres':
                    compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_{run}_compound',
                                                             f'{storm}_{climate}_{run}_runoff',
                                                             f'{storm}_{climate}_{run}_coastal']
                out = classify_zsmax_by_driver(da=da_zsmax,
                                               compound_key=compound_key, runoff_key=runoff_key,
                                               coastal_key=coastal_key, name_out=name_out, hmin=hmin)

                da_classified, fld_cells_by_driver, da_compound, da_diff = out
                fld_da_classified.append(da_classified)
                fld_cells[f'{da_compound.name}'] = fld_cells_by_driver
                fld_da_compound.append(da_compound)
                fld_da_diff.append(da_diff)
                run_ids.append(f'{da_compound.name}')
                print(da_compound.name)

                # Loop through 5 different SLR scenarios for each present Scaled run
                slr_runs = [f'{run}_SLR{i}' for i in np.arange(1, 6, 1)]
                for slr_run in slr_runs:
                    # Calculate the number of grid cells attributed to each driver for the peak flood extent
                    # and return data array of compound
                    name_out = f'{storm}_{climate}_{slr_run}'
                    if climate == 'presScaled':
                        compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_{slr_run}_compound',
                                                                 f'{storm}_{climate}_{run}_runoff',
                                                                 f'{storm}_{climate}_{slr_run}_coastal']
                    elif climate == 'pres':
                        compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_{slr_run}_compound',
                                                                 f'{storm}_{climate}_{run}_runoff',
                                                                 f'{storm}_presScaled_{slr_run}_coastal']
                        cc_run_ids.append(name_out)
                    out = classify_zsmax_by_driver(da=da_zsmax,
                                                   compound_key=compound_key, runoff_key=runoff_key,
                                                   coastal_key=coastal_key, name_out=name_out, hmin=hmin)

                    da_classified, fld_cells_by_driver, da_compound, da_diff = out
                    fld_da_classified.append(da_classified)
                    fld_cells[f'{da_compound.name}'] = fld_cells_by_driver
                    fld_da_compound.append(da_compound)
                    fld_da_diff.append(da_diff)
                    run_ids.append(f'{da_compound.name}')
                    print(da_compound.name)

    # Concatenate the data arrays
    fld_da_compound = xr.concat(fld_da_compound, dim='run')
    fld_da_compound['run'] = xr.IndexVariable('run', run_ids)
    fld_da_compound.to_netcdf('pgw_compound_extent_all.nc')

    fld_da_classified = xr.concat(fld_da_classified, dim='run')
    fld_da_classified['run'] = xr.IndexVariable('run', run_ids)
    fld_da_classified.to_netcdf('pgw_drivers_classified_all.nc')

    fld_da_diff = xr.concat(fld_da_diff, dim='run')
    fld_da_diff['run'] = xr.IndexVariable('run', run_ids)
    fld_da_diff.to_netcdf('pgw_WL_maxCmpd_minus_maxIndiv_all.nc')

    # Cleanup flood area dataframe
    fld_cells.index = ['no_flood', 'coastal', 'compound_coastal', 'runoff', 'compound_runoff']
    fld_cells = pd.DataFrame(fld_cells)
    fld_cells.to_csv('pgw_drivers_classified_all_cellCount.csv')

''' Parse the drivers of the ensemble mean '''
for type in ['mean', 'max']:
    zsmax_file = os.path.join(work_dir, 'zsmax', f'pgw_ensmean_zsmax_{type}.nc')
    da_ensmean = xr.open_dataarray(zsmax_file)

    if os.path.exists(f'pgw_drivers_classified_ensmean_{type}.nc') is False:
        fld_cells = pd.DataFrame()  # dataframe populated with total flooded area
        fld_da_compound = []  # populated with data arrays of the compound areas for each run
        fld_da_classified = []
        fld_da_diff = []
        run_ids = []
        for storm in ['flor', 'floy', 'matt']:
            for climate in ['pres', 'presScaled']:
                compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_compound_{type}',
                                                         f'{storm}_{climate}_runoff_{type}',
                                                         f'{storm}_{climate}_coastal_{type}']
                out = classify_zsmax_by_driver(da=da_ensmean,
                                               compound_key=compound_key, runoff_key=runoff_key,
                                               coastal_key=coastal_key, name_out=f'{storm}_{climate}_ensmean',
                                               hmin=0.05)
                da_classified, fld_cells_by_driver, da_compound, da_diff = out
                fld_da_classified.append(da_classified)
                fld_cells[f'{da_compound.name}'] = fld_cells_by_driver
                fld_da_compound.append(da_compound)
                fld_da_diff.append(da_diff)
                run_ids.append(f'{da_compound.name}')
                print(da_compound.name)

        # Concatenate the data arrays
        fld_da_compound = xr.concat(fld_da_compound, dim='run')
        fld_da_compound['run'] = xr.IndexVariable('run', run_ids)
        fld_da_compound.to_netcdf(f'pgw_compound_extent_ensmean_{type}.nc')

        fld_da_classified = xr.concat(fld_da_classified, dim='run')
        fld_da_classified['run'] = xr.IndexVariable('run', run_ids)
        fld_da_classified.to_netcdf(f'pgw_drivers_classified_ensmean_{type}.nc')

        fld_da_diff = xr.concat(fld_da_diff, dim='run')
        fld_da_diff['run'] = xr.IndexVariable('run', run_ids)
        fld_da_diff.to_netcdf(f'pgw_WL_maxCmpd_minus_maxIndiv_ensmean_{type}.nc')

        # Cleanup flood area dataframe
        # No Flood = 0, Coastal = 1, Compound-coastal = 2, Runoff = 3, Compound-runoff = 4
        fld_cells.index = ['no_flood', 'coastal', 'compound_coastal', 'runoff', 'compound_runoff']
        fld_cells = pd.DataFrame(fld_cells)
        fld_cells.to_csv(f'pgw_drivers_classified_ensmean_{type}_cellCount.csv')
