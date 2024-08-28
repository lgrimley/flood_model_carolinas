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


work_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\analysis'
out_dir = os.path.join(work_dir, 'driver_analysis')
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)
os.chdir(out_dir)

storms = ['flor', 'floy', 'matt']
climates = ['pres', 'fut']
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
        nruns = 8
        if storm == 'matt':
            nruns = 7
        for climate in climates:
            if climate == 'pres':
                # Get the model names to query zsmax results from data array
                compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_compound',
                                                         f'{storm}_{climate}_runoff',
                                                         f'{storm}_{climate}_coastal']
                nameout = f'{storm}_{climate}'

                # Attribute max water levels to coastal, runoff or compound processes
                # Outputs:
                #       da_classified - a data array with the zsmax attributed to processes (codes 0 to 4)
                #       fld_cells_by_driver - a numpy array with the total number of grid cells for each process
                #       da_compound - a data array of the cells where compound flooding occurred (e.g., 0 and 1)
                #       da_diff - a data array of the diff in waterlevel compound minus max. single driver
                out = classify_zsmax_by_driver(da=da_zsmax,  # data array with zsmax for all simulations
                                               compound_key=compound_key,
                                               runoff_key=runoff_key,
                                               coastal_key=coastal_key,
                                               name_out=nameout,
                                               hmin=hmin  # depth difference threshold for compound vs. individual
                                               )
                da_classified, fld_cells_by_driver, da_compound, da_diff = out

                # Append all the output from the function above to lists
                fld_da_classified.append(da_classified)
                fld_cells[f'{da_compound.name}'] = fld_cells_by_driver
                fld_da_compound.append(da_compound)
                fld_da_diff.append(da_diff)
                run_ids.append(f'{da_compound.name}')
                print(da_compound.name)

            if climate == 'fut':
                for sf in np.arange(1, nruns, 1):
                    for slr in np.arange(1, 6, 1):
                        compound_key, runoff_key, coastal_key = [f'{storm}_{climate}_SF{sf}_SLR{slr}_compound',
                                                                 f'{storm}_{climate}_SF{sf}_runoff',
                                                                 f'{storm}_{climate}_SF{sf}_SLR{slr}_coastal']

                        nameout = f'{storm}_{climate}_SF{sf}_SLR{slr}'
                        out = classify_zsmax_by_driver(da=da_zsmax,
                                                       compound_key=compound_key,
                                                       runoff_key=runoff_key,
                                                       coastal_key=coastal_key,
                                                       name_out=nameout,
                                                       hmin=hmin
                                                       )

                        da_classified, fld_cells_by_driver, da_compound, da_diff = out
                        fld_da_classified.append(da_classified)
                        fld_cells[f'{da_compound.name}'] = fld_cells_by_driver
                        fld_da_compound.append(da_compound)
                        fld_da_diff.append(da_diff)
                        run_ids.append(f'{da_compound.name}')
                        print(da_compound.name)

    # Concatenate the data arrays
    fld_da_classified = xr.concat(fld_da_classified, dim='run')
    fld_da_classified['run'] = xr.IndexVariable('run', run_ids)
    fld_da_classified.to_netcdf('pgw_drivers_classified_all.nc')

    fld_da_compound = xr.concat(fld_da_compound, dim='run')
    fld_da_compound['run'] = xr.IndexVariable('run', run_ids)
    fld_da_compound.to_netcdf('pgw_compound_extent_all.nc')

    # Cleanup flood area dataframe
    fld_cells.index = ['no_flood', 'coastal', 'compound_coastal', 'runoff', 'compound_runoff']
    fld_cells = pd.DataFrame(fld_cells)
    fld_cells.to_csv('pgw_drivers_classified_all_cellCount.csv')

    fld_da_diff = xr.concat(fld_da_diff, dim='run')
    fld_da_diff['run'] = xr.IndexVariable('run', run_ids)
    fld_da_diff.to_netcdf('pgw_WL_maxCmpd_minus_maxIndiv_all.nc')


''' Parse the drivers of the ensemble mean '''
for type in ['mean', 'max']:
    zsmax_file = os.path.join(work_dir, 'zsmax', f'fut_ensemble_zsmax_{type}.nc')
    da_ensmean = xr.open_dataarray(zsmax_file)

    # if os.path.exists(f'pgw_drivers_classified_ensmean_{type}.nc') is False:
    fld_cells = pd.DataFrame()  # dataframe populated with total flooded area
    fld_da_compound = []  # populated with data arrays of the compound areas for each run
    fld_da_classified = []
    fld_da_diff = []
    run_ids = []
    for storm in ['flor', 'floy', 'matt']:
        for climate in ['fut']:
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
