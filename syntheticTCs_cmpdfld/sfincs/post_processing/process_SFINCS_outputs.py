import os

import hydromt
import xarray as xr
from hydromt_sfincs import SfincsModel
from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld')
from src.core import TCFloodHazard
import time

#sbg = sfincs_mod.data_catalog.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\subgrid\dep_subgrid_20m.tif')

# Load SFINCS model for importing results
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
sfincs_mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
sfincs_mod.read() # for some reason this has to be run twice for the code to work below

# Change the directory to the model results
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

run_group = 'ncep'
foldname = 'NCEP_Reanalysis'
group_size = 125

root = Path(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\{run_group}_runs\completed_runs')
# These are landfall vmax
tc_ids_vmax = pd.read_csv(fr'.\02_DATA\{foldname}\tracks\{run_group}_landfall_vmax.csv', index_col=0)
# But these are the storms we actually have ADCIRC data for
correct = pd.read_csv(fr'.\02_DATA\{foldname}\stormTide\gage_peaks_ZerosRemoved_{run_group}.csv',index_col=0)
# So take only the TC IDs of the storms we care about and have runs for
selected = tc_ids_vmax[tc_ids_vmax['tc_id'].isin(correct.index)]
tc_ids_vmax_ordered = selected.sort_values(by='vstore100', ascending=True)
groups = [tc_ids_vmax_ordered[i:i + group_size] for i in range(0, len(tc_ids_vmax_ordered), group_size)]

outputdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS'
outdir = os.path.join(outputdir, run_group)

if os.path.exists(outdir) is False:
    os.makedirs(outdir)
counter = 0
for group in groups:
    vmin, vmax = (int(group['vstore100'].min()), int(group['vstore100'].max()))
    group_info = f'VmaxLf_{vmin}_{vmax}'
    start_time = time.time()

    # Create an empty dictionary to save the SFINCS results as we loop through the TC tracks
    scenario_results = []
    attribution_results = []
    for tc_index in group['tc_id'].values:
        # This is the directory where the model run files are saved that we read in
        tc_root = os.path.join(root, f'TC_{str(tc_index).zfill(4)}')
        try:
            # Try to load the SFINCS outputs for the track run and save them to the results dictionary
            r = TCFloodHazard(tc_root=Path(tc_root), sfincs_mod = sfincs_mod, tc_index=tc_index, tracks=run_group,
                              zsmax_threshold=0.05)
            scenario_results.append(r.scenario_results)
            attribution_results.append(r.attribution_results)
            print(f'Successfully processed {tc_index}')
        except Exception as e:
            print(e)
            print(f'Issue with {tc_index}')
            continue

    # Write the outputs to NetCDFs
    attrs = {
        'description': f'Modeled SFINCS ouputs for {group_info}',
        'track_model' : run_group,
        'author': 'Lauren Grimley',
        'date_created' : '1/16/2025'
    }

    ds1 = xr.concat(objs=scenario_results, dim='tc_id')
    ds1['tc_id'] = xr.IndexVariable(dims='tc_id', data=group['tc_id'].values)
    var = 'zsmax'
    outfilepath = os.path.join(outputdir, run_group, f'{var}_{group_info}.nc')
    if os.path.exists(outfilepath) is False:
        print(f'Writing {var}...')
        dsout = ds1[var]
        dsout.attrs = attrs
        dsout.to_netcdf(outfilepath)

    # for var in ds1.data_vars:
    #     outfilepath = os.path.join(outputdir, run_group, f'{var}_{group_info}.nc')
    #     if os.path.exists(outfilepath) is False:
    #         print(f'Writing {var}...')
    #         dsout = ds1[var]
    #         dsout.attrs = attrs
    #         dsout.to_netcdf(outfilepath)

    outfilepath = os.path.join(outputdir, run_group, f'attribution_{group_info}.nc')
    if os.path.exists(outfilepath) is False:
        ds2 = xr.concat(objs=attribution_results, dim='tc_id')
        ds2['tc_id'] = xr.IndexVariable(dims='tc_id', data=group['tc_id'].values)
        ds2.attrs = attrs
        outfilepath = os.path.join(outputdir, run_group, f'attribution_{group_info}.nc')
        print(f'Writing attribution...')
        ds2.to_netcdf(outfilepath)

    print(f'Finished writing output for {group_info}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time writing {group_info}: {elapsed_time} seconds")
    counter += 1
    print(f'{counter} out of {len(groups)} groups processed...')



