import os
import xarray as xr
from hydromt_sfincs import SfincsModel
from pathlib import Path
import pandas as pd
import numpy as np
from src.core import TCFloodHazard



# Load SFINCS model for importing results
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
sfincs_mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
sfincs_mod.read() # for some reason this has to be run twice for the code to work below

# Change the directory to the model results
root = Path(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\canesm_ssp585_runs\completed_runs')
outputdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS'
group_size = 400
run_group = 'CANESM_SSP585'

# Get a list of all the folders with model results saved in them
# Split them into groups of 400 to save
result_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
tc_ids = [int(x.split('_')[-1]) for x in result_dirs]
tc_ids = sorted(tc_ids)
groups = [tc_ids[i:i + group_size] for i in range(0, len(tc_ids), group_size)]
scenarios = ['compound', 'runoff', 'coastal']
outdir = os.path.join(outputdir, run_group)
if os.path.exists(outdir) is False:
    os.makedirs(outdir)
for group in groups:
    # Create an empty dictionary to save the SFINCS results as we loop through the TC tracks
    tc_results = {'tc_index': [],
                  'zsmax_classified': [],
                  'total_flood_extent': [],
                  'zsmax_diff': [],
                  'total_precip': [],
                  'mean_rain_rate': [],
                  'max_rain_rate': [],
                  'mean_windspeed': [],
                  'max_windspeed': [],
                  'fld_extent': {'compound': [], 'runoff': [], 'coastal': []},
                  'zsmax': {'compound': [], 'runoff': [], 'coastal': []},
                  'vmax': {'compound': [], 'runoff': [], 'coastal': []},
                  'tmax': {'compound': [], 'runoff': [], 'coastal': []},
                  }
    for tc_index in group:
        # This is the directory where the model run files are saved that we read in
        tc_root = os.path.join(root, f'TC_{str(tc_index).zfill(4)}')
        try:
            # Try to load the SFINCS outputs for the track run and save them to the results dictionary
            results = TCFloodHazard(tc_root=Path(tc_root), sfincs_mod = sfincs_mod, tc_index=tc_index,
                                    tracks=run_group, zsmax_threshold=0.1)
            tc_results['zsmax_diff'].append(results.zsmax_diff)
            tc_results['zsmax_classified'].append(results.da_classified)
            tc_results['total_flood_extent'].append(results.da_extents.sel(scenario='total'))
            tc_results['tc_index'].append(tc_index)

            tc_results['total_precip'].append(results.precip.sel(name='total'))
            tc_results['mean_rain_rate'].append(results.precip.sel(name='mean'))
            tc_results['max_rain_rate'].append(results.precip.sel(name='max'))
            tc_results['mean_windspeed'].append(results.wind.sel(name='mean'))
            tc_results['max_windspeed'].append(results.wind.sel(name='max'))

            for scenario in scenarios:
                tc_results['zsmax'][scenario].append(results.varmax[scenario]['zsmax'])
                tc_results['vmax'][scenario].append(results.varmax[scenario]['vmax'])
                tc_results['tmax'][scenario].append(results.varmax[scenario]['tmax'])
                tc_results['fld_extent'][scenario].append(results.da_extents.sel(scenario=scenario))
            print(f'Successfully processed {tc_index}')
        except Exception as e:
            print(e)
            print(f'Issue with {tc_index}')
            continue

    # Loop through the results dictionary and write the outputs to NetCDFs
    group_info = f'TCIDs_{str(group[0]).zfill(4)}_{str(group[-1]).zfill(4)}'
    tc_ids_group = tc_results['tc_index']
    for v in tc_results.keys():
        try:
            print(f'Working on writing output for: {v}')
            if v == 'tc_index':
                continue
            elif isinstance(tc_results[v], dict) is True:
                # These are variables that are saved for each scenario
                for x in tc_results[v].keys():
                    outfilename = f'{v}_{x}_{group_info}.nc'
                    outfilepath = os.path.join(outputdir, run_group, outfilename)
                    if os.path.exists(outfilepath):
                        print(f'Skipping {outfilename} because path already exists.')
                        continue
                    else:
                        da = xr.concat(objs=tc_results[v][x], dim='tc_id')
                        da['tc_id'] = xr.IndexVariable(dims='tc_id', data=tc_results['tc_index'])
                        da.to_netcdf(outfilepath)
                        print(f'Finished writing {outfilename}')
            else:
                outfilename = f'{v}_{group_info}.nc'
                outfilepath = os.path.join(outputdir, run_group, outfilename)
                if os.path.exists(outfilepath):
                    print(f'Skipping {outfilename} because path already exists.')
                    continue
                else:
                    da = xr.concat(objs=tc_results[v], dim='tc_id')
                    da['tc_id'] = xr.IndexVariable(dims='tc_id', data=tc_results['tc_index'])
                    da.to_netcdf(outfilepath)
                    print(f'Finished writing {outfilename}')
        except:
            print(f'Issue writing output for {v}')






