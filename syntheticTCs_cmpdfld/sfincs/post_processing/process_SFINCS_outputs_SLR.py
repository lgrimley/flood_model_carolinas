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


# Load SFINCS model for importing results
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
sfincs_mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
sfincs_mod.read() # for some reason this has to be run twice for the code to work below

# Change the directory to the model results
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

run_group = 'canesm_ssp585' # 'ncep'
foldname = 'CMIP6_585' #'NCEP_Reanalysis'
root = Path(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\{run_group}_runs\completed_runs_SLR112cm')
tc_ids = [int(x.split('_')[-1]) for x in os.listdir(root)]
group_info = 'canesm_slr'

outputdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS'
outdir = os.path.join(outputdir, f'{run_group}_SRL112cm')

if os.path.exists(outdir) is False:
    os.makedirs(outdir)

start_time = time.time()
# Create an empty dictionary to save the SFINCS results as we loop through the TC tracks
scenario_results = []
attribution_results = []
keep_ids = []
for tc_index in tc_ids:
    # This is the directory where the model run files are saved that we read in
    tc_root = os.path.join(root, f'TC_{str(tc_index).zfill(4)}')
    if os.path.exists(os.path.join(tc_root, 'runoff')) is False:
        print(f'skipping {tc_index}')
        continue
    else:
        for x in [None,'SLR112cm']:
            try:
                if x is None:
                    scenarios_dict = {'coastal': f'coastal', 'runoff': f'runoff', 'compound': f'compound'}
                    keep_ids.append(tc_index)
                else:
                    # Try to load the SFINCS outputs for the track run and save them to the results dictionary
                    scenarios_dict = {'coastal': f'coastal_{x}','runoff': f'runoff_{x}', 'compound': f'compound_{x}'}
                    keep_ids.append(f'{tc_index}_{x}')

                r = TCFloodHazard(tc_root=Path(tc_root), sfincs_mod = sfincs_mod, tc_index=tc_index, tracks=run_group,
                                  zsmax_threshold=0.2, scenarios_dict=scenarios_dict)

                r.scenario_results.attrs['SLR'] = x
                scenario_results.append(r.scenario_results)

                r.attribution_results.attrs['SLR'] = x
                attribution_results.append(r.attribution_results)

                print(f'Successfully processed {tc_index} {x}')

            except Exception as e:
                print(e)
                print(f'Issue with {tc_index} {x}')
                continue

# Write the outputs to NetCDFs
attrs = {
    'description': f'Modeled SFINCS ouputs for {group_info}',
    'track_model' : run_group,
    'author': 'Lauren Grimley',
    'date_created' : '3/17/2025'
}

# ds1 = xr.concat(objs=scenario_results, dim='tc_id')
# ds1['tc_id'] = xr.IndexVariable(dims='tc_id', data=keep_ids)
# var = 'zsmax'
# outfilepath = os.path.join(outdir, f'{var}_{group_info}.nc')
# print(f'Writing {var}...')
# dsout = ds1[var]
# dsout.attrs = attrs
# dsout.to_netcdf(outfilepath)

ds2 = xr.concat(objs=attribution_results, dim='tc_id')
ds2['tc_id'] = xr.IndexVariable(dims='tc_id', data=keep_ids)#group['tc_id'].values)
ds2.attrs = attrs
outfilepath = os.path.join(outdir, f'attribution_{group_info}.nc')
print(f'Writing attribution...')
ds2.to_netcdf(outfilepath)

print(f'Finished writing output for {group_info}')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time writing {group_info}: {elapsed_time} seconds")


