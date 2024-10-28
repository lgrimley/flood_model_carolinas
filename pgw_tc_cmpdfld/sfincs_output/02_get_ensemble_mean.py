import sys

sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\pgw_tc_cmpdfld\sfincs_output')
from pgw_utils import *

# Analysis directory
workdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3'
da_zsmax = xr.open_dataset(os.path.join(workdir, 'pgw_zsmax.nc'))

''' Calculate SFINCS ensemble mean/max water levels '''
storms = ['flor', 'matt', 'floy']
scenarios = ['coastal', 'runoff', 'compound']
for ntype in ['mean', 'max']:
    # Output directory
    out_dir = os.path.join(workdir, f'ensemble_{ntype}')
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    # Empty lists for storing
    ensemble_da_list = []
    run_ids = []
    for storm in storms:
        for climate in ['fut']:
            for scenario in scenarios:
                ''' Calculate the ensemble mean/max '''
                da_ensemble = summarize_ensemble(da_zsmax=da_zsmax,
                                                 storm=storm,
                                                 climate=climate,
                                                 scenario=scenario,
                                                 ntype=ntype)

                # Append the summarized ensemble data array to a list
                ensemble_da_list.append(da_ensemble)
                # Append the run ID for indexing at the end
                run_ids.append(da_ensemble.name)
                print(da_ensemble.name)

    da_ensmean = xr.concat(ensemble_da_list, dim='run')
    da_ensmean['run'] = xr.IndexVariable('run', run_ids)
    da_ensmean.to_netcdf(os.path.join(out_dir, f'fut_ensemble_zsmax_{ntype}.nc'))
