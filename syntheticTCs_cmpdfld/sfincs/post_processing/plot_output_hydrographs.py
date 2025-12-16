import os
import pandas as pd
import hydromt_sfincs.utils
from hydromt_sfincs import SfincsModel, utils
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld')


tc_ids = [1218, 2041, 5701, 4762, 2818]

station_name = 'Tar_0020'
#'Neuse_0015'
    #'WhiteOak_0004'
    #'New_0011'
#'CapeFear_0007'
#'Waccamaw_0010'
#'CapeFear_0013'  #['NCEM_30015','USGS_2135200'
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(7, 10), sharex=False, sharey=True)

for i, tc_id in enumerate(tc_ids):
    tc_dir = rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\\canesm_ssp585_runs\completed_runs\TC_{tc_id}'

    for p in ['compound', 'runoff', 'coastal']:
        ds = xr.open_dataset(os.path.join(tc_dir, p, 'sfincs_his.nc'))

        # Get the station data for querying SFINCS results
        mod_zs_da = ds['point_zs']
        mod_zs_lookup = pd.DataFrame()
        mod_zs_lookup['station_id'] = mod_zs_da['station_id'].values
        mod_zs_lookup['station_name'] = [x.decode('utf-8').strip() for x in mod_zs_da['station_name'].values]
        mod_zs_lookup['data_source'] = [x.rsplit('_', 1)[0] for x in mod_zs_lookup['station_name']]
        mod_zs_lookup['data_source_id'] = [x.split('_')[-1] for x in mod_zs_lookup['station_name']]
        #mod_zs_lookup['x']= mod_zs_da['point_x'].values
        #mod_zs_lookup['y']= mod_zs_da['point_y'].values
        #mod_zs_lookup.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\stations.csv', index=False)

        mod_zs_lookup = mod_zs_lookup.apply(lambda column: column.astype(str))
        ts = mod_zs_lookup[mod_zs_lookup['station_name'] == station_name]
        ds_selected = ds.sel(stations=ds.station_id == float(ts['station_id'].item()))['point_zs']

        # Plot each variable with its own label
        ds_selected.plot(ax=axes[i], label=p)

    axes[i].set_title(f'TC {tc_id} at {station_name}', fontsize=9)
    axes[i].legend(fontsize=9)
    axes[i].set_xlabel('')

plt.tight_layout()
plt.show()

