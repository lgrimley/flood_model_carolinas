import os
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# This script plots model output at an obs location matching based on the ID in the sfincs_his.nc
def get_pointzs_at_station(sta_name, mod_name, results_dir):
    try:
        results = xr.open_dataset(os.path.join(results_dir, mod_name, 'sfincs_his.nc'))
    except:
        print(f'Could not load sfincs_his.nc for {mod_name}')
    else:
        res_da = results['point_zs']
        sta_df = results.stations.to_dataframe()
        sta_df['station_name'] = sta_df['station_name'].str.decode('utf-8')
        sta_df['station_name'] = sta_df['station_name'].str.strip()

        if sta_name in sta_df['station_name'].values:
            sta_id = sta_df[sta_df['station_name'] == sta_name]['stations']
            zs_df = res_da.sel(stations=sta_id.item()).to_dataframe()
            zs_df = zs_df['point_zs']
            zs_df.name = f'{mod_name}'
            zs_df = pd.DataFrame(zs_df).sort_index()
            return zs_df
        else:
            print(f'{sta_name} not in sfincs_his.nc for {mod_name}')
            return False


def combine_mod_pointzs(sta_name, results_dir, storm, climate, scenario):
    ts_df = []
    nruns = 8
    if storm == 'matt':
        nruns = 7
    runs = [f'ens{i}' for i in np.arange(1, nruns, 1)] + ['ensmean']
    for run in runs:
        if climate == 'preScaled':
            slr_runs = [f'{run}_SLR{i}' for i in np.arange(1, 6, 1)]
            for slr_run in slr_runs:
                mod_name = f'{storm}_{climate}_{slr_run}_{scenario}'
                zs_df = get_pointzs_at_station(sta_name=sta_name, mod_name=mod_name, results_dir=results_dir)
                if zs_df is False:
                    continue
                else:
                    if len(ts_df) > 0:
                        ts_df = pd.concat([ts_df, zs_df], axis=1)
                    else:
                        ts_df = zs_df
        else:
            mod_name = f'{storm}_{climate}_{run}_{scenario}'
            zs_df = get_pointzs_at_station(sta_name=sta_name, mod_name=mod_name, results_dir=results_dir)
            if zs_df is False:
                continue
            else:
                if len(ts_df) > 0:
                    ts_df = pd.concat([ts_df, zs_df], axis=1)
                else:
                    ts_df = zs_df
    return ts_df


results_dir = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\arch'
out_dir = os.path.join(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis\hydrographs')
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)
os.chdir(out_dir)

storms = ['flor', 'floy', 'matt']
climates = ['pres', 'presScaled']
scenario = 'compound'
sta_names = ['RVR_19102',
             'RVR_19120',
             'NCEM_30611',
             'NCEM_30001',
             'RVR_16111'
             ]

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
nrow = 3
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)

for sta_name in sta_names:
    sta_ts_df = []
    for storm in storms:
        for climate in climates:
            ts_df = combine_mod_pointzs(sta_name=sta_name, results_dir=results_dir,
                                        storm=storm, climate=climate, scenario=scenario)
            sta_ts_df.append(ts_df)
    if len(ts_df) > 0:
        # Plot setup
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6.5, 6),
                                 sharex=False, sharey=True,
                                 tight_layout=True, layout='constrained')
        axes = axes.flatten()
        for i in range(len(sta_ts_df)):
            ax = axes[i]
            sta_ts_df[i].plot(ax=ax, legend=False)
            ax.set_xlabel('')
            ax.set_ylabel('Water Level (m)')

        axes[first_row[0]].set_title('Present')
        axes[first_row[1]].set_title('Future')
        plt.suptitle(sta_name)
        plt.savefig(f'{sta_name}_{scenario}.png', dpi=225, tight_layout=True)
        plt.close()
