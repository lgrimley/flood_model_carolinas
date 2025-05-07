import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\canesm_ssp585_runs\completed_runs_SLR112cm')
tc_id = 5902
data = xr.open_dataset(filename_or_obj=os.path.join(os.getcwd(), f'TC_{tc_id}', 'compound', 'sfincs_his.nc'))
# Get the station data for querying SFINCS results
mod_zs_da = data['point_zs']
mod_zs_lookup = pd.DataFrame()
mod_zs_lookup['station_id'] = mod_zs_da['station_id'].values
mod_zs_lookup['station_name'] = [x.decode('utf-8').strip() for x in mod_zs_da['station_name'].values]
mod_zs_lookup['data_source'] = [x.rsplit('_', 1)[0] for x in mod_zs_lookup['station_name']]
mod_zs_lookup['data_source_id'] = [x.split('_')[-1] for x in mod_zs_lookup['station_name']]

scenarios = ['runoff','coastal','compound','runoff_SLR112cm', 'coastal_SLR112cm', 'compound_SLR112cm']

river_name = 'Tar'
river_stations = mod_zs_lookup[mod_zs_lookup['data_source'] == river_name]
river_stations_df = []
sel = river_stations.index.values[28:39]
num_stas = len(sel)
for sta in sel:
    sta_df = pd.DataFrame()
    for scen in scenarios:
        data = xr.open_dataset(filename_or_obj=os.path.join(os.getcwd(), f'TC_{tc_id}', scen, 'sfincs_his.nc'))
        sta_data = data.sel(stations=sta)['point_zs'].to_dataframe()[['point_zs']]
        sta_data.columns = [scen]
        sta_df = pd.concat(objs=[sta_df, sta_data], axis=1, ignore_index=False)
    river_stations_df.append(sta_df)

# PLOTTTTTTT
ncol = 2
nrow = int(np.ceil(num_stas/2))
fig, axes = plt.subplots(nrows=nrow, ncols=ncol,figsize=(6,8), sharey=False, sharex=True)
axes = axes.flatten()
if (ncol*nrow) > num_stas:
    axes[-1].set_axis_off()
for i in range(num_stas):
    d = river_stations_df[i]
    cs = d.plot(ax=axes[i], legend=False)
    #axes[i].set_title(river_stations.index.values[i])
    if i == (num_stas - 1):
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.margins(x=0, y=0)
plt.savefig(rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\SLR_analysis\canesme_TC_{tc_id}_{river_name}.jpg',
            bbox_inches='tight', dpi=300)
plt.close()

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS')
period = 'future'
rp_threshold = 80
# Now just look at the storms we already have
rp_file = fr'.\results_jointprob\basin_data\Neuse_data_rp_{period}.csv'
rp_data = pd.read_csv(rp_file, index_col=0)


wl_da = xr.open_dataset(r'.\SLR_analysis\canesm_ssp585_SRL112cm\zsmax_canesm_slr.nc')

station_data = wl_da.sel(x=855057.831403,y=3901339.81608,method='nearest')
tcs = [x for x in station_data.tc_id.values if 'SLR' not in x]
tcs_slr = [x for x in station_data.tc_id.values if 'SLR' in x]

sta_data = station_data.sel(tc_id=tcs)
slr_sta_data = station_data.sel(tc_id=tcs_slr)

# Compound
df = sta_data.sel(scenario='compound').to_dataframe()
df.index =df.index.astype(int)
df1 = df.join(rp_data, how='inner')
df2 = df1.sort_values(by='Compound_rp')

df = slr_sta_data.sel(scenario='compound').to_dataframe()
df.index = [x.split('_')[0] for x in df.index.values]
df.index =df.index.astype(int)
df1 = df.join(rp_data, how='inner')
df3 = df1.sort_values(by='Compound_rp')


fig, ax = plt.subplots()
df2.plot.scatter(x='Compound_rp',y='zsmax', ax=ax, label='NoSLR')
df3.plot.scatter(x='Compound_rp',y='zsmax', ax=ax, label='SLR', color='green')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1,2000)

# Runoff
df = sta_data.sel(scenario='runoff').to_dataframe()
df.index =df.index.astype(int)
df1 = df.join(rp_data, how='inner')
df2 = df1.sort_values(by='Runoff_rp')

df = slr_sta_data.sel(scenario='runoff').to_dataframe()
df.index = [x.split('_')[0] for x in df.index.values]
df.index =df.index.astype(int)
df1 = df.join(rp_data, how='inner')
df3 = df1.sort_values(by='Runoff_rp')


fig, ax = plt.subplots()
df2.plot.scatter(x='Runoff_rp',y='zsmax', ax=ax, label='NoSLR')
df3.plot.scatter(x='Runoff_rp',y='zsmax', ax=ax, label='SLR', color='green')
ax.set_xscale('log')
ax.set_xlim(1,2000)

# Coastal
df = sta_data.sel(scenario='coastal').to_dataframe()
df.index =df.index.astype(int)
df1 = df.join(rp_data, how='inner')
df2 = df1.sort_values(by='Coastal_rp')

df = slr_sta_data.sel(scenario='coastal').to_dataframe()
df.index = [x.split('_')[0] for x in df.index.values]
df.index =df.index.astype(int)
df1 = df.join(rp_data, how='inner')
df3 = df1.sort_values(by='Coastal_rp')


fig, ax = plt.subplots()
df2.plot.scatter(x='Coastal_rp',y='zsmax', ax=ax, label='NoSLR')
df3.plot.scatter(x='Coastal_rp',y='zsmax', ax=ax, label='SLR', color='green')
ax.set_xscale('log')
ax.set_xlim(1,2000)
