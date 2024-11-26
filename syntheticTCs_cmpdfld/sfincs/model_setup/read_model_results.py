import os

import hydromt_sfincs
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(r'C:\Users\lelise\Documents\GitHub\flood_model_carolinas\syntheticTCs_cmpdfld')
from SFINCS_postP_utils import *

# Script used to build model with Hydromt-SFINCS v.1.1.0
print(f'Hydromt version: {hydromt.__version__}')
print(f'Hydromt-Sfincs version: {hydromt_sfincs.__version__}')

# Load in the data catalogs needed for building the model
cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
yml_sfincs_Carolinas_Ch3 = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas_Ch3.yml')
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\SFINCS_mod_setup\base_model'
mod = SfincsModel(root=root, mode='r',
                  data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas, yml_sfincs_Carolinas_Ch3])
cat = mod.data_catalog

''' Loop through and get the model results '''
tc_id = 2645
tc_dir = fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\SFINCS_mod_setup\TC_{tc_id}'
os.chdir(tc_dir)

# Read results
da_zsmax_list = []
scenarios = ['coastal', 'runoff', 'compound', 'coastal_SLR','compound_SLR']
for scen in scenarios:
    print(scen)
    mod.read_results(fn_map=os.path.join(tc_dir, scen, 'sfincs_map.nc'), fn_his=os.path.join(tc_dir, scen, 'sfincs_his.nc'))
    zsmax = mod.results["zsmax"].max(dim='timemax')
    da_zsmax_list.append(zsmax)

da_zsmax = xr.concat(da_zsmax_list, dim='run')
da_zsmax['run'] = xr.IndexVariable('run', scenarios)
da_zsmax = da_zsmax.assign_attrs(tc_id=tc_id, units='m', datum='NAVD88')
#da_zsmax.to_netcdf('zsmax.nc')

hmin = 0.1
da_diff, da_classified, da_compound_extent = classify_zsmax_by_process(da_zsmax=da_zsmax,
                                                                       compound_key='compound_SLR',
                                                                       runoff_key='runoff',
                                                                       coastal_key='coastal_SLR',
                                                                       hmin=hmin)
da_diff.to_netcdf(f'peakWL_diff_{hmin}m_SLR.nc')
da_diff.raster.to_raster(f'peakWL_diff_{hmin}m_SLR.tif', nodata=np.nan)
da_classified.to_netcdf(f'peakWL_class_{hmin}m_SLR.nc')
da_classified.raster.set_crs(32617)
da_classified.raster.to_raster(f'peakWL_class_{hmin}m_SLR.tif', nodata=0)
da_compound_extent.to_netcdf(f'compound_extent_{hmin}m_SLR.nc')

master_df = pd.DataFrame()
for scen in scenarios:
    results = xr.open_dataset(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\SFINCS_mod_setup\TC_{tc_id}\{scen}\sfincs_his.nc')
    point_zs = results['point_zs']
    point_zb = results['point_zb']

    point_name_mapper = point_zs['station_id'].to_dataframe()
    point_name_mapper['station_name'] = [sta.decode('utf-8').strip() for sta in point_name_mapper['station_name']]
    subset = point_name_mapper[point_name_mapper['station_name'].str.contains('NECapeFear')]
    sta_maxzs_ls = []
    point_zb_ls = []
    for station in subset.index:
        sta_maxzs = point_zs.sel(stations=station).max().values.item()
        sta_maxzs_ls.append(np.round(sta_maxzs,decimals=3))

        sta_zb = point_zb.sel(stations=station).values.item()
        point_zb_ls.append(np.round(sta_zb,decimals=3))

    print(sta_maxzs_ls)

    df = pd.DataFrame()
    df['station'] = subset.index
    df[f'peakWL_{scen}'] = sta_maxzs_ls
    df.set_index('station', inplace=True, drop=True)

    master_df = pd.concat([master_df, df], axis=1, ignore_index=False)

master_df['order'] = [int(s.split('_')[-1]) for s in subset['station_name']]
#master_df['zb'] = point_zb_ls
ds_sorted = master_df.sort_values(by='order', ascending=True)
ds_sorted.set_index('order', inplace=True, drop=True)


fig, ax = plt.subplots()
ds_sorted.plot(ax=ax, style='o-')
plt.ylabel('Elevation (m+NAVD88)')
plt.xlabel('Downstream to Upstream')
plt.xticks(ds_sorted.index, rotation=0)
plt.savefig(f'NECapeFear_{tc_id}.png', bbox_inches="tight" )
plt.close()


results = mod.results
point_zs = results['point_zs']
point_name_mapper = point_zs['station_id'].to_dataframe()
point_name_mapper['station_name'] = [sta.decode('utf-8').strip() for sta in point_name_mapper['station_name']]
subset = point_name_mapper[point_name_mapper['station_name'].str.contains('NOAA')]

sub = point_zs.sel(stations=subset.index.values)
sub_df = sub.to_dataframe()
sub_df_unstacked = sub_df.unstack(level='stations')

fig, ax = plt.subplots()
sub_df_unstacked['point_zs'].plot(ax=ax)
plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\stormTide\ADCIRC_storm_vs_reanalysis\test.png', bbox_inches="tight" )
plt.close()
