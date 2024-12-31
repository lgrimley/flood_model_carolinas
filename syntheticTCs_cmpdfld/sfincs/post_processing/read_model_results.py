
import hydromt_sfincs
from hydromt_sfincs import SfincsModel

from src.core import *
from src.utils import *

# Script used to build model with Hydromt-SFINCS v.1.1.0
print(f'Hydromt version: {hydromt.__version__}')
print(f'Hydromt-Sfincs version: {hydromt_sfincs.__version__}')

base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r')
mod.read()


run_dir=r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\completed_runs\TC_2645\coastal'
mod.read_config(os.path.join(run_dir, 'sfincs.inp'))
mod.read_results(fn_map=os.path.join(run_dir, 'sfincs_map.nc'),
                 fn_his=os.path.join(run_dir, 'sfincs_his.nc'),
                 decode_times=False)
zsmax = mod.results['zsmax'].max(dim='timemax')
zsmax.raster.to_raster('coastal_2645.tif', nodata=-999.0)

# zsmax_da2 = xr.concat(zsmax_da, dim='tc_index')
# zsmax_da2['tc_index'] = xr.IndexVariable('tc_index', tc_da)
# zsmax_da2.to_netcdf(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\NCEP\zsmax.nc')
#
# zsmax_da2 = xr.concat(zsmax_runoff, dim='tc_index')
# zsmax_da2['tc_index'] = xr.IndexVariable('tc_index', tc_runoff)
# zsmax_da2.to_netcdf(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\NCEP\zsmax_runoff.nc')
#
# zsmax_da2 = xr.concat(zsmax_coastal, dim='tc_index')
# zsmax_da2['tc_index'] = xr.IndexVariable('tc_index', tc_coastal)
# zsmax_da2.to_netcdf(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\NCEP\zsmax_coastal.nc')

# s = [x[0] for x in problem]
#
#
# tc_index_to_redo = []
# for d in os.listdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\runs'):
#     tc_index = int(d.split('_')[-1])
#     tc_index_to_redo.append(tc_index)
#
# tc_index_to_redo = tc_index_to_redo + s
# df = pd.DataFrame(tc_index_to_redo)
# df.columns=['tc_id']
# df.drop_duplicates(inplace=True)
# df.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\runs_remaining.csv')


# rivers=['Tar', 'Grindle', 'BriceCreek', 'Swift','Contentnea', 'New_',
#        'WhiteOak', 'NewPort','Neuse', 'CapeFear_', 'BlackNC','BlackRiverSC',
#        'NorthEastCapeFear','HollyShelterCreek', 'Waccamaw', 'UpperPeeDee', 'LowerPeeDee',
#        'LittlePeeDee', 'TownCreek', 'LockwoodFolly', 'Shallotte', 'Trent', 'Sampit'] #
# riv_locs = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_base_mod\obsfile\river_obs_locs_v3.csv')
#
# for river in rivers:
#     subset_riv_locs = riv_locs[riv_locs['HYDRAID'].str.contains(river)]
#     subset_riv_locs.set_index('ORIG_SEQ', inplace=True)
#
#     master_df = pd.DataFrame()
#     for scen in scenarios:
#         results = xr.open_dataset(fr'.\{scen}\sfincs_his.nc')
#         point_zs = results['point_zs']
#         point_zb = results['point_zb']
#
#         point_name_mapper = point_zs['station_id'].to_dataframe()
#         point_name_mapper['station_name'] = [sta.decode('utf-8').strip() for sta in point_name_mapper['station_name']]
#
#         point_subset = point_name_mapper[point_name_mapper['station_name'].str.contains(river)]
#         point_subset['ORIG_SEQ'] = [int(s.split('_')[-1]) for s in point_subset['station_name']]
#         point_subset.set_index('ORIG_SEQ', inplace=True)
#
#         sta_maxzs_ls = []
#         point_zb_ls = []
#         for station in point_subset.index:
#             pzb = point_zb.sel(stations=station).values.item()
#             point_zb_ls.append(pzb)
#
#             #pzs = point_zs.sel(stations=station).max().values.item()
#             t = point_zs['time'][int(np.ceil(len(point_zs['time'])/2))]
#             pzs = point_zs.sel(stations=station).sel(time=t).values.item()
#
#             if pzs < pzb:
#                 sta_maxzs_ls.append(np.round(pzb,decimals=3))
#             else:
#                 sta_maxzs_ls.append(np.round(pzs,decimals=3))
#
#         df = pd.DataFrame()
#         df['station'] = point_subset.index
#         df[f'{scen}'] = sta_maxzs_ls
#         df.set_index('station', inplace=True, drop=True)
#
#         master_df = pd.concat([master_df, df], axis=1, ignore_index=False)
#         master_df['zb'] = point_zb_ls
#
#     combined = pd.concat(objs=[master_df, subset_riv_locs], axis=1, ignore_index=False)
#     keep = ['ORIG_LEN', 'zb'] + scenarios
#     plt_df = combined[keep]
#     plt_df.set_index('ORIG_LEN', inplace=True, drop=True)
#     fig, ax = plt.subplots(figsize=(6,4))
#     plt_df.iloc[:][:].plot(ax=ax, marker='.',style='--',alpha=0.7)
#     plt.ylabel('Elevation (m+NAVD88)')
#     plt.xlabel('Distance Upstream (m)')
#     plt.savefig(f'{river}_{tc_id}.png', bbox_inches="tight" )
#     plt.close()

# results = mod.results
# point_zs = results['point_zs']
# point_name_mapper = point_zs['station_id'].to_dataframe()
# point_name_mapper['station_name'] = [sta.decode('utf-8').strip() for sta in point_name_mapper['station_name']]
# subset = point_name_mapper[point_name_mapper['station_name'].str.contains('NOAA')]
#
# sub = point_zs.sel(stations=subset.index.values)
# sub_df = sub.to_dataframe()
# sub_df_unstacked = sub_df.unstack(level='stations')
#
# fig, ax = plt.subplots()
# sub_df_unstacked['point_zs'].plot(ax=ax)
# plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\NCEP_Reanalysis\stormTide\ADCIRC_storm_vs_reanalysis\test.png', bbox_inches="tight" )
# plt.close()
