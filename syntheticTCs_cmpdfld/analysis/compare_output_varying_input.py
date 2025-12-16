import os

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
from src.utils import classify_zsmax_by_process, calc_diff_in_zsmax_compound_minus_max_individual



# Load SFINCS model for importing results
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\sfincs_initcond_mod'
sfincs_mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
sfincs_mod.read() # for some reason this has to be run twice for the code to work below

tc_id = 1218
tc_dir = rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\\canesm_ssp585_runs\completed_runs\TC_{tc_id}'

folder_name = 'compound'
sfincs_mod.read_config(os.path.join(tc_dir, folder_name, 'sfincs.inp'))
sfincs_mod.read_results(fn_map=os.path.join(tc_dir, folder_name, 'sfincs_map.nc'),
                        fn_his=os.path.join(tc_dir, folder_name, 'sfincs_his.nc'),
                        decode_times=False)
res = sfincs_mod.results.copy()
da_compound = res['zsmax'].max(dim='timemax')


folder_name = 'runoff'
sfincs_mod.read_config(os.path.join(tc_dir, folder_name, 'sfincs.inp'))
sfincs_mod.read_results(fn_map=os.path.join(tc_dir, folder_name, 'sfincs_map.nc'),
                        fn_his=os.path.join(tc_dir, folder_name, 'sfincs_his.nc'),
                        decode_times=False)
res = sfincs_mod.results.copy()
da_runoff = res['zsmax'].max(dim='timemax')

folder_name = 'coastal'
sfincs_mod.read_config(os.path.join(tc_dir, folder_name, 'sfincs.inp'))
sfincs_mod.read_results(fn_map=os.path.join(tc_dir, folder_name, 'sfincs_map.nc'),
                        fn_his=os.path.join(tc_dir, folder_name, 'sfincs_his.nc'),
                        decode_times=False)
res = sfincs_mod.results.copy()
da_coastal = res['zsmax'].max(dim='timemax')


da_list = [da_compound, da_runoff, da_coastal]
scenarios = ['compound', 'runoff', 'coastal']
x = xr.concat(objs=da_list, dim='scenario')
x['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

da_diff = calc_diff_in_zsmax_compound_minus_max_individual(da = x)
ds = classify_zsmax_by_process(da=x, da_diff=da_diff, hmin=0.05)
ds['zsmax_diff'].raster.to_raster(os.path.join(tc_dir, 'zsmax_diff.tif'))
ds['zsmax_attr'].raster.to_raster(os.path.join(tc_dir, 'zsmax_attr.tif'))

############
###########
############
folder_name = 'runoff_V1'
sfincs_mod.read_config(os.path.join(tc_dir, folder_name, 'sfincs.inp'))
sfincs_mod.read_results(fn_map=os.path.join(tc_dir, folder_name, 'sfincs_map.nc'),
                        fn_his=os.path.join(tc_dir, folder_name, 'sfincs_his.nc'),
                        decode_times=False)
res = sfincs_mod.results.copy()
da_runoff = res['zsmax'].max(dim='timemax')

folder_name = 'coastal_V1'
sfincs_mod.read_config(os.path.join(tc_dir, folder_name, 'sfincs.inp'))
sfincs_mod.read_results(fn_map=os.path.join(tc_dir, folder_name, 'sfincs_map.nc'),
                        fn_his=os.path.join(tc_dir, folder_name, 'sfincs_his.nc'),
                        decode_times=False)
res = sfincs_mod.results.copy()
da_coastal = res['zsmax'].max(dim='timemax')


da_list = [da_compound, da_runoff, da_coastal]
scenarios = ['compound', 'runoff', 'coastal']
x = xr.concat(objs=da_list, dim='scenario')
x['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

da_diff = calc_diff_in_zsmax_compound_minus_max_individual(da = x)
ds = classify_zsmax_by_process(da=x, da_diff=da_diff, hmin=0.05)
ds['zsmax_diff'].raster.to_raster(os.path.join(tc_dir, 'zsmax_diff_V1.tif'))
ds['zsmax_attr'].raster.to_raster(os.path.join(tc_dir, 'zsmax_attr_V1.tif'))





# os.chdir(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\zsmax')
# file_paths = [file for file in os.listdir() if ('zsmax' in file) & (file.endswith('.nc'))]
# ds_list = [xr.open_dataarray(file, chunks={'x': 300, 'y': 300, 'tc_id': 25},
#                              engine='netcdf4').to_dataset(dim='scenario') for file in file_paths]
# ds = xr.concat(ds_list, dim='tc_id')
#
# da_processed = ds.sel(tc_id=3).compute()
# dap_coast = da_processed['coastal']
#
#
# # Using comparison methods:
# print(f"a.equals(b): {da.equals(dap_coast)}") # True (attributes are ignored)
# print(f"a.identical(b): {da.identical(dap_coast)}") # False (attributes are different)
# print(f"a.equals(c): {da.equals(dap_coast)}") # False (values are different)
#
#
# da.raster.to_raster(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\tc3_daV1.tif')
# dap_coast.raster.to_raster(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\tc3_dap_coast.tif')