import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from src.utils import  classify_zsmax_by_process, calculate_flooded_area_by_process
from hydromt_sfincs import SfincsModel


'''' Load in the data '''
# Load the water level data for the historical return periods
histdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep\aep'
file_paths = [os.path.join(histdir, file) for file in os.listdir(histdir) if ('returnPeriods' in file) & (file.endswith('.nc'))]
scenarios = [file.split('_')[-1].split('.')[0] for file in os.listdir(histdir) if ('returnPeriods' in file) & (file.endswith('.nc'))]
da_list = [xr.open_dataarray(file, engine='netcdf4') for file in file_paths]
h_aep = xr.concat(objs=da_list, dim='scenario')
h_aep['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

# Load the water level data for the projected return periods
projdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\canesm_ssp585\aep'
file_paths = [os.path.join(projdir, file) for file in os.listdir(projdir) if ('returnPeriods' in file) & (file.endswith('.nc'))]
da_list = [xr.open_dataarray(file, engine='netcdf4') for file in file_paths]
p_aep = xr.concat(objs=da_list, dim='scenario')
p_aep['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

# Load model data/DEM
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
dem = mod.grid['dep']

# Calculate change in the relative contribution of different processes to AEP floods
wb_mask = mod.data_catalog.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\waterbody_mask.nc')
sel_rp = h_aep['return_period'].values
d_plot = []
combined_df = pd.DataFrame()
for T in sel_rp:
    h = h_aep.sel(return_period=T)
    h = h.where(wb_mask == 1)
    da_single_max = h.sel(scenario=['runoff', 'coastal']).max('scenario')
    da_diff_h = (h.sel(scenario='compound') - da_single_max).compute()
    da_diff_h.name = 'zsmax_diff'
    da_class = classify_zsmax_by_process(da=h, da_diff=da_diff_h, hmin= 0.05)
    fld_area_h = calculate_flooded_area_by_process(da = da_class['zsmax_attr'], tc_index=T).T
    fld_area_h.columns = [f'hist_{T}']

    p = p_aep.sel(return_period=T)
    p = p.where(wb_mask == 1)
    da_single_max = p.sel(scenario=['runoff', 'coastal']).max('scenario')
    da_diff_p = (p.sel(scenario='compound') - da_single_max).compute()
    da_diff_p.name = 'zsmax_diff'
    da_class_p = classify_zsmax_by_process(da=p, da_diff=da_diff_p, hmin= 0.05)
    fld_area_p = calculate_flooded_area_by_process(da = da_class_p['zsmax_attr'], tc_index=T).T
    fld_area_p.columns = [f'proj_{T}']

    combined_df = pd.concat(objs=[combined_df, fld_area_h, fld_area_p], axis=1, ignore_index=False)
#combined_df.round(2).to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\aep_extent_rel_contribution_Wbmasked.csv' )

