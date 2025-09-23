import os
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from hydromt_sfincs import SfincsModel
from src.utils import classify_zsmax_by_process, calc_diff_in_zsmax_compound_minus_max_individual
matplotlib.use('TkAgg')
plt.ion()



os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS\ncep\aep\probabilistic_WSE')

da_compound = xr.open_dataarray(fr'.\ncep_AEP_WSE_compound.nc')
da_coastal = xr.open_dataarray(fr'.\ncep_AEP_WSE_coastal.nc')
da_runoff = xr.open_dataarray(fr'.\ncep_AEP_WSE_runoff.nc')

# Combine into a single da to attribute
da_list = [da_compound, da_coastal, da_runoff]
aep_da = xr.concat(objs=da_list, dim='scenario')
aep_da['scenario'] = xr.IndexVariable(dims='scenario', data=['compound','coastal','runoff'])

attr_list = []
for rp in aep_da['return_period'].values.tolist():
    aep_rp = aep_da.sel(return_period=rp)
    da_diff = calc_diff_in_zsmax_compound_minus_max_individual(da=aep_rp)
    da_attr = classify_zsmax_by_process(da=aep_rp, da_diff=da_diff, hmin=0.05)
    attr_list.append(da_attr)

aep_attr_da = xr.concat(objs=attr_list, dim='return_period')
aep_attr_da['return_period'] = xr.IndexVariable(dims='return_period', data=aep_da['return_period'].values.tolist())
aep_attr_da.to_netcdf(r'.\attribution\ncep_AEP_WSE_attribution_v2.nc')

v1 = xr.open_dataset(r'.\attribution\ncep_AEP_WSE_attribution.nc')

xr.testing.assert_equal(v1, aep_attr_da)



