import xarray as xr
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
import hydromt
import pandas as pd
from hydromt_sfincs import SfincsModel

os.chdir(rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS')

data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod = SfincsModel(root=r'..\03_MODEL_RUNS\sfincs_base_mod', mode='r', data_libs=data_catalog_yml)

##################################################################################################################
res = 5
dir = fr'.\canesm_ssp585\aep\probabilistic_WSE\floodmaps_{res}m'
chunks_size = {'x': 5000, 'y': 5000}

total = cat.get_rasterdataset(os.path.join(dir, f'canesm_ssp585_RP100_hmax_sbgRes{res}m.tif'), chunks=chunks_size)
runoff = cat.get_rasterdataset(os.path.join(dir, f'canesm_ssp585_RP100_runoff_hmax_sbgRes{res}m.tif'), chunks=chunks_size)
coastal = cat.get_rasterdataset(os.path.join(dir, f'canesm_ssp585_RP100_coastal_hmax_sbgRes{res}m.tif'), chunks=chunks_size)

# Combine the three scenarios
da = xr.concat([runoff, coastal, total], dim=xr.IndexVariable("scenario", ["runoff", "coastal", "compound"]))

# Get the max of the individual processes - runoff and coastal
da_single_max = da.sel(scenario=['runoff', 'coastal']).max('scenario')
depths = da_single_max.values
depths = depths[~np.isnan(depths)]
depths = pd.DataFrame(depths[depths != 0])
df = depths.describe(percentiles=[0.5, 0.9, 0.95])
mdf = df.T
mdf['Area_sqkm'] = (mdf['count'] * res * res) / (1000 ** 2)
mdf_da_single_max = mdf.round(3)


# Subtract the 1% AEP from each other
# Max individual
tot_minus_max  = total.fillna(0) - da_single_max.fillna(0)
depths = tot_minus_max.values
depths = depths[~np.isnan(depths)]
depths = pd.DataFrame(depths[depths != 0])
df = depths.describe(percentiles=[0.5, 0.9, 0.95])
mdf = df.T
mdf['Area_sqkm'] = (mdf['count'] * res * res) / (1000 ** 2)
mdf_tot_minus_max = mdf.round(3)

# Subtract the 1% AEP from each other
# Runoff
tot_minus_run = total.fillna(0) - runoff.fillna(0)
depths = tot_minus_run.values
depths = depths[~np.isnan(depths)]
depths = pd.DataFrame(depths[depths != 0])
df = depths.describe(percentiles=[0.5, 0.9, 0.95])
mdf = df.T
mdf['Area_sqkm'] = (mdf['count'] * res * res) / (1000 ** 2)
mdf_tot_minus_run = mdf.round(3)

# Subtract the 1% AEP from each other
# Coastal
tot_minus_coast = total.fillna(0) - coastal.fillna(0)
depths = tot_minus_coast.values
depths = depths[~np.isnan(depths)]
depths = pd.DataFrame(depths[depths != 0])
df = depths.describe(percentiles=[0.5, 0.9, 0.95])
mdf = df.T
mdf['Area_sqkm'] = (mdf['count'] * res * res) / (1000 ** 2)
mdf_tot_minus_coast = mdf.round(3)


# Load the water body mask dataarray
# water_mask = rf'./masks/water_mask_sbgRes{res}m.tif'
# wb_mask = cat.get_rasterdataset(water_mask, chunks=chunks_size)
# wb_mask.rio.write_crs(32617, inplace=True)
# wb_mask.name = 'wb_mask'