import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydromt
import os
import numpy as np
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS'
os.chdir(wdir)

# Load model data
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
root = r'..\03_MODEL_RUNS\sfincs_initcond_mod'
mod = SfincsModel(root, data_libs=data_catalog_yml, mode='r')

# Data specifics
chunks_size = {'x': 5000, 'y': 5000}
res= 5
rp= 100

da_hist = cat.get_rasterdataset(rf'.\ncep\aep\floodmaps_{res}m\ncep_RP{rp}_hmax_sbgRes{res}m.tif', chunks=chunks_size)
depths_hist = da_hist.values
depths = depths_hist[~np.isnan(depths_hist)]
counts_hist, bins_hist = np.histogram(depths, bins=100, range=(0.05, 5))
mean_val = np.mean(depths)
median_val = np.median(depths)

da_fut = cat.get_rasterdataset(rf'.\canesm_ssp585\aep\floodmaps_{res}m\canesm_ssp585_RP{rp}_hmax_sbgRes{res}m.tif', chunks=chunks_size)
depths_fut = da_fut.values
depths2 = depths_fut[~np.isnan(depths_fut)]
counts_fut, bins_fut = np.histogram(depths2, bins=100, range=(0.05, 5))
mean_val2 = np.mean(depths2)
median_val2 = np.median(depths2)

# plot the histogram
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5))
ax.bar(bins_fut[:-1], counts_fut, width=np.diff(bins_fut), align='edge', color='red', edgecolor='black',label='Future Period', alpha=0.8)
ax.bar(bins_hist[:-1], counts_hist,width=np.diff(bins_hist), align='edge', color='steelblue', edgecolor='black', label='Historic Period')
# Add vertical lines for mean and median
# Calculate mean and median

ax.axvline(mean_val2, color='r', linestyle='dashed', linewidth=1.5, label=f'Fut Mean: {mean_val2:.2f}')
ax.axvline(median_val2, color='r', linestyle='solid', linewidth=1.5, label=f'Fut Median: {median_val2:.2f}')
ax.axvline(mean_val, color='steelblue', linestyle='dashed', linewidth=1.5, label=f'Hist Mean: {mean_val:.2f}')
ax.axvline(median_val, color='steelblue', linestyle='solid', linewidth=1.5, label=f'Hist Median: {median_val:.2f}')

ax.legend()
ax.set_xlabel('Overland Flood Depths (m)')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\03_AEP_floodmaps_compound\overland_depth_histogram_1%floodHazard.png', dpi=300)
plt.close()



