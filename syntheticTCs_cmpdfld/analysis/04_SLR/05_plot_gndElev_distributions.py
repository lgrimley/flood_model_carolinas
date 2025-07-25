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
from scipy import ndimage
import time

wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS'
os.chdir(wdir)

# Load model data
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
root = r'..\03_MODEL_RUNS\sfincs_initcond_mod'
mod = SfincsModel(root, data_libs=data_catalog_yml, mode='r')

def resized_gridded_output(da_source: xr.DataArray, da_target: xr.DataArray,
                           output_type: str='float32') -> xr.DataArray:
    start_time = time.time()
    target_shape = da_target.shape
    scaling_factors = [target_shape[i] / da_source.shape[i] for i in range(len(da_source.shape))]
    # add this if you want to include the tc_id dimension + [1.0]

    ra = ndimage.zoom(input=da_source, zoom=scaling_factors, order=1,
                      output=output_type, mode='grid-constant',
                      cval=np.nan, prefilter=False, grid_mode=True)
    rda = xr.DataArray(ra,
                       dims=da_source.dims,
                       coords={dim: np.linspace(da_source.coords[dim].min(), da_source.coords[dim].max(),
                                                target_shape[i]) for i, dim in enumerate(da_source.dims)},
                       attrs=da_source.attrs)
    rda['spatial_ref'] = da_source['spatial_ref']

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    return(rda)

res = 5
proj_crs = 32617
chunks_size = {'x': 5000, 'y': 5000}

data_filepath = r'..\05_ANALYSIS\05_SLR\version2_TCs_127\cmpd_freq_SLR.nc'
elevation_file = rf'..\03_MODEL_RUNS\subgrid\dep_subgrid_{res}m.tif'

data_da = cat.get_rasterdataset(data_filepath, crs=proj_crs,chunks=chunks_size)

elevation_da = cat.get_rasterdataset(elevation_file, crs=proj_crs, chunks=chunks_size)

rda = resized_gridded_output(da_source=data_da, da_target=elevation_da, output_type='int8')

# compound with slr
mask = (rda.data > 0)
elevation_da_masked = elevation_da.where(mask)
vals = elevation_da_masked.values
vals = vals[~np.isnan(vals)]
mean_val = np.mean(vals)
median_val = np.median(vals)
max_val = np.max(vals)
min_val = np.min(vals)
counts, bins = np.histogram(vals, bins=50, range=(-5, 20))

# compound without slr
mask2 = (rda.data < 0)
elevation_da_masked2 = elevation_da.where(mask2)
vals2 = elevation_da_masked2.values
vals2 = vals2[~np.isnan(vals2)]
mean_val2 = np.mean(vals2)
median_val2 = np.median(vals2)
max_val2 = np.max(vals2)
min_val2 = np.min(vals2)
counts2, bins2 = np.histogram(vals2, bins=50, range=(-5, 20))


# plot the histogram
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5))
ax.bar(bins[:-1], counts, width=np.diff(bins), align='edge', color='seagreen', edgecolor='black',
       label=f'Future w/ SLR', alpha=0.6)
ax.bar(bins2[:-1], counts2,width=np.diff(bins2), align='edge', color='darkgrey', edgecolor='black',
       label=f'Future w/out SLR')
# Add vertical lines for mean and median
# Calculate mean and median

ax.axvline(mean_val, color='darkgreen', linestyle='dashed', linewidth=2, label=f'w/ SLR Mean: {mean_val:.2f}')
#ax.axvline(median_val, color='seagreen', linestyle='solid', linewidth=1.5, label=f'w/ SLR Median: {median_val:.2f}')
ax.axvline(mean_val2, color='black', linestyle='dashed', linewidth=2, label=f'w/out SLR Mean: {mean_val2:.2f}')
#ax.axvline(median_val2, color='black', linestyle='solid', linewidth=1.5, label=f'w/out SLR Median: {median_val2:.2f}')

ax.legend()
ax.set_xlabel('Ground Elevations (m+NAVD88)')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\05_SLR\version2_TCs_127\ground_elev_histogram_compoundZones_SLR.png', dpi=300)
plt.close()

