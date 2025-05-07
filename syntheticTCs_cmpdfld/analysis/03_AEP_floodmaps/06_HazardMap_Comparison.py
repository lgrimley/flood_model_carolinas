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
# Load the water level data for the historical return periods
histdir = fr'.\ncep\aep'
file_paths = [os.path.join(histdir, file) for file in os.listdir(histdir) if ('returnPeriods' in file) & (file.endswith('.nc'))]
scenarios = [file.split('_')[-1].split('.')[0] for file in os.listdir(histdir) if ('returnPeriods' in file) & (file.endswith('.nc'))]
da_list = [xr.open_dataarray(file, engine='netcdf4') for file in file_paths]
hist_aep = xr.concat(objs=da_list, dim='scenario')
hist_aep['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

# Load the water level data for the projected return periods
projdir = fr'.\canesm_ssp585\aep'
file_paths = [os.path.join(projdir, file) for file in os.listdir(projdir) if ('returnPeriods' in file) & (file.endswith('.nc'))]
da_list = [xr.open_dataarray(file, engine='netcdf4') for file in file_paths]
fut_aep = xr.concat(objs=da_list, dim='scenario')
fut_aep['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

rp = 100
res = 200

##################################################################################################################
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
mod = SfincsModel(root=r'..\03_MODEL_RUNS\sfincs_base_mod', mode='r', data_libs=data_catalog_yml)

elevation_file = rf'..\03_MODEL_RUNS\subgrid\dep_subgrid_{res}m.tif'
elevation_da = cat.get_rasterdataset(elevation_file)
water_mask = rf'./masks/water_mask_sbgRes{res}m.tif'
wb_mask = cat.get_rasterdataset(water_mask)

# Load layers - run once because it takes a while...
coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
coastal_wb = coastal_wb.to_crs(32617)
coastal_wb_clip = coastal_wb.clip(mod.region)

major_rivers = mod.data_catalog.get_geodataframe('carolinas_nhd_area_rivers')
major_rivers = major_rivers.to_crs(32617)
major_rivers_clip = major_rivers.clip(mod.region)

nc_major_rivers = mod.data_catalog.get_geodataframe('carolinas_major_rivers')
nc_major_rivers = nc_major_rivers.to_crs(32617)
nc_major_rivers_clip = nc_major_rivers.clip(mod.region)
##################################################################################################################
da = hist_aep
da_comp = da.sel(scenario='compound', return_period=rp)
da_comp = da_comp - elevation_da
da_comp = da_comp.where(da_comp > 0.05).compute()

da_run = da.sel(scenario='runoff', return_period=rp)
da_run = da_run - elevation_da
da_run = da_run.where(da_run > 0.05).compute()

da_coast = da.sel(scenario='coastal', return_period=rp)
da_coast = da_coast - elevation_da
da_coast = da_coast.where(da_coast > 0.05).compute()

da_single_max = da.sel(scenario=['runoff', 'coastal'], return_period=rp).max('scenario')
da_single_max = da_single_max - elevation_da
da_single_max = da_single_max.where(da_single_max > 0.05).compute()

diff_run = da_comp.fillna(0) - da_run.fillna(0)
diff_run = diff_run.where(wb_mask != 1)

diff_coast = da_comp.fillna(0) - da_coast.fillna(0)
diff_coast = diff_coast.where(wb_mask != 1)

diff_max = da_comp.fillna(0) - da_single_max.fillna(0)
diff_max = diff_max.where(wb_mask != 1)

hist_da = [diff_run, diff_coast, diff_max]
##################################################################################################################
da = fut_aep
da_comp = da.sel(scenario='compound', return_period=rp)
da_comp = da_comp - elevation_da
da_comp = da_comp.where(da_comp > 0.05).compute()

da_run = da.sel(scenario='runoff', return_period=rp)
da_run = da_run - elevation_da
da_run = da_run.where(da_run > 0.05).compute()

da_coast = da.sel(scenario='coastal', return_period=rp)
da_coast = da_coast - elevation_da
da_coast = da_coast.where(da_coast > 0.05).compute()

da_single_max = da.sel(scenario=['runoff', 'coastal'], return_period=rp).max('scenario')
da_single_max = da_single_max - elevation_da
da_single_max = da_single_max.where(da_single_max > 0.05).compute()

diff_run = da_comp.fillna(0) - da_run.fillna(0)
diff_run = diff_run.where(wb_mask != 1)

diff_coast = da_comp.fillna(0) - da_coast.fillna(0)
diff_coast = diff_coast.where(wb_mask != 1)

diff_max = da_comp.fillna(0) - da_single_max.fillna(0)
diff_max = diff_max.where(wb_mask != 1)

fut_da = [diff_run, diff_coast, diff_max]
##################################################################################################################
wkt = elevation_da.raster.crs.to_wkt()
utm_zone = elevation_da.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
##################################################################################################################
nrow = 3
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)
# Setup colorbar info
levels = [-0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5, 1, 2]
color_list2 = np.array([
    [51, 153, 255],
    [153, 204, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 153, 153],
    [255, 102, 102],
    [255, 0, 0],
    [204, 0, 0],
    [153, 0, 0],
]) / 255
cmap, norm = mpl.colors.from_levels_and_colors(levels, color_list2)
##################################################################################################################
periods = ['Historic (1980-2005)', 'Future (2070-2100)']
scenarios = ['Compound\nminus\nRunoff', 'Compound\nminus\nCoastal', 'Compound\nminus\nMax Individual']

fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6, 6), subplot_kw={'projection': utm})
for i in range(len(scenarios)):
    data1 = hist_da[i]
    data2 = fut_da[i]
    ax = axes[i][0]
    cs = data1.plot(ax=ax, add_colorbar=False, cmap=cmap, norm=norm, zorder=2)
    ax = axes[i][1]
    cs2 = data2.plot(ax=ax, add_colorbar=False, cmap=cmap, norm=norm, zorder=2)
axes =axes.flatten()
for i in range(len(axes)):
    ax= axes[i]
    # ckwargs = dict(cmap='Greys_r', vmin=1, vmax=1)
    # mask = wb_mask.where(wb_mask == 1)
    # mask.plot(ax=ax, add_colorbar=False, zorder=2, **ckwargs)
    mod.region.plot(ax=ax, color='white', edgecolor='none', zorder=0, alpha=0)
    major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.5, zorder=2, alpha=1)
    nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.5, zorder=2, alpha=1)
    coastal_wb_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.5, zorder=2, alpha=1)
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=3, alpha=1)
    ax.set_axis_off()
    ax.set_title('')
    if i in first_row:
        ax.set_title(periods[i], loc='center', fontsize=10)
    for i in range(len(first_in_row)):
        axes[first_in_row[i]].text(-0.05, 0.5, scenarios[i],
                                   horizontalalignment='right',
                                   verticalalignment='center',
                                   rotation='horizontal',
                                   transform=axes[first_in_row[i]].transAxes)
label = '1% AEP Water Level Difference (m)'
ax = axes[3]
pos0 = ax.get_position()  # get the original position
cax = fig.add_axes([pos0.x1 + 0.12, pos0.y0 -0.1, 0.03, pos0.height * 1.75])
cbar = fig.colorbar(cs,
                cax=cax,
                orientation='vertical',
                label=label,
                extend='max')
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.tight_layout()
plt.savefig(r'..\\05_ANALYSIS\03_AEP_floodmaps\1pctAEP_floodmap_comparison2.jpg', bbox_inches='tight', dpi=300)
plt.close()