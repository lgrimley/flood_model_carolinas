import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydromt
import os
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

wdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_MODEL_OUTPUTS'
os.chdir(wdir)

'''' Load in the data '''
# Load the water level data for the historical return periods
histdir = r'.\ncep\aep\probabilistic_WSE'
file_paths = [os.path.join(histdir, file) for file in os.listdir(histdir) if ('AEP' in file) & (file.endswith('.nc'))]
scenarios = [file.split('_')[-1].split('.')[0] for file in os.listdir(histdir) if ('AEP' in file) & (file.endswith('.nc'))]
da_list = [xr.open_dataarray(file, engine='netcdf4') for file in file_paths]
h_aep = xr.concat(objs=da_list, dim='scenario')
h_aep['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

# Load the water level data for the projected return periods
projdir = r'.\canesm_ssp585\aep\probabilistic_WSE'
file_paths = [os.path.join(projdir, file) for file in os.listdir(projdir) if ('AEP' in file) & (file.endswith('.nc'))]
da_list = [xr.open_dataarray(file, engine='netcdf4') for file in file_paths]
p_aep = xr.concat(objs=da_list, dim='scenario')
p_aep['scenario'] = xr.IndexVariable(dims='scenario', data=scenarios)

##################################################################################################################
# Load model data/DEM
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml])
root = r'..\03_MODEL_RUNS\sfincs_initcond_mod'
mod = SfincsModel(root, data_libs=data_catalog_yml, mode='r')

# Data specifics
chunks_size = {'x': 5000, 'y': 5000}
res = 200
hmin = 0.05
elevation_file = rf'../03_MODEL_RUNS/subgrid/dep_subgrid_{res}m.tif'
dem = cat.get_rasterdataset(elevation_file, chunks=chunks_size)
water_mask = rf'./masks/water_mask_sbgRes{res}m.tif'
wb_mask = cat.get_rasterdataset(water_mask, chunks=chunks_size)
wb_mask.rio.write_crs(32617, inplace=True)

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

# Plotting details
wkt = dem.raster.crs.to_wkt()
utm_zone = dem.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

''' Plot AEP difference between Historic and Projected '''
sel_rp = [10,25,100, 250]
d_plot = []
df_combined = pd.DataFrame()
df_combined_ids = []
for T in sel_rp:
    h = h_aep.sel(scenario='compound', return_period=T)
    h_depth = (h - dem).compute()
    h_depth = h_depth.where(h_depth > 0.05)

    p = p_aep.sel(scenario='compound', return_period=T)
    p_depth = (p - dem).compute()
    p_depth = p_depth.where(p_depth > 0.05)

    diff = (p_depth - h_depth).compute()
    name = f'diff_depth_{T}'

    mask = (wb_mask != 1)
    diff = diff.where(mask).compute()
    diff = diff.where(diff > 0.05).compute()

    d_plot = d_plot + [h_depth, p_depth, diff]


# Custom white-only colormap
white_cmap = ListedColormap(['white'])
outdir = r'.\05_ANALYSIS\aep'
plot_AEP_depth_climate_comparsion = True
if plot_AEP_depth_climate_comparsion is True:
    # Plotting
    rp = (1/ np.array(sel_rp))*100
    scenarios = ['Historic (1980-2005)', 'Projected (2070-2100)', 'Overland Depth\nDifference']
    nrow = len(rp)
    ncol = len(scenarios)
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6.5, 6),
                             subplot_kw={'projection': utm},tight_layout=True, layout='constrained')
    axes = axes.flatten()
    for i in range(len(axes)):
        ax = axes[i]
        data = d_plot[i]
        if i in last_in_row:
            ckwargs = dict(cmap='Reds', vmin=0.05, vmax=1.5)
            cs2 = data.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=3)
            # ckwargs = dict(cmap=white_cmap, vmin=1, vmax=1)
            # mask = wb_mask.where(wb_mask == 1)
            # mask.plot(ax=ax, add_colorbar=False, zorder=2, **ckwargs)
            mod.region.plot(ax=ax, color='white', edgecolor='none', zorder=0, alpha=0)
            #major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.5, zorder=2, alpha=1)
            #nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.5, zorder=2, alpha=1)
            coastal_wb_clip.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5, zorder=2, alpha=1)
            mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=3, alpha=1)
        else:
            ckwargs = dict(cmap='Blues', vmin=0.05, vmax=6)
            cs = data.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=1)
            mod.region.plot(ax=ax, color='grey', edgecolor='none', zorder=0, alpha=0.6)
            mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=2, alpha=1)
        ax.set_axis_off()
        ax.set_title('')

        if i in first_row:
            ax.set_title(scenarios[i], loc='center', fontsize=10)
        for i in range(len(first_in_row)):
            label = f'{rp[i]}%\n({sel_rp[i]}-yr)'
            axes[first_in_row[i]].text(-0.05, 0.5, label,
                                       horizontalalignment='right',
                                       verticalalignment='center',
                                       rotation='horizontal',
                                       transform=axes[first_in_row[i]].transAxes)

    label = 'Depth (m)'
    ax = axes[2]
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.12, pos0.y0 + pos0.height * -0.4, 0.02, pos0.height * 1.5])
    cbar2 = fig.colorbar(cs,
                         cax=cax,
                         orientation='vertical',
                         label=label,
                         ticks=[0.05, 2, 4, 6, 8],
                         extend='max'
                         )

    label = 'Depth Difference (m)'
    ax = axes[8]
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.12, pos0.y0 + pos0.height * -0.8, 0.02, pos0.height * 1.5])
    cbar2 = fig.colorbar(cs2,
                         cax=cax,
                         orientation='vertical',
                         label=label,
                         ticks=[0.05, 0.5, 1, 1.5],
                         extend='max'
                         )

    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\03_AEP_floodmaps_compound\1pct_aep_flood_depths_historic_projected_difference.jpg', bbox_inches='tight', dpi=300)
    plt.close()


plot_AEP_depth_1PCT = True
if plot_AEP_depth_1PCT is True:
    T=100
    h = h_aep.sel(scenario='compound', return_period=T)
    h_depth = (h - dem).compute()
    h_depth = h_depth.where(h_depth > 0.05)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6),
                             subplot_kw={'projection': utm},tight_layout=True, layout='constrained')

    ckwargs = dict(cmap='Blues', vmin=0.05, vmax=8)
    cs = h_depth.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=1)
    mod.region.plot(ax=ax, color='grey', edgecolor='none', zorder=0, alpha=0.8)
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=2, alpha=1)

    ax.set_axis_off()
    ax.set_title('')

    label = 'Depth (m)'
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.12, pos0.y0 + pos0.height * 0.01, 0.04, pos0.height * 1])
    cbar2 = fig.colorbar(cs,
                         cax=cax,
                         orientation='vertical',
                         label=label,
                         ticks=[0.05, 2, 4, 6, 8],
                         extend='max'
                         )


    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\03_AEP_floodmaps_compound\1pct_aep_historic.jpg',
                bbox_inches='tight', dpi=300)
    plt.close()

