import xarray as xr
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
import matplotlib.colors as mcolors
import pandas as pd

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

# Load SFINCS model
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
mod.read_grid()

''' Downscale WSE to 20m resolution '''
# Load model DEM
dem_filepath = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\subgrid\dep_subgrid_20m.tif'
#dem_20m = mod.data_catalog.get_rasterdataset(dem_filepath)
dem = mod.grid['dep']

# Load SFHA rasters
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS')
# sfha = xr.open_dataarray('Carolinas_SFHA_200m.nc')
# sfha.plot()

''' Setup plotting info  '''
# Load layers - run once because it takes a while...
coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
coastal_wb = coastal_wb.to_crs(mod.crs)
coastal_wb_clip = coastal_wb.clip(mod.region)

major_rivers = mod.data_catalog.get_geodataframe('carolinas_nhd_area_rivers')
major_rivers = major_rivers.to_crs(mod.crs)
major_rivers_clip = major_rivers.clip(mod.region)

nc_major_rivers = mod.data_catalog.get_geodataframe('carolinas_major_rivers')
nc_major_rivers = nc_major_rivers.to_crs(mod.crs)
nc_major_rivers_clip = nc_major_rivers.clip(mod.region)

wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\figures')

''' Plot AEP water levels for 3 processes for Historic and Projected'''
plot_AEP_WSE = False
if plot_AEP_WSE is True:
    # Plotting the 100-yr Water Levels
    T = 100
    rp = 1 / np.array(T)
    scenarios = ['Runoff', 'Coastal', 'Compound']
    periods = ['Historic (1980-2005)', 'Future (2070-2100)']
    ss = ['runoff', 'coastal', 'compound']
    nrow = len(scenarios)
    ncol = len(periods)
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)
    cmap = ListedColormap(['black'])
    ckwargs = dict(cmap='turbo', vmin=-5, vmax=60)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4.5, 5),
                             subplot_kw={'projection': utm}, tight_layout=True, layout='constrained')
    for i in range(len(scenarios)):
        scen = ss[i]
        # First column, historic
        ax = axes[i][0]
        h = h_aep.sel(scenario=scen, return_period=T)
        mask = (h > dem).compute()
        h = h.where(mask)
        cs = h.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=3, alpha=1)

        # Second column, projected
        ax = axes[i][1]
        p = p_aep.sel(scenario=scen, return_period=T)
        mask = (p > dem).compute()
        p = p.where(mask)
        cs = h.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=3,alpha=1)
    axes= axes.flatten()
    for i in range(len(axes)):
        ax = axes[i]
        mod.region.plot(ax=ax, color='white', edgecolor='none', zorder=0, alpha=0.6)
        mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=2, alpha=1)
        ax.set_axis_off()
        ax.set_title('')
        if i in first_row:
            ax.set_title(periods[i], loc='center', fontsize=10)
        for i in range(len(first_in_row)):
            axes[first_in_row[i]].text(-0.05, 0.5, scenarios[i],
                                       horizontalalignment='right',
                                       verticalalignment='center',
                                       rotation='vertical',
                                       transform=axes[first_in_row[i]].transAxes)
    label = 'Water Surface Elevation\n(m+NAVD88)'
    ax = axes[3]
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.15, pos0.y0 + pos0.height * -0.4, 0.03, pos0.height * 1.5])
    cbar2 = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='both')
    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(r'100_year_WSE.jpg', bbox_inches='tight', dpi=300)
    plt.close()


'''Plot the diff between compound and processes for 100-yr Water Levels'''
T = 100
rp = 1 / np.array(T)
scenarios = ['Runoff', 'Coastal', 'Max Individual']
periods = ['Historic (1980-2005)', 'Future (2070-2100)']
ss = ['runoff', 'coastal', 'both']
h = h_aep.sel(return_period=T)
p = p_aep.sel(return_period=T)
d_plot = []
for scen in ss:
    if scen == 'both':
        da_single_max = h.sel(scenario=['runoff', 'coastal']).max('scenario')
    else:
        da_single_max = h.sel(scenario=scen)

    # Now get water levels that are above ground for single process - historic
    da_single_max_depth = (da_single_max - dem).compute()
    mask = da_single_max > dem
    da_single_max_depth = da_single_max_depth.where(mask).fillna(0)

    # Now get water levels that are above ground for compound process - historic
    da_compound = h.sel(scenario='compound')
    da_compound_depth = (da_compound-dem).compute()
    mask = da_compound > dem
    da_compound_depth = da_compound_depth.where(mask).fillna(0)

    # Calculate the difference between single process and compound - historic
    da_diff_h = (da_compound_depth - da_single_max_depth).compute()
    mask = abs(da_diff_h) > 0.05
    da_diff_h = da_diff_h.where(mask)
    d_plot.append(da_diff_h)
    print(f'Done with historic {scen}')

    # Now do the same thing as above but for future
    if scen == 'both':
        da_single_max = p.sel(scenario=['runoff', 'coastal']).max('scenario')
    else:
        da_single_max = p.sel(scenario=scen)

    # Now get water levels that are above ground for single process - future
    da_single_max_depth = (da_single_max - dem).compute()
    mask = da_single_max > dem
    da_single_max_depth = da_single_max_depth.where(mask).fillna(0)

    # Now get water levels that are above ground for compound process - future
    da_compound = p.sel(scenario='compound')
    da_compound_depth = (da_compound - dem).compute()
    mask = da_compound > dem
    da_compound_depth = da_compound_depth.where(mask).fillna(0)

    # Calculate the difference between single process and compound - future
    da_diff_p = (da_compound_depth - da_single_max_depth).compute()
    mask = abs(da_diff_p) > 0.05
    da_diff_p = da_diff_p.where(mask)
    d_plot.append(da_diff_p)
    print(f'Done with projected {scen}')

# Calculate area/depth difference for the different 100-yr flood maps
wb_mask = mod.data_catalog.get_rasterdataset(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\waterbody_mask.nc')
df_combined = pd.DataFrame()
for i in range(len(d_plot)):
    d = d_plot[i].where(wb_mask == 1)
    hname = f'depth_diff_{i}'
    hda = xr.DataArray(d, name=hname)
    hdf = pd.DataFrame(hda.to_dataframe()[hname].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]), columns=[hname])
    df_combined = pd.concat(objs=[df_combined, hdf], axis=1, ignore_index=True)
df_combined.columns = ['runoff_hist', 'runoff_fut','coast_hist','coast_fut','max_hist', 'max_fut']
df_combined.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\100yr_CompMinusIndiv_stats_v2.csv' )


'''Plot the diff between compound and processes for 100-yr Water Levels'''
plot_AEP_WSE_diff = True
if plot_AEP_WSE_diff is True:
    # Setup plot specifics
    nrow = len(scenarios)
    ncol = len(periods)
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

    # Now create the plot
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4.5, 5), subplot_kw={'projection': utm},
                             tight_layout=True, layout='constrained')
    axes = axes.flatten()
    for i in range(len(axes)):
        ax = axes[i]
        data = d_plot[i]
        cs = data.plot(ax=ax, add_colorbar=False, cmap=cmap, norm=norm, zorder=3, alpha=1)

        mod.region.plot(ax=ax, color='white', edgecolor='none', zorder=0, alpha=1)
        major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=1)
        nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
        coastal_wb_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
        #ax.plot(*polygon.exterior.xy, color='black', linewidth=1.5, zorder=3, alpha=1)
        mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.35, zorder=2, alpha=1)
        ax.set_axis_off()
        ax.set_title('')
        if i in first_row:
            ax.set_title(periods[i], loc='center', fontsize=10)
        for i in range(len(first_in_row)):
            axes[first_in_row[i]].text(-0.05, 0.5, scenarios[i],
                                       horizontalalignment='right',
                                       verticalalignment='center',
                                       rotation='vertical',
                                       transform=axes[first_in_row[i]].transAxes)
    label = 'Water Level Difference (m)\ncompound - individual'
    ax = axes[3]
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.12, pos0.y0 -0.1, 0.03, pos0.height * 1.75])
    cbar = fig.colorbar(cs,
                        cax=cax,
                        orientation='vertical',
                        label=label,
                        extend='max')
    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(r'100_year_WSE_compoundDiff_v2.jpg', bbox_inches='tight', dpi=300)
    plt.close()


import shapely
#extent = [730599.1620, 3745648.4726, 809931.1997, 3832420.8957]
extent = [809506.8881,3837554.3556,904183.9238,3960339.8379]
polygon = shapely.geometry.box(*extent)
plot_AEP_WSE_diff = True
if plot_AEP_WSE_diff is True:
    # Now create the plot
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4, 8), subplot_kw={'projection': utm},
                             tight_layout=True, layout='constrained')
    axes = axes.flatten()
    for i in range(len(axes)):
        ax = axes[i]
        data = d_plot[i]
        cs = data.plot(ax=ax, add_colorbar=False, cmap=cmap, norm=norm, zorder=3, alpha=1)

        mod.region.plot(ax=ax, color='white', edgecolor='none', zorder=0, alpha=1)
        major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=1)
        nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
        coastal_wb_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
        ax.plot(*polygon.exterior.xy, color='black', linewidth=1.5, zorder=3, alpha=1)
        mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.35, zorder=2, alpha=1)
        ax.set_axis_off()
        ax.set_title('')
        if i in first_row:
            ax.set_title(periods[i], loc='center', fontsize=10)
        for i in range(len(first_in_row)):
            axes[first_in_row[i]].text(-0.05, 0.5, scenarios[i],
                                       horizontalalignment='right',
                                       verticalalignment='center',
                                       rotation='vertical',
                                       transform=axes[first_in_row[i]].transAxes)
        minx, miny, maxx, maxy = extent
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    label = 'Water Level Difference (m)\ncompound - individual'
    ax = axes[3]
    pos0 = ax.get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.12, pos0.y0 -0.1, 0.03, pos0.height * 1.5])
    cbar = fig.colorbar(cs,
                        cax=cax,
                        orientation='vertical',
                        label=label,
                        extend='both')
    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(r'100_year_WSE_compoundDiff_downeast.jpg', bbox_inches='tight', dpi=300)
    plt.close()

