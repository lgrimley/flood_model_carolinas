import os
import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydromt
import os
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns
import cartopy.crs as ccrs
from src.utils import  classify_zsmax_by_process, calculate_flooded_area_by_process
from matplotlib.colors import ListedColormap

from hydromt_sfincs import SfincsModel
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()


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


# Plotting details
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

''' Plot AEP difference between Historic and Projected '''
sel_rp = [10, 25, 100, 250]
d_plot = []
df_combined = pd.DataFrame()
df_combined_ids = []
for T in sel_rp:
    h = h_aep.sel(scenario='compound', return_period=T)
    h_depth = (h - dem).compute()
    h_depth = h_depth.where(h_depth > 0.05)
    hname = f'hist_depth_{T}'
    hda = xr.DataArray(h_depth, name=hname)
    hdf = pd.DataFrame(hda.to_dataframe()[hname].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]), columns=[hname])

    p = p_aep.sel(scenario='compound', return_period=T)
    p_depth = (p - dem).compute()
    p_depth = p_depth.where(p_depth > 0.05)
    pname = f'fut_depth_{T}'
    pda = xr.DataArray(p_depth, name=pname)
    pdf = pd.DataFrame(pda.to_dataframe()[pname].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]), columns=[pname])

    diff = (p_depth - h_depth).compute()
    name = f'diff_depth_{T}'
    da = xr.DataArray(diff, name=name)
    df = pd.DataFrame(da.to_dataframe()[name].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]), columns=[name])

    df_combined_ids = df_combined_ids + [hname, pname, name]
    df_combined = pd.concat(objs=[df_combined, hdf, pdf, df], axis=1, ignore_index=True)

    d_plot = d_plot + [h_depth, p_depth, diff]
df_combined.columns = df_combined_ids
# df_combined.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\aep_depth_diff_stats.csv' )


plot_AEP_depth_climate_comparsion = False
if plot_AEP_depth_climate_comparsion is True:
    # Plotting
    os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\figures')
    rp = (1/ np.array(sel_rp))*100
    scenarios = ['Historic (1980-2005)', 'Projected (2070-2100)', 'Difference']
    nrow = len(rp)
    ncol = len(scenarios)
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6, 6),
                             subplot_kw={'projection': utm},tight_layout=True, layout='constrained')
    axes = axes.flatten()
    for i in range(len(axes)):
        ax = axes[i]
        data = d_plot[i]
        if i in last_in_row:
            ckwargs = dict(cmap='seismic', vmin=-1, vmax=1)
            cs2 = data.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=1)
        else:
            ckwargs = dict(cmap='Blues', vmin=0.05, vmax=6)
            cs = data.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=1)
        mod.region.plot(ax=ax, color='grey', edgecolor='none', zorder=0, alpha=0.6)
        mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=2, alpha=1)
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
                         #ticks=[0, 0.5, 1, 1.5],
                         extend='both')

    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(r'aep_flood_depths_historic_projected_difference.jpg', bbox_inches='tight', dpi=300)
    plt.close()









