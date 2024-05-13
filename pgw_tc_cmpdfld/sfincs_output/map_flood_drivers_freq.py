import os
import glob
import hydromt
from hydromt import DataCatalog
import xarray as xr
import numpy as np
from os.path import join
import geopandas as gpd
import pandas as pd
from hydromt_sfincs import SfincsModel, utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors, patheffects
from string import ascii_lowercase as abcd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from matplotlib.ticker import FormatStrFormatter

font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

# Filepath to data catalogs yml
yml_BASE_Carolinas = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'
root = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_BASE_Carolinas])
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)

# Directory to output stuff
os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models\analysis')
fld_da_compound = xr.open_dataarray('pgw_drivers.nc')
pres_cmpda = []
fut_cmpda = []
storms = ['flor', 'flor', 'matt']
# Ensemble means
for storm in storms:
    # Present WRF
    runs = [f'{storm}_pres_ensmean']
    ds_ensmean_pres = fld_da_compound.sel(run=runs).sum(dim='run')
    pres_cmpda.append(ds_ensmean_pres)

    # Present WRF Scaled
    slr_runs = [f'{storm}_presScaled_ensmean_SLR{i}' for i in np.arange(1, 6, 1)]
    ds_ensmean_fut = fld_da_compound.sel(run=slr_runs).sum(dim='run')
    fut_cmpda.append(ds_ensmean_fut)

# Ensemble members
for storm in storms:
    nruns = 8
    if storm == 'matt':
        nruns = 7

    # Present WRF
    sel_runs = [f'{storm}_pres_ens{i}' for i in np.arange(1, nruns, 1)]
    ds_pres = get_compound_flood_extent(runs=sel_runs, da=fld_da_compound, value=1,
                                        name=f'{storm}_pres_members', write_tif=False)
    # Present WRF Scaled
    sel_runs = [f'{storm}_presScaled_ens{i}' for i in np.arange(1, nruns, 1)]
    slr_runs = []
    for run in sel_runs:
        slr_runs = slr_runs + [f'{run}_SLR{i}' for i in np.arange(1, 6, 1)]
    ds_presScaled = get_compound_flood_extent(runs=slr_runs, da=fld_da_compound, value=3,
                                              name=f'{storm}_presScaled_members', write_tif=False)

    # diff == 3 is compound in the future only
    # diff == 2 is compound in the future and present
    # diff == 1 is compound in the present only
    diff = (ds_presScaled - ds_pres).compute()
    diff = xr.where(diff == -1, 1, diff)
    diff.name = f'{storm}_preScaled_minus_pres_members'
    da_cmpd_diff_list.append(diff)

# PLOTTING COMPOUND FLOOD EXTENT (BINARY)
run_id = ['Florence', 'Floyd', 'Matthew', '', '', '']
ncol = 3
nrow = 1
# cmap = mpl.colors.ListedColormap(['magenta', 'royalblue', 'darkorange'])
# bounds = [1, 2, 3, 4]
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
extent = np.array(mod.region.buffer(10).total_bounds)[[0, 2, 1, 3]]

## Load layers
# coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
# coastal_wb = coastal_wb.to_crs(mod.crs)
# coastal_wb_clip = coastal_wb.clip(mod.region)
#
# major_rivers = mod.data_catalog.get_geodataframe('carolinas_nhd_area_rivers')
# major_rivers = major_rivers.to_crs(mod.crs)
# major_rivers_clip = major_rivers.clip(mod.region)

plot_entire_domain = True
if plot_entire_domain is True:
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                             figsize=(6.5, 4), subplot_kw={'projection': utm},
                             sharex=True, sharey=True, tight_layout=True)
    axes = axes.flatten()
    for i in range(len(axes)):
        ax = axes[i]
        dsp = fut_cmpda[i].where(fut_cmpda[i] > 0)
        cs = dsp.plot(ax=ax, cmap='Reds',
                      #norm=norm,
                      add_colorbar=False, zorder=0, alpha=1)
        dsp = pres_cmpda[i].where(pres_cmpda[i] > 0)
        cs2 = dsp.plot(ax=ax, cmap='Blues',
                      #norm=norm,
                      add_colorbar=False, zorder=1, alpha=1)
        ax.set_title('')
        ax.set_title(f'{run_id[i]}')
    for ax in axes:
        # minx, miny, maxx, maxy = extent
        # ax.set_xlim(minx, maxx)
        # ax.set_ylim(miny, maxy)
        mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=2, alpha=1)
        # major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey',
        #                        linewidth=0.25, zorder=0, alpha=0.75)
        # coastal_wb_clip.plot(ax=ax, color='slategrey', edgecolor='black',
        #                      linewidth=0.25, zorder=0, alpha=0.75)
        ax.set_aspect('equal')
        ax.set_axis_off()

    # Row 1
    axes[0].text(-0.05, 0.5, 'Ensemble Mean',
                 horizontalalignment='right',
                 verticalalignment='center',
                 rotation='vertical',
                 transform=axes[0].transAxes)
    pos0 = axes[2].get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.1, pos0.y0 + pos0.height * 0.02, 0.025, pos0.height * 0.9])
    cbar = fig.colorbar(cs,
                        cax=cax,
                        orientation='vertical',
                        extend='neither',
                        #ticks=[1.5, 2.5, 3.5],
                        # label='Ensemble Mean'
                        )
    #cbar.ax.set_yticklabels(['Present\n(n=1)', 'Both', 'Future\n(n=5)'])

    # Row 2
    # axes[3].text(-0.05, 0.5, 'Ensemble Members',
    #              horizontalalignment='right',
    #              verticalalignment='center',
    #              rotation='vertical',
    #              transform=axes[3].transAxes)
    # pos0 = axes[-1].get_position()  # get the original position
    # cax = fig.add_axes([pos0.x1 + 0.1, pos0.y0 + pos0.height * 0.02, 0.025, pos0.height * 0.9])
    # cbar = fig.colorbar(cs,
    #                     cax=cax,
    #                     orientation='vertical',
    #                     extend='neither',
    #                     ticks=[1.5, 2.5, 3.5],
    #                     # label='Ensemble Members'
    #                     )
    # cbar.ax.set_yticklabels(['Present\n(n=7-7-6)', 'Both', 'Future\n(n=35-35-30)'])

    plt.subplots_adjust(wspace=0, hspace=0.0, top=0.92)
    plt.margins(x=0, y=0)
    plt.suptitle('Compound Peak Flood Extent')
    plt.savefig(fr'test.png', dpi=225, bbox_inches="tight")
    plt.close()

nc_major_rivers = mod.data_catalog.get_geodataframe('carolinas_major_rivers')
nc_major_rivers = nc_major_rivers.to_crs(mod.crs)
nc_major_rivers_clip = nc_major_rivers.clip(mod.region)

plot_by_subbasin = True
if plot_by_subbasin is True:
    nrow = 2
    ncol = 3
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)

    basins = mod.data_catalog.get_geodataframe(
        r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU10.shp',
        bbox=mod.region.to_crs(4326).total_bounds)
    basins = basins.to_crs(mod.crs)
    # Subset/Clip geo data
    hucids = [
        # 'Headwaters White Oak River',
        'Newport River',
        # 'Jones Bay-Bay River',
        # 'Middle Trent River',
        # 'Outlet Waccamaw River-Atlantic Intracoastal Waterway',
        # 'Brunswick River-Cape Fear River',
        # 'Headwaters New River',
        # 'Town of Oriental-Neuse River',
        # 'Lower Trent River',
        # 'South Creek-Pamlico River'
    ]
    hucids_vertical = [
        'Outlet Waccamaw River-Atlantic Intracoastal Waterway',
        'Brunswick River-Cape Fear River',
        'Headwaters New River',
        'Town of Oriental-Neuse River',
        'South Creek-Pamlico River'
    ]
    for hucid in hucids:
        if hucid in hucids_vertical:
            figsize = (5, 6.5)
        else:
            figsize = (6.5, 4)
        extent = basins[basins['Name'] == hucid].buffer(5000).total_bounds
        major_rivers_clip = major_rivers.clip(extent)
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                 figsize=figsize, subplot_kw={'projection': utm},
                                 constrained_layout=True,
                                 sharex=True, sharey=True, tight_layout=True)
        axes = axes.flatten()
        for i in range(len(axes)):
            ax = axes[i]
            dsp = ds_plot[i].where(ds_plot[i] > 0)
            cs = dsp.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=2, alpha=0.9)
            ax.set_title('')
            ax.set_title(f'{run_id[i]}')
        for ax in axes:
            mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=3, alpha=0.9)
            major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey',
                                   linewidth=0.8, zorder=0, alpha=0.9)
            nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey',
                                      linewidth=0.8, zorder=0, alpha=0.9)
            coastal_wb_clip.plot(ax=ax, color='slategrey', edgecolor='black',
                                 linewidth=0.25, zorder=1, alpha=0.9)
            basins[basins['Name'] == hucid].plot(ax=ax, color='none', edgecolor='black',
                                                 linewidth=0.75, zorder=3, alpha=1)
            ax.set_aspect('equal')
            minx, miny, maxx, maxy = extent
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)

        # for ii in range(len(axes)):
        #     if ii in first_in_row:
        #         axes[ii].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
        #         axes[ii].yaxis.set_visible(True)
        #         axes[ii].ticklabel_format(style='sci', useOffset=False)
        #     if ii in last_row:
        #         axes[ii].set_xlabel(f"X Coord UTM {utm_zone} (m)")
        #         axes[ii].xaxis.set_visible(True)
        #         axes[ii].ticklabel_format(style='sci', useOffset=False)

        # Row 1
        axes[0].text(-0.05, 0.5, 'Ensemble Mean',
                     horizontalalignment='right',
                     verticalalignment='center',
                     rotation='vertical',
                     transform=axes[0].transAxes)
        pos0 = axes[2].get_position()  # get the original position
        cax = fig.add_axes([pos0.x1 + 0.03, pos0.y0, 0.05, pos0.height * 0.7])
        cbar = fig.colorbar(cs,
                            cax=cax,
                            orientation='vertical',
                            extend='neither',
                            ticks=[1.5, 2.5, 3.5],
                            # label='Ensemble Mean'
                            )
        cbar.ax.set_yticklabels(['Present\n(n=1)', 'Both', 'Future\n(n=5)'])

        # Row 2
        axes[3].text(-0.05, 0.5, 'Ensemble Members',
                     horizontalalignment='right',
                     verticalalignment='center',
                     rotation='vertical',
                     transform=axes[3].transAxes)
        pos0 = axes[-1].get_position()  # get the original position
        cax = fig.add_axes([pos0.x1 + 0.03, pos0.y0, 0.05, pos0.height * 0.7])
        cbar = fig.colorbar(cs,
                            cax=cax,
                            orientation='vertical',
                            extend='neither',
                            ticks=[1.5, 2.5, 3.5],
                            # label='Ensemble Members'
                            )
        cbar.ax.set_yticklabels(['Present\n(n=7-7-6)', 'Both', 'Future\n(n=35-35-30)'])

        plt.subplots_adjust(wspace=0, hspace=0, top=0.5)
        plt.margins(x=0, y=0)
        name = hucid
        plt.suptitle(f'Compound Peak Flood Extent:\n{name}')
        plt.savefig(fr'compound_flood_extent_{name}.png', dpi=225, bbox_inches="tight")
        plt.close()
