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
import matplotlib.gridspec as gridspec

font = {'family': 'Arial',
        'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

yml = r'Z:\users\lelise\data\data_catalog.yml'
cat = hydromt.DataCatalog(yml)
os.chdir(r'Z:\users\lelise\projects\Carolinas\Chapter2\sfincs_models')
out_dir = os.path.join(os.getcwd(), '00_analysis')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

scenarios = [
    'flor_ensmean_present',
    'flor_ensmean_future_45cm'
]
scenarios_keys = [
    'Florence',
    'Matthew',
    'Future (4 degC)\n WRF Ensemble Mean MSL 1.10m'
]

# Plotting the data on a map with contextual layers
load_lyrs = True
if load_lyrs is True:
    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee', 'Upper Pee Dee'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    l_gdf.set_index('Name', inplace=True)
    basins = l_gdf
    #
    # l_gdf = cat.get_geodataframe('carolinas_major_rivers')
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # major_rivers = l_gdf.clip(basins)
    #
    # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\fris\bathy_v5\NHDArea_LowerPeeDee.shp')
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # lpd_riv = l_gdf.clip(basins)
    #
    # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
    # l_gdf = l_gdf[l_gdf['Name'].isin(
    #     ['Myrtle Beach', 'Wilmington', 'Raleigh'])]
    # l_gdf.set_index('Name', inplace=True)
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # cities = l_gdf  # .clip(basins)
    #
    # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_reservoirs.shp')
    # l_gdf.set_index('Name', inplace=True)
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # reservoirs = l_gdf.clip(basins)
    #
    # l_gdf = cat.get_geodataframe(
    #     r'Z:\users\lelise\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
    # l_gdf = l_gdf[l_gdf['NAME'].isin(['South Carolina', 'North Carolina'])]
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # l_gdf.set_index('NAME', inplace=True)
    # states = l_gdf

    # l_gdf = cat.get_geodataframe(
    #     r'Z:\users\lelise\geospatial\infrastructure\tl_2019_us_primaryroads\tl_2019_us_primaryroads.shp')
    # l_gdf = l_gdf[l_gdf['FULLNAME'].isin(['I- 95'])]
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # roads = l_gdf.clip(states.total_bounds)
    #
    # l_gdf = cat.get_geodataframe('carolinas_coastal_wb')
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # coastal_wb = l_gdf.clip(basins)
    #
    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(
        ['Myrtle Beach', 'Wilmington', 'Raleigh'])]
    l_gdf.set_index('Name', inplace=True)
    l_gdf.to_crs(epsg=32617, inplace=True)
    cities = l_gdf

# Tracks
nhc_track = cat.get_geodataframe(r'Z:\users\lelise\geospatial\tropical_cyclone\hurricane_tracks\IBTrACS.NA.list'
                                 r'.v04r00.lines\IBTrACS.NA.list.v04r00.lines.shp')
nhc_track.to_crs(epsg=32617, inplace=True)
nhc_track = nhc_track[nhc_track['NAME'] == 'MATTHEW']

wrf_track_dir = r'Z:\users\lelise\projects\Carolinas\Chapter2\wrf_output_20231006\matt_ensmean_txt_files'
flor_lat = pd.read_table(os.path.join(wrf_track_dir, 'ensmean_minlat.txt'), header=None)
flor_lon = pd.read_table(os.path.join(wrf_track_dir, 'ensmean_minlon.txt'), header=None)
wrf_track = gpd.GeoDataFrame(data=None, geometry=gpd.points_from_xy(x=flor_lon.squeeze().to_list(),
                                                                    y=flor_lat.squeeze().to_list(), crs=4326))
matt_wrf_track.to_crs(epsg=32617, inplace=True)

flor_lat = pd.read_table(os.path.join(wrf_track_dir, 'ensmean_pgw_minlat.txt'), header=None)
flor_lon = pd.read_table(os.path.join(wrf_track_dir, 'ensmean_pgw_minlon.txt'), header=None)
wrf_pgw_track = gpd.GeoDataFrame(data=None, geometry=gpd.points_from_xy(x=flor_lon.squeeze().to_list(),
                                                                        y=flor_lat.squeeze().to_list(), crs=4326))
matt_wrf_pgw_track.to_crs(epsg=32617, inplace=True)

tracks = [matt_wrf_track, matt_wrf_pgw_track]

''' stuff '''
mod = SfincsModel(scenarios[0], mode='r')
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]

dep = cat.get_rasterdataset(hydromt.data_catalog.join('matt_ensmean_present', "gis", "dep.tif"))
hmin = 0.1

''' Max Water Depth '''
plot_max_fldpth = True
if plot_max_fldpth is True:
    fig, axs = plt.subplots(
        nrows=1, ncols=2,
        figsize=(6.5, 3),
        subplot_kw={'projection': utm},
        tight_layout=True,
        layout='constrained',
        sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(len(scenarios)):
        mod = SfincsModel(scenarios[i], mode='r')
        # Load and calculate total precipitation
        mod.read_results()
        zsmax = mod.results["zsmax"].max(dim='timemax')

        # Downscale to subgrid
        hmax = utils.downscale_floodmap(
            zsmax=zsmax,
            dep=dep,
            hmin=hmin,
            gdf_mask=None,
        )
        ckwargs = dict(cmap='Blues', vmin=0.15, vmax=10)
        cs = hmax.plot(ax=axs[i],
                       add_colorbar=False,
                       zorder=2,
                       alpha=1,
                       **ckwargs)
        # Plot background/geography layers
        mod.region.plot(ax=axs[i], color='grey', edgecolor='none', zorder=1, alpha=1)
        mod.region.plot(ax=axs[i], color='none', edgecolor='black', linewidth=0.5,
                        linestyle='-', zorder=1, alpha=1)

        # x = tracks[i+1].geometry.x
        # y = tracks[i+1].geometry.y
        # axs[i].plot(x, y, color='black', linewidth=2, linestyle='-', zorder=3, alpha=0.8)

        minx, miny, maxx, maxy = basins.total_bounds
        axs[i].set_xlim(minx, maxx)
        axs[i].set_ylim(miny, maxy)
        axs[i].set_extent(extent, crs=utm)
        axs[i].set_title('')
        axs[i].set_title(scenarios_keys[i], loc='center')
        if i in [0]:
            axs[i].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
            axs[i].yaxis.set_visible(False)

        axs[i].set_xlabel(f"X Coord UTM {utm_zone} (m)")
        axs[i].xaxis.set_visible(False)
        axs[i].ticklabel_format(style='sci', useOffset=False)
        axs[i].set_aspect('equal')

    # Colorbar
    pos1 = axs[-1].get_position()  # get the original position
    cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.1, 0.02, pos1.height * 0.9])
    cb = fig.colorbar(cs,
                      cax=cbar_ax,
                      shrink=0.7,
                      extend='max',
                      spacing='uniform',
                      label='Max Water Depth (m)',
                      pad=0,
                      aspect=15
                      )
    st = np.append(0.1, np.arange(2, 11, 2))
    cb.set_ticks(st)
    fig.suptitle('Present WRF Ensemble\nMean MSL 0.13m', fontsize=10, fontweight="bold")
    # Save and close plot
    plt.subplots_adjust(wspace=0.03, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'max_water_depth_wrf_poster.png'),
                bbox_inches='tight',
                dpi=255)
    plt.close()

''' Plot difference in Max Water Level '''
scenarios = ['flor_ensmean_present', 'flor_ensmean_future_45cm', 'flor_ensmean_future_110cm']
mod = SfincsModel(scenarios[0], mode='r')
zsmax_pres = mod.results["zsmax"].max(dim='timemax')
hmax_pres = utils.downscale_floodmap(zsmax=zsmax_pres, dep=dep, hmin=hmin)

mod = SfincsModel(scenarios[1], mode='r')
zsmax_pgw_45cm = mod.results["zsmax"].max(dim='timemax')
hmax_pgw_45cm = utils.downscale_floodmap(zsmax=zsmax_pgw_45cm, dep=dep, hmin=hmin)

mod = SfincsModel(scenarios[2], mode='r')
zsmax_pgw_110cm = mod.results["zsmax"].max(dim='timemax')
hmax_pgw_110cm = utils.downscale_floodmap(zsmax=zsmax_pgw_110cm, dep=dep, hmin=hmin)

hmax_diff1 = (hmax_pgw_45cm - hmax_pres).compute()
hmax_diff2 = (hmax_pgw_110cm - hmax_pres).compute()
da_diff1 = hmax_diff1
da_diff2 = hmax_diff2

scenarios = ['matt_ensmean_present', 'matt_ensmean_future_45cm', 'matt_ensmean_future_110cm']
mod = SfincsModel(scenarios[0], mode='r')
zsmax_pres = mod.results["zsmax"].max(dim='timemax')
hmax_pres = utils.downscale_floodmap(zsmax=zsmax_pres, dep=dep, hmin=hmin)

mod = SfincsModel(scenarios[1], mode='r')
zsmax_pgw_45cm = mod.results["zsmax"].max(dim='timemax')
hmax_pgw_45cm = utils.downscale_floodmap(zsmax=zsmax_pgw_45cm, dep=dep, hmin=hmin)

mod = SfincsModel(scenarios[2], mode='r')
zsmax_pgw_110cm = mod.results["zsmax"].max(dim='timemax')
hmax_pgw_110cm = utils.downscale_floodmap(zsmax=zsmax_pgw_110cm, dep=dep, hmin=hmin)

hmax_diff1 = (hmax_pgw_45cm - hmax_pres).compute()
hmax_diff2 = (hmax_pgw_110cm - hmax_pres).compute()
da_diff3 = hmax_diff1
da_diff4 = hmax_diff2

scenarios = [da_diff1, da_diff3, da_diff2, da_diff4]
scenarios_keys = ['Florence', 'Matthew', '', '']
nrow = 2
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol - 1)
last_row = np.arange(ncol, n_subplots, 1)

plot_diff = True
if plot_diff is True:
    fig, axs = plt.subplots(
        nrows=nrow, ncols=ncol,
        figsize=(6.5, 5),
        subplot_kw={'projection': utm},
        tight_layout=True,
        layout='constrained',
        sharex=True, sharey=True)
    axs = axs.flatten()

    for i in range(len(scenarios)):
        ax = axs[i]
        ckwargs = dict(cmap='seismic', vmin=-2, vmax=2)
        cs = scenarios[i].plot(ax=axs[i], add_colorbar=False, **ckwargs, zorder=2)
        ax.set_title('')
        ax.set_title(scenarios_keys[i], loc='center')

    for ii in range(len(axs)):
        minx, miny, maxx, maxy = basins.total_bounds
        axs[ii].set_xlim(minx, maxx)
        axs[ii].set_ylim(miny, maxy)
        axs[ii].set_extent(extent, crs=utm)
        mod.region.plot(ax=axs[ii], color='lightgrey', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
        # Plot background/geography layers
        # if ii in first_in_row:
        # axs[ii].yaxis.set_visible(False)
        # axs[ii].ticklabel_format(style='sci', useOffset=False)

        # elif ii in last_row:
        # axs[ii].set_xlabel(f"X Coord UTM {utm_zone} (m)")
        # axs[ii].xaxis.set_visible(False)
        # axs[ii].ticklabel_format(style='sci', useOffset=False)

    # Colorbar - Precip
    label = 'Water Level Difference (m)\nFuture minus Present'
    pos0 = axs[-1].get_position()  # get the original position
    cax = fig.add_axes([pos0.x1 + 0.01, pos0.y0 + pos0.height * 0.5, 0.025, pos0.height * 0.9])
    cbar = fig.colorbar(cs,
                        cax=cax,
                        orientation='vertical',
                        label=label,
                        extend='both')

    # Save and close plot
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'wse_diff.png'),
                bbox_inches='tight',
                dpi=255)
    plt.close()

# plot_hmax_diff = True
# if plot_hmax_diff is True:
#     fig, axs = plt.subplots(
#         nrows=2, ncols=2,
#         figsize=(6.5, 4.5),
#         subplot_kw={'projection': utm},
#         tight_layout=True)
#     for i in range(len(axs)):
#         minx, miny, maxx, maxy = basins.total_bounds
#         axs[i].set_xlim(minx, maxx)
#         axs[i].set_ylim(miny, maxy)
#         axs[i].set_extent(extent, crs=utm)
#         axs[i].set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
#         axs[i].set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
#         axs[i].yaxis.set_visible(False)
#         axs[i].xaxis.set_visible(False)
#         axs[i].ticklabel_format(style='plain', useOffset=False)
#         axs[i].ticklabel_format(style='sci', useOffset=False)
#         axs[i].set_aspect('equal')
# 
#         # Plot difference in water level raster
#         ckwargs = dict(cmap='seismic', vmin=-2, vmax=2)
#         if i == 0:
#             cs1 = da_diff1.plot(ax=axs[i], add_colorbar=False, **ckwargs, zorder=2)
#             axs[i].set_title('')
#             axs[i].set_title(scenarios_keys[i + 1], loc='center')
#             axs[i].set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
#             axs[i].set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
#         elif i == 4:
#             cs2 = da_diff2.plot(ax=axs[i], add_colorbar=False, **ckwargs, zorder=2)
#             axs[i].yaxis.set_visible(False)
#             axs[i].set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
#             axs[i].set_title(scenarios_keys[i + 1], loc='center')
#             label = 'Water Level Difference (m)\nFuture minus Present'
#             pos0 = axs[i].get_position()  # get the original position
#             cax = fig.add_axes([pos0.x1 + 0.1, pos0.y0 + pos0.height * 0.1, 0.025, pos0.height * 0.9])
#             cbar = fig.colorbar(cs2,
#                                 cax=cax,
#                                 orientation='vertical',
#                                 label=label,
#                                 extend='both')
# 
#         # basins.plot(ax=axs[i], color='white', edgecolor='black',
#         #             linewidth=0.1, linestyle='-', zorder=0, alpha=0.25, hatch='xxx')
#         mod.region.plot(ax=axs[i], color='lightgrey', edgecolor='black', linewidth=0.5, zorder=1, alpha=1)
#         # basins.plot(ax=axs[i], color='none', edgecolor='black',
#         #             linewidth=0.75, linestyle='-', zorder=3, alpha=1)
#         # x = tracks[i+1].geometry.x
#         # y = tracks[i+1].geometry.y
#         # axs[i].plot(x, y, color='black', linewidth=2, linestyle='-', zorder=3, alpha=0.8)
# 
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.margins(x=0, y=0)
#     plt.savefig(os.path.join(out_dir, 'future_minus_present_hmax_diff.png'), dpi=225, bbox_inches="tight")
#     plt.close()
