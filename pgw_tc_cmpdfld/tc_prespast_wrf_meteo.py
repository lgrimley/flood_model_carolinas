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
        'size': 8}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 8})

yml = r'Z:\users\lelise\data\data_catalog.yml'
cat = hydromt.DataCatalog(yml)
os.chdir(r'Z:\users\lelise\projects\Carolinas\Chapter2\sfincs_models')
out_dir = os.path.join(os.getcwd(), '00_analysis')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

scenarios = [
    'matt_ensmean_present',
    'matt_ensmean_future_110cm',
    'flor_ensmean_present',
    'flor_ensmean_future_110cm'
]

scenarios_keys = [
    'Present\n WRF Ensemble Mean',
    # 'Future (4 degC) MSL 0.45m',
    'Future (4 degC)\nWRF Ensemble Mean'
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

    l_gdf = cat.get_geodataframe('carolinas_major_rivers')
    l_gdf.to_crs(epsg=32617, inplace=True)
    major_rivers = l_gdf.clip(basins)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\fris\bathy_v5\NHDArea_LowerPeeDee.shp')
    l_gdf.to_crs(epsg=32617, inplace=True)
    lpd_riv = l_gdf.clip(basins)

    # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
    # l_gdf = l_gdf[l_gdf['Name'].isin(
    #     ['Myrtle Beach', 'Wilmington', 'Raleigh'])]
    # l_gdf.set_index('Name', inplace=True)
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # cities = l_gdf  # .clip(basins)

    # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_reservoirs.shp')
    # l_gdf.set_index('Name', inplace=True)
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # reservoirs = l_gdf.clip(basins)

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

    l_gdf = cat.get_geodataframe('carolinas_coastal_wb')
    l_gdf.to_crs(epsg=32617, inplace=True)
    coastal_wb = l_gdf.clip(basins)

# Tracks
# nhc_track = cat.get_geodataframe(r'Z:\users\lelise\geospatial\tropical_cyclone\hurricane_tracks\IBTrACS.NA.list'
#                                  r'.v04r00.lines\IBTrACS.NA.list.v04r00.lines.shp')
# nhc_track.to_crs(epsg=32617, inplace=True)
# nhc_track = nhc_track[nhc_track['NAME'] == 'FLORENCE']
#
# wrf_track_dir = r'Z:\users\lelise\projects\Carolinas\Chapter2\wrf_output_20231006\flor_ensmean_txt_files'
# flor_lat = pd.read_table(os.path.join(wrf_track_dir, 'ensmean_minlat.txt'), header=None)
# flor_lon = pd.read_table(os.path.join(wrf_track_dir, 'ensmean_minlon.txt'), header=None)
# wrf_track = gpd.GeoDataFrame(data=None, geometry=gpd.points_from_xy(x=flor_lon.squeeze().to_list(),
#                                                                     y=flor_lat.squeeze().to_list(), crs=4326))
# wrf_track.to_crs(epsg=32617, inplace=True)
#
# flor_lat = pd.read_table(os.path.join(wrf_track_dir, 'ensmean_pgw_minlat.txt'), header=None)
# flor_lon = pd.read_table(os.path.join(wrf_track_dir, 'ensmean_pgw_minlon.txt'), header=None)
# wrf_pgw_track = gpd.GeoDataFrame(data=None, geometry=gpd.points_from_xy(x=flor_lon.squeeze().to_list(),
#                                                                         y=flor_lat.squeeze().to_list(), crs=4326))
# wrf_pgw_track.to_crs(epsg=32617, inplace=True)
# tracks = [nhc_track, wrf_track, wrf_pgw_track]

# UTM / Figure Extents
mod = SfincsModel(scenarios[0], mode='r')
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]

max_precip = []
mean_precip = []
max_ws = []
mean_ws = []
for i in range(len(scenarios)):
    mod = SfincsModel(scenarios[i], mode='r')
    # Load and calculate total precipitation
    precip = mod.forcing['precip_2d']
    precip_total = precip.sum(dim='time')
    precip_clipped = precip_total.rio.clip(mod.region.geometry, crs=mod.crs.to_epsg())
    d_max = round(precip_clipped.max().values.item(), 1)
    d_mean = round(precip_clipped.mean().values.item(), 1)
    max_precip.append(d_max)
    mean_precip.append(d_mean)

    wind_u = mod.forcing['wind_u']
    wind_v = mod.forcing['wind_v']
    wind_spd = ((wind_u ** 2) + (wind_u ** 2)) ** 0.5
    max_wind_spd = wind_spd.max(dim='time')
    wind_clipped = max_wind_spd.rio.clip(mod.region.geometry, crs=mod.crs.to_epsg())
    d_max = round(wind_clipped.max().values.item(), 1)
    d_mean = round(wind_clipped.mean().values.item(), 1)
    max_ws.append(d_max)
    mean_ws.append(d_mean)

meteo_df = pd.DataFrame()
meteo_df['scenarios'] = scenarios
meteo_df['mean_total_precip'] = mean_precip
meteo_df['max_total_precip'] = max_precip
meteo_df['mean_peak_windspd'] = mean_ws
meteo_df['max_peak_windspd'] = max_ws
meteo_df.to_csv(os.path.join(out_dir, 'domain_meteo_summary.csv'), index=False)

''' Meteo Plots '''
nrow = 2
ncol = len(scenarios)
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol - 1)
last_row = np.arange(ncol, n_subplots, 1)

# precip_bounds = np.arange(100, 600, 50)
# wind_bounds = np.arange(10, 40, 5)
# plot_total_meteo = False
# if plot_total_meteo is True:
#     fig, axs = plt.subplots(
#         nrows=nrow, ncols=ncol,
#         figsize=(12, 6),
#         subplot_kw={'projection': utm},
#         tight_layout=True,
#         layout='constrained',
#         sharex=True, sharey=True)
#     axs = axs.flatten()
#
#     for i in range(len(scenarios)):
#         mod = SfincsModel(scenarios[i], mode='r')
#         # Load and calculate total precipitation
#         precip = mod.forcing['precip_2d']
#         precip_total = precip.sum(dim='time')
#         wind_u = mod.forcing['wind_u']
#         wind_v = mod.forcing['wind_v']
#         wind_spd = ((wind_u ** 2) + (wind_u ** 2)) ** 0.5
#         max_wind_spd = wind_spd.max(dim='time')
#
#         # Precipitation!!!
#         ax = axs[i]
#         cmap = mpl.cm.jet
#         norm = mpl.colors.BoundaryNorm(precip_bounds, cmap.N, extend='both')
#         precip_total.plot(ax=ax,
#                           cmap=cmap,
#                           norm=norm,
#                           add_colorbar=False,
#                           zorder=2,
#                           alpha=0.8)
#         ax.set_title('')
#         ax.set_title(scenarios_keys[i], loc='center')
#         if i == 0:
#             tracks[i].plot(ax=ax, color='black', edgecolor='none', linewidth=2.5, linestyle='-', zorder=3, alpha=1)
#         else:
#             x = tracks[i].geometry.x
#             y = tracks[i].geometry.y
#             ax.plot(x, y, color='black', linewidth=2.5, linestyle='-', zorder=3, alpha=1)
#
#         # WIND!!!!!!
#         ax = axs[last_row[i]]
#         cmap2 = mpl.cm.YlOrBr
#         norm2 = mpl.colors.BoundaryNorm(wind_bounds, cmap2.N, extend='both')
#         max_wind_spd.plot(ax=ax,
#                           cmap=cmap2,
#                           norm=norm2,
#                           add_colorbar=False,
#                           zorder=2,
#                           alpha=0.8)
#         ax.set_title('')
#         if i == 0:
#             tracks[i].plot(ax=ax, color='black', edgecolor='none', linewidth=2.5, linestyle='-', zorder=3, alpha=1)
#         else:
#             x = tracks[i].geometry.x
#             y = tracks[i].geometry.y
#             ax.plot(x, y, color='black', linewidth=2.5, linestyle='-', zorder=3, alpha=1)
#
#     for ii in range(len(axs)):
#         minx, miny, maxx, maxy = basins.total_bounds
#         axs[ii].set_xlim(minx, maxx)
#         axs[ii].set_ylim(miny, maxy)
#         axs[ii].set_extent(extent, crs=utm)
#
#         riv_net_color = 'darkblue'
#         major_rivers.plot(ax=axs[ii], color=riv_net_color, edgecolor='none', linewidth=0.5,
#                           linestyle='-', zorder=2, alpha=1, label='Major Rivers')
#         lpd_riv.plot(ax=axs[ii], color=riv_net_color, edgecolor=riv_net_color, linewidth=0.5,
#                      linestyle='-', zorder=2, alpha=1)
#         basins.plot(ax=axs[ii], color='none', edgecolor='black', linewidth=1,
#                     linestyle='-', zorder=2, alpha=1, label='HUC6 Basins')
#
#         # Plot background/geography layers
#         if ii in first_in_row:
#             axs[ii].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
#             axs[ii].yaxis.set_visible(True)
#             axs[ii].ticklabel_format(style='sci', useOffset=False)
#
#         elif ii in last_row:
#             axs[ii].set_xlabel(f"X Coord UTM {utm_zone} (m)")
#             axs[ii].xaxis.set_visible(True)
#             axs[ii].ticklabel_format(style='sci', useOffset=False)
#
#     # Colorbar - Precip
#     pos1 = axs[last_in_row[0]].get_position()  # get the original position
#     cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.05, 0.02, pos1.height * 0.9])
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     cb = fig.colorbar(sm,
#                       cax=cbar_ax,
#                       shrink=0.7,
#                       extend='both',
#                       spacing='uniform',
#                       label='Total Precipitation\n(mm)',
#                       pad=0,
#                       aspect=10)
#
#     # Colorbar - Wind
#     pos1 = axs[last_in_row[1]].get_position()  # get the original position
#     cbar_ax2 = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.05, 0.02, pos1.height * 0.9])
#     sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
#     cb2 = fig.colorbar(sm2,
#                        cax=cbar_ax2,
#                        shrink=0.7,
#                        extend='both',
#                        spacing='uniform',
#                        label='Max Wind Speed\n(m/s)',
#                        pad=0,
#                        aspect=10)
#
#     # Save and close plot
#     plt.subplots_adjust(wspace=0.05, hspace=0.1)
#     plt.margins(x=0, y=0)
#     plt.savefig(os.path.join(out_dir, 'matthew_meteo_comparison.png'),
#                 bbox_inches='tight',
#                 dpi=255)
#     plt.close()

precip_bounds = np.arange(10, 50, 5)
wind_bounds = np.arange(5, 40, 5)
plot_meteo_at_each_timestep = True
mod1 = SfincsModel(scenarios[0], mode='r')
if plot_meteo_at_each_timestep is True:
    for t in range(0, len(mod1.forcing['precip_2d']['time']), 1):
        fout = os.path.join(out_dir, f'meteo_wrf_timestep_{t:03}.png')
        fig, axs = plt.subplots(
            nrows=nrow, ncols=ncol,
            figsize=(5, 4.3),
            subplot_kw={'projection': utm},
            tight_layout=True,
            layout='constrained',
            sharex=True, sharey=True)
        axs = axs.flatten()

        for i in range(len(scenarios)):
            mod = SfincsModel(scenarios[i], mode='r')

            # Load
            timestamp = mod.forcing['precip_2d']['time'][t]
            timestamp_str = pd.to_datetime(timestamp.to_pandas()).strftime('%b %d, %Y %H:%M')
            precip = mod.forcing['precip_2d'][t, :, :]
            wind_u = mod.forcing['wind_u'][t, :, :]
            wind_v = mod.forcing['wind_v'][t, :, :]
            wind_spd = ((wind_u ** 2) + (wind_u ** 2)) ** 0.5

            # Precipitation!!!
            ax = axs[i]
            cmap = mpl.cm.jet
            norm = mpl.colors.BoundaryNorm(precip_bounds, cmap.N, extend='both')
            precip.plot(ax=ax,
                        cmap=cmap,
                        norm=norm,
                        add_colorbar=False,
                        zorder=2,
                        alpha=0.8)
            ax.set_title('')
            ax.set_title(scenarios_keys[i], loc='center')
            # x = tracks[i+1].geometry.x
            # y = tracks[i+1].geometry.y
            # ax.plot(x, y, color='black', linewidth=2.5, linestyle='-', zorder=3, alpha=1)

            # WIND!!!!!!
            ax = axs[last_row[i]]
            cmap2 = mpl.cm.YlOrBr
            norm2 = mpl.colors.BoundaryNorm(wind_bounds, cmap2.N, extend='both')
            wind_spd.plot(ax=ax,
                          cmap=cmap2,
                          norm=norm2,
                          add_colorbar=False,
                          zorder=2,
                          alpha=0.8)
            ax.set_title('')

            # x = tracks[i+1].geometry.x
            # y = tracks[i+1].geometry.y
            # ax.plot(x, y, color='black', linewidth=2.5, linestyle='-', zorder=3, alpha=1)

        for ii in range(len(axs)):
            minx, miny, maxx, maxy = basins.total_bounds
            axs[ii].set_xlim(minx, maxx)
            axs[ii].set_ylim(miny, maxy)
            axs[ii].set_extent(extent, crs=utm)

            # Plot background/geography layers
            if ii in first_in_row:
                axs[ii].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
                axs[ii].yaxis.set_visible(False)
                axs[ii].ticklabel_format(style='sci', useOffset=False)
                riv_net_color = 'white'
            if ii in last_row:
                axs[ii].set_xlabel(f"X Coord UTM {utm_zone} (m)")
                axs[ii].xaxis.set_visible(False)
                axs[ii].ticklabel_format(style='sci', useOffset=False)
                riv_net_color = 'darkblue'

            major_rivers.plot(ax=axs[ii], color=riv_net_color, edgecolor='none', linewidth=0.25,
                              linestyle='-', zorder=2, alpha=1, label='Major Rivers')
            lpd_riv.plot(ax=axs[ii], color=riv_net_color, edgecolor=riv_net_color, linewidth=0.25,
                         linestyle='-', zorder=2, alpha=1)
            basins.plot(ax=axs[ii], color='none', edgecolor='black', linewidth=0.75,
                        linestyle='-', zorder=2, alpha=1, label='HUC6 Basins')

        # Colorbar - Precip
        pos1 = axs[last_in_row[0]].get_position()  # get the original position
        cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.05, 0.02, pos1.height * 0.9])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm,
                          cax=cbar_ax,
                          shrink=0.7,
                          extend='both',
                          spacing='uniform',
                          label='Rain Rate\n(mm/hr)',
                          pad=0,
                          aspect=10)

        # Colorbar - Wind
        pos1 = axs[last_in_row[1]].get_position()  # get the original position
        cbar_ax2 = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.05, 0.02, pos1.height * 0.9])
        sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
        cb2 = fig.colorbar(sm2,
                           cax=cbar_ax2,
                           shrink=0.7,
                           extend='both',
                           spacing='uniform',
                           label='Wind Speed\n(m/s)',
                           pad=0,
                           aspect=10)

        axs[3].set_title(timestamp_str, loc='right', fontsize=6)
        # Save and close plot
        fig.suptitle('Hurricane Matthew', fontsize=10, fontweight="bold")
        plt.subplots_adjust(wspace=0.03, hspace=0.03)
        plt.margins(x=0, y=0)
        plt.savefig(os.path.join(out_dir, fout),
                    bbox_inches='tight',
                    dpi=200)
        plt.close()

# plot_meteo_at_each_timestep = True
# mod1 = SfincsModel(scenarios[0], mode='r')
# # dep = cat.get_rasterdataset(hydromt.data_catalog.join('matt_ensmean_present', "gis", "dep.tif"))
# # hmin = 0.1
# if plot_meteo_at_each_timestep is True:
#     for t in range(0, len(mod1.forcing['precip_2d']['time']), 1):
#         fout = os.path.join(out_dir, f'meteo_wrf_timestep_{t:03}.png')
#         if os.path.isfile(fout):
#             continue
#         else:
#             fig, axs = plt.subplots(
#                 nrows=2, ncols=2,
#                 figsize=(6, 5),
#                 subplot_kw={'projection': utm},
#                 tight_layout=True,
#                 layout='constrained',
#                 sharex=True, sharey=True)
#             axs = axs.flatten()
#
#             for i in range(len(scenarios)):
#                 mod = SfincsModel(scenarios[i], mode='r')
#
#                 # Load and calculate total precipitation
#                 timestamp = mod.forcing['precip_2d']['time'][t]
#                 timestamp_str = pd.to_datetime(timestamp.to_pandas()).strftime('%b %d, %Y %H:%M')
#                 precip = mod.forcing['precip_2d'][t, :, :]
#                 wind_u = mod.forcing['wind_u'][t, :, :]
#                 wind_v = mod.forcing['wind_v'][t, :, :]
#                 wind_spd = ((wind_u ** 2) + (wind_u ** 2)) ** 0.5
#                 # zs_pres = mod.results["zs"].sel(time=timestamp, method='nearest')
#                 # hmax_pres = utils.downscale_floodmap(zsmax=zs_pres, dep=dep, hmin=hmin)
#
#                 # Precipitation!!!
#                 ckwargs = dict(cmap='jet', vmin=0, vmax=50)
#                 p = precip.plot(ax=axs[i],
#                                 add_colorbar=False,
#                                 zorder=2,
#                                 alpha=0.8,
#                                 **ckwargs
#                                 )
#                 # Plot background/geography layers
#                 major_rivers.plot(ax=axs[i], color='lightgrey', edgecolor='none', linewidth=0.5,
#                                   linestyle='-', zorder=3, alpha=1, label='Major Rivers')
#                 lpd_riv.plot(ax=axs[i], color='lightgrey', edgecolor='lightgrey', linewidth=0.5,
#                              linestyle='-', zorder=3, alpha=1)
#                 basins.plot(ax=axs[i], color='none', edgecolor='black', linewidth=1,
#                             linestyle='-', zorder=3, alpha=1, label='HUC6 Basins')
#
#                 minx, miny, maxx, maxy = basins.total_bounds
#                 axs[i].set_xlim(minx, maxx)
#                 axs[i].set_ylim(miny, maxy)
#                 axs[i].set_extent(extent, crs=utm)
#                 axs[i].set_title('')
#                 axs[i].set_title(scenarios_keys[i], loc='center')
#
#                 # WIND!!!!!!
#                 ckwargs = dict(cmap='afmhot_r', vmin=0, vmax=30)
#                 w = wind_spd.plot(ax=axs[i + 2],
#                                   add_colorbar=False,
#                                   zorder=2,
#                                   alpha=0.8,
#                                   **ckwargs)
#                 # Plot background/geography layers
#                 major_rivers.plot(ax=axs[i + 2], color='darkblue', edgecolor='none', linewidth=0.5,
#                                   linestyle='-', zorder=3, alpha=1, label='Major Rivers')
#                 lpd_riv.plot(ax=axs[i + 2], color='darkblue', edgecolor='darkblue', linewidth=0.5,
#                              linestyle='-', zorder=3, alpha=1)
#                 basins.plot(ax=axs[i + 2], color='none', edgecolor='black', linewidth=1,
#                             linestyle='-', zorder=3, alpha=1, label='HUC6 Basins')
#
#                 minx, miny, maxx, maxy = basins.total_bounds
#                 axs[i + 2].set_xlim(minx, maxx)
#                 axs[i + 2].set_ylim(miny, maxy)
#                 axs[i + 2].set_extent(extent, crs=utm)
#                 axs[i + 2].set_title('')
#
#                 # Depth!!!!!!
#                 # ckwargs = dict(cmap='Blues', vmin=0.1, vmax=8)
#                 # d = hmax_pres.plot(ax=axs[i + 4],
#                 #                    add_colorbar=False,
#                 #                    zorder=2,
#                 #                    alpha=0.8,
#                 #                    **ckwargs
#                 #                    )
#                 # # Plot background/geography layers
#                 # mod.region.plot(ax=axs[i + 4], color='lightgrey', edgecolor='black', linewidth=0.5,
#                 #                 linestyle='-', zorder=0, alpha=1)
#                 # basins.plot(ax=axs[i], color='none', edgecolor='black', linewidth=1,
#                 #             linestyle='-', zorder=3, alpha=1)
#                 #
#                 # minx, miny, maxx, maxy = basins.total_bounds
#                 # axs[i + 4].set_xlim(minx, maxx)
#                 # axs[i + 4].set_ylim(miny, maxy)
#                 # axs[i + 4].set_extent(extent, crs=utm)
#                 # axs[i + 4].set_title('')
#
#             axs[0].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
#             axs[0].yaxis.set_visible(True)
#
#             axs[2].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
#             axs[2].yaxis.set_visible(True)
#             axs[2].set_xlabel(f"X Coord UTM {utm_zone} (m)")
#             axs[2].xaxis.set_visible(True)
#
#             axs[3].set_xlabel(f"X Coord UTM {utm_zone} (m)")
#             axs[3].xaxis.set_visible(True)
#
#             for ax in axs:
#                 ax.ticklabel_format(style='sci', useOffset=False)
#                 ax.set_aspect('equal')
#
#                 cities.plot(ax=ax, marker='o', markersize=20, color="black",
#                             edgecolor='white', linewidth=0.5, label='Cities', zorder=4)
#
#                 for label, grow in cities.iterrows():
#                     x, y = grow.geometry.x, grow.geometry.y
#                     if label == 'Raleigh':
#                         ann_kwargs = dict(
#                             xytext=(-15, 5),
#                             textcoords="offset points",
#                             zorder=4,
#                             path_effects=[
#                                 patheffects.Stroke(linewidth=2, foreground="w", alpha=1),
#                                 patheffects.Normal(), ], )
#                     else:
#                         ann_kwargs = dict(
#                             xytext=(6, -15),
#                             textcoords="offset points",
#                             zorder=4,
#                             path_effects=[
#                                 patheffects.Stroke(linewidth=2, foreground="white", alpha=1),
#                                 patheffects.Normal(), ], )
#                     ax.annotate(f'{label}', xy=(x, y), **ann_kwargs)
#
#             # Colorbar - Precip
#             pos1 = axs[1].get_position()  # get the original position
#             cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.05, 0.02, pos1.height * 0.9])
#             cb = fig.colorbar(p,
#                               cax=cbar_ax,
#                               shrink=0.7,
#                               extend='both',
#                               spacing='uniform',
#                               label='Precipitation\n(mm/hr)',
#                               pad=0,
#                               aspect=10
#                               )
#
#             # Colorbar - Wind
#             pos1 = axs[3].get_position()  # get the original position
#             cbar_ax2 = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.05, 0.02, pos1.height * 0.9])
#             cb2 = fig.colorbar(w,
#                                cax=cbar_ax2,
#                                shrink=0.7,
#                                extend='both',
#                                spacing='uniform',
#                                label='Wind Speed\n(m/s)',
#                                pad=0,
#                                aspect=10
#                                )
#
#             # Colorbar - Depth
#             # pos1 = axs[5].get_position()  # get the original position
#             # cbar_ax3 = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.05, 0.02, pos1.height * 0.9])
#             # cb3 = fig.colorbar(d,
#             #                    cax=cbar_ax3,
#             #                    shrink=0.7,
#             #                    extend='both',
#             #                    spacing='uniform',
#             #                    label='Water Depth\n(m)',
#             #                    pad=0,
#             #                    aspect=10
#             #                    )
#
#             axs[3].set_title(timestamp_str, loc='right')
#             # Save and close plot
#             plt.subplots_adjust(wspace=0.05, hspace=0.05)
#             plt.margins(x=0, y=0)
#             plt.savefig(os.path.join(out_dir, f'meteo_wrf_florence_timestep_{t:03}.png'),
#                         bbox_inches='tight',
#                         dpi=190)
#             plt.close()
#
# if plot_mean_precip_ts is True:
#
#     for i in range(len(scenarios)):
#         mod = SfincsModel(scenarios[i], mode='r')
#         # Load and calculate total precipitation
#         da = mod.forcing['precip_2d'].transpose("time", ...)
#         da = da.mean(dim=[da.raster.x_dim, da.raster.y_dim])
#         df = da.to_pandas()
#         if i == 0:
#             mdf = df
#         else:
#             mdf = pd.concat([mdf, df], ignore_index=False, axis=1)
#         mdf.columns = scenarios_keys
#
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5), tight_layout=True)
#     mdf.plot(drawstyle="steps",
#              ax=ax,
#              linewidth=1.5)
#     plt.ylim([0, 5])
#     plt.ylabel('Mean Precipitation (mm/hr)')
#     plt.margins(x=0, y=0)
#     plt.savefig('precipitation_ts_wrf.png',
#                 bbox_inches='tight',
#                 dpi=255)
#     plt.close()
