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
os.chdir(
    r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models')
scenarios = [
    'flor_ensmean_present',
    'flor_ens1_present',
    'flor_ens2_present',
    'flor_ens3_present',
    'flor_ens4_present',
    'flor_ens5_present',
    'flor_ens6_present',
    'flor_ens7_present'
]
scenarios_keys = [
    'Ensmean',
    'Ens1',
    'Ens2',
    'Ens3',
    'Ens4',
    'Ens5',
    'Ens6',
    'Ens7'
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

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(
        ['Myrtle Beach', 'Wilmington', 'Raleigh'])]
    l_gdf.set_index('Name', inplace=True)
    l_gdf.to_crs(epsg=32617, inplace=True)
    cities = l_gdf  # .clip(basins)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_reservoirs.shp')
    l_gdf.set_index('Name', inplace=True)
    l_gdf.to_crs(epsg=32617, inplace=True)
    reservoirs = l_gdf.clip(basins)

    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
    l_gdf = l_gdf[l_gdf['NAME'].isin(['South Carolina', 'North Carolina'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    l_gdf.set_index('NAME', inplace=True)
    states = l_gdf

    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\infrastructure\tl_2019_us_primaryroads\tl_2019_us_primaryroads.shp')
    l_gdf = l_gdf[l_gdf['FULLNAME'].isin(['I- 95'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    roads = l_gdf.clip(states.total_bounds)

    l_gdf = cat.get_geodataframe('carolinas_coastal_wb')
    l_gdf.to_crs(epsg=32617, inplace=True)
    coastal_wb = l_gdf.clip(basins)

    # tc_tracks = cat.get_geodataframe(r'Z:\users\lelise\geospatial\tropical_cyclone\hurricane_tracks\IBTrACS.NA.list'
    #                                  r'.v04r00.lines\IBTrACS.NA.list.v04r00.lines.shp')
    # tc_tracks.to_crs(epsg=32617, inplace=True)
mod = SfincsModel(scenarios[0], mode='r')
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(10000).total_bounds)[[0, 2, 1, 3]]

''' Precipitation '''
if plot_total_precip is True:
    fig, axs = plt.subplots(
        nrows=2, ncols=4,
        figsize=(12, 6),
        subplot_kw={'projection': utm},
        tight_layout=True,
        layout='constrained',
        sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(len(scenarios)):
        mod = SfincsModel(scenarios[i], mode='r')
        # Load and calculate total precipitation
        precip = mod.forcing['precip_2d']
        precip_total = precip.sum(dim='time')

        # Precipitation!!!
        cmap = mpl.cm.jet
        bounds = np.linspace(start=100, stop=1200, num=12)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
        precip_total.plot(ax=axs[i],
                          cmap=cmap,
                          norm=norm,
                          add_colorbar=False,
                          zorder=2,
                          alpha=0.8
                          )
        # Plot background/geography layers
        major_rivers.plot(ax=axs[i], color='darkblue', edgecolor='none', linewidth=0.5,
                          linestyle='-', zorder=3, alpha=1, label='Major Rivers')
        lpd_riv.plot(ax=axs[i], color='darkblue', edgecolor='darkblue', linewidth=0.5,
                     linestyle='-', zorder=3, alpha=1)
        basins.plot(ax=axs[i], color='none', edgecolor='black', linewidth=1.5,
                    linestyle='-', zorder=3, alpha=1, label='HUC6 Basins')

        minx, miny, maxx, maxy = basins.total_bounds
        axs[i].set_xlim(minx, maxx)
        axs[i].set_ylim(miny, maxy)
        axs[i].set_extent(extent, crs=utm)
        axs[i].set_title('')
        axs[i].set_title(scenarios_keys[i], loc='center')
        if i in [0, 4]:
            axs[i].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
        if i >= 4:
            axs[i].set_xlabel(f"X Coord UTM {utm_zone} (m)")
            axs[i].xaxis.set_visible(True)

        axs[i].ticklabel_format(style='sci', useOffset=False)
        axs[i].set_aspect('equal')

    # Colorbar
    pos1 = axs[7].get_position()  # get the original position
    cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.3, 0.02, pos1.height * 1.6])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm,
                      cax=cbar_ax,
                      shrink=0.7,
                      extend='both',
                      spacing='uniform',
                      label='Total Precipitation (mm)',
                      pad=0,
                      aspect=30
                      )
    # Save and close plot
    plt.subplots_adjust(wspace=0.03, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig('precipitation_wrf.png',
                bbox_inches='tight',
                dpi=255)
    plt.close()

''' Wind '''
if plot_max_wndspd is True:
    fig, axs = plt.subplots(
        nrows=2, ncols=4,
        figsize=(12, 6),
        subplot_kw={'projection': utm},
        tight_layout=True,
        layout='constrained',
        sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(len(scenarios)):
        mod = SfincsModel(scenarios[i], mode='r')
        # Load and calculate total precipitation
        wind_u = mod.forcing['wind_u']
        wind_v = mod.forcing['wind_v']
        wind_spd = ((wind_u ** 2) + (wind_u ** 2)) ** 0.5
        max_wind_spd = wind_spd.max(dim='time')

        # Precipitation!!!
        cmap = mpl.cm.YlOrBr
        bounds = np.linspace(start=10, stop=55, num=10)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
        max_wind_spd.plot(ax=axs[i],
                          cmap=cmap,
                          norm=norm,
                          add_colorbar=False,
                          zorder=2,
                          alpha=0.8
                          )
        # Plot background/geography layers
        major_rivers.plot(ax=axs[i], color='darkblue', edgecolor='none', linewidth=0.5,
                          linestyle='-', zorder=3, alpha=1, label='Major Rivers')
        lpd_riv.plot(ax=axs[i], color='darkblue', edgecolor='darkblue', linewidth=0.5,
                     linestyle='-', zorder=3, alpha=1)
        basins.plot(ax=axs[i], color='none', edgecolor='black', linewidth=1.5,
                    linestyle='-', zorder=3, alpha=1, label='HUC6 Basins')

        minx, miny, maxx, maxy = basins.total_bounds
        axs[i].set_xlim(minx, maxx)
        axs[i].set_ylim(miny, maxy)
        axs[i].set_extent(extent, crs=utm)
        axs[i].set_title('')
        axs[i].set_title(scenarios_keys[i], loc='center')
        if i in [0, 4]:
            axs[i].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
        if i >= 4:
            axs[i].set_xlabel(f"X Coord UTM {utm_zone} (m)")
            axs[i].xaxis.set_visible(True)

        axs[i].ticklabel_format(style='sci', useOffset=False)
        axs[i].set_aspect('equal')

    # Colorbar
    pos1 = axs[7].get_position()  # get the original position
    cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.3, 0.02, pos1.height * 1.6])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm,
                      cax=cbar_ax,
                      shrink=0.7,
                      extend='both',
                      spacing='uniform',
                      label='Max Wind Speed (m/s)',
                      pad=0,
                      aspect=30
                      )
    # Save and close plot
    plt.subplots_adjust(wspace=0.03, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig('wind_wrf.png',
                bbox_inches='tight',
                dpi=255)
    plt.close()

''' Max Water Depth '''
if plot_max_fldpth is True:
    dep = cat.get_rasterdataset(hydromt.data_catalog.join('flor_ensmean_present', "gis", "dep.tif"))
    hmin = 0.05

    fig, axs = plt.subplots(
        nrows=2, ncols=4,
        figsize=(12, 6),
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

        minx, miny, maxx, maxy = basins.total_bounds
        axs[i].set_xlim(minx, maxx)
        axs[i].set_ylim(miny, maxy)
        axs[i].set_extent(extent, crs=utm)
        axs[i].set_title('')
        axs[i].set_title(scenarios_keys[i], loc='center')
        if i in [0, 4]:
            axs[i].set_ylabel(f"Y Coord UTM {utm_zone} (m)")
            axs[i].yaxis.set_visible(True)
        if i >= 4:
            axs[i].set_xlabel(f"X Coord UTM {utm_zone} (m)")
            axs[i].xaxis.set_visible(True)

        axs[i].ticklabel_format(style='sci', useOffset=False)
        axs[i].set_aspect('equal')

    # Colorbar
    pos1 = axs[7].get_position()  # get the original position
    cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.3, 0.02, pos1.height * 1.6])
    cb = fig.colorbar(cs,
                      cax=cbar_ax,
                      shrink=0.7,
                      extend='both',
                      spacing='uniform',
                      label='Max Water Depth (m)',
                      pad=0,
                      aspect=30
                      )
    # Save and close plot
    plt.subplots_adjust(wspace=0.03, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig('max_water_depth_flor.png',
                bbox_inches='tight',
                dpi=255)
    plt.close()

if plot_mean_precip_ts is True:

    for i in range(len(scenarios)):
        mod = SfincsModel(scenarios[i], mode='r')
        # Load and calculate total precipitation
        da = mod.forcing['precip_2d'].transpose("time", ...)
        da = da.mean(dim=[da.raster.x_dim, da.raster.y_dim])
        df = da.to_pandas()
        if i == 0:
            mdf = df
        else:
            mdf = pd.concat([mdf, df], ignore_index=False, axis=1)
        mdf.columns = scenarios_keys

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5), tight_layout=True)
    mdf.plot(drawstyle="steps",
             ax=ax,
             linewidth=1.5)
    plt.ylim([0, 5])
    plt.ylabel('Mean Precipitation (mm/hr)')
    plt.margins(x=0, y=0)
    plt.savefig('precipitation_ts_wrf.png',
                bbox_inches='tight',
                dpi=255)
    plt.close()

''' Model Boundary Conditions Figure '''
# bzs = mod.forcing['bzs'].vector.to_gdf()
# dis = mod.forcing['dis'].vector.to_gdf()
# gdf_msk = utils.get_bounds_vector(mod.grid["msk"])
# gdf_msk3 = gdf_msk[gdf_msk["value"] == 3]
#
# plt_timeseries = True
# if plt_timeseries is True:
#     fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 3.25), tight_layout=True, sharex=True)
#
#     # convert to Single index dataframe (bar plots don't work with xarray)
#     bnd_type = 'dis'
#     da = mod.forcing[bnd_type]
#     df = da.to_pandas().transpose()
#     df.index = mdates.date2num(df.index)
#     df.plot(ax=axs[0],
#             label=False)
#     locator = mdates.AutoDateLocator()
#     formatter = mdates.ConciseDateFormatter(locator)
#     axs[0].xaxis.set_major_locator(locator)
#     axs[0].xaxis.set_major_formatter(formatter)
#     # axs[0].set_yscale('log')
#     # axs[0].set_ylim((-1, 2000))
#     axs[0].set_ylabel('(m/s)')
#     # axs[0].set_aspect('equal')
#     axs[0].set_title('')
#     axs[0].set_title('USGS Discharge Boundary Condition', loc='left', fontsize=10)
#
#     legend_kwargs0 = dict(
#         ncol=3,
#         # bbox_to_anchor=(1.05, 1),
#         title="Legend",
#         loc="best",
#         frameon=False,
#         # prop=dict(size=10),
#     )
#     axs[0].legend(**legend_kwargs0)
#
#     bnd_type = 'bzs'
#     da = mod.forcing[bnd_type]
#     df = da.to_pandas().transpose()
#     df.index = mdates.date2num(df.index)
#     df.plot(ax=axs[1],
#             label=False,
#             legend=False)
#     axs[1].set_ylim((-2.25, 2.25))
#     axs[1].xaxis.set_major_locator(locator)
#     axs[1].xaxis.set_major_formatter(formatter)
#     axs[1].set_ylabel('m +NAVD88')
#     # axs[1].set_aspect('equal')
#     axs[1].set_title('')
#     axs[1].set_title('ADCIRC Water Level Boundary Condition', loc='left', fontsize=10)
#
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.margins(x=0, y=0)
#     plt.savefig(os.path.join(mod.root, 'figs', 'boundary_conditions_timeseries.png'), bbox_inches='tight',
#                 dpi=225)  # , pil_kwargs={'quality': 95})
#     plt.close()
