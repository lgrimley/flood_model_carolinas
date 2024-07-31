import os
import hydromt_sfincs
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import datetime as dt
import rioxarray as rio
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import cartopy.crs as ccrs


def calc_stats(observed, modeled):
    mae = abs(observed - modeled).values.mean()
    rmse = ((observed - modeled) ** 2).mean() ** 0.5
    bias = (modeled - observed).values.mean()
    return [round(mae, 2), round(rmse, 2), round(bias, 2)]


# Load in model and read results
cat_dir = r'Z:\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter1\sfincs\final_model')
model_root = 'ENC_200m_sbg5m_noChannels_avgN'
mod = SfincsModel(root=model_root, mode='r',
                  data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
cat = mod.data_catalog

out_dir = os.path.join(mod.root, 'validation', 'hwm')
hwm_locs = pd.read_csv(os.path.join(out_dir, 'hwm_error_all.csv'))
hwm_locs.drop(columns='geometry', inplace=True)
hwm = gpd.GeoDataFrame(hwm_locs, geometry=gpd.points_from_xy(x=hwm_locs['xcoords'], y=hwm_locs['ycoords'], crs=32617))

# Create a directory to save data and figures to
out_dir = os.path.join(os.getcwd(), model_root, 'validation', 'hwm')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
os.chdir(out_dir)

''' PLOTTING THE DATA '''
# Update plot fonts
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

# Plot a histogram of the HWM errors (Figure S5)
plot_hist_errors = True
if plot_hist_errors is True:
    hwm['elev_grp'] = 'xx'
    hwm['elev_grp'][hwm['elev_m'] <= 20] = 'Elevation <= 20m'
    hwm['elev_grp'][(hwm['elev_m'] > 20)] = 'Elevation > 20m'
    attr = 'elev_grp'

    s = len(hwm[attr].unique())

    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(3, 3))
    hwm.hist(column='error',
             ax=ax, by=attr,
             sharex=True, sharey=True,
             bins=40, grid=True,
             layout=(s, 1),
             color='darkgrey', rwidth=0.9,
             )
    plt.xlabel("Peak Water Level Error (m)\n Modeled minus Observed")
    plt.ylabel("Frequency")
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'peak_errors_wl_histogram.png'), bbox_inches='tight', dpi=225)
    plt.close()

    # Depth errors
    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(3, 3))
    hwm.hist(column='depth_error',
             ax=ax, by=attr,
             sharex=True, sharey=True,
             bins=40, grid=True,
             layout=(s, 1),
             color='darkgrey', rwidth=0.9,
             )
    plt.xlabel("Peak Depth Error (m)\nModeled minus Observed")
    plt.ylabel("Frequency")
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'peak_errors_depth_histogram.png'), bbox_inches='tight', dpi=225)
    plt.close()

# Plot a QQ of the HWM errors (Figure 4)
plot_qq_vert = True
if plot_qq_vert is True:
    nrow = 3
    ncol = 2
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)
    colors = ["lightgray", "gray", "darkgray", "gray", 'black']
    basin = ['Lower Pee Dee', 'Neuse', 'Cape Fear', 'Pamlico', 'Onslow Bay']
    legend_nick = ['LPD', 'N', 'CF', 'P', 'OB']
    marker = ["o", "^", "s", "d", "x"]

    # Axis limits for the subplots
    axlimits = [[0, 100], [0, 20], [20, 40]]
    stp = [20, 2, 5, 2, 5, 2]
    text_locator = []

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, layout='constrained', figsize=(5, 7))
    axs = axs.flatten()
    for i in range(len(axs)):
        if i in first_row:
            axislim = axlimits[0]
        elif i in last_row:
            axislim = axlimits[2]
        else:
            axislim = axlimits[1]

        subset = hwm[(hwm['elev_m'] > axislim[0]) & (hwm['elev_m'] <= axislim[1])]
        stats_wl = calc_stats(observed=subset['elev_m'], modeled=subset['sfincs_m'])
        subset = subset[~subset['height_above_gnd_m'].isna()]
        stats_depth = calc_stats(observed=subset['height_above_gnd_m'], modeled=subset['sfincs_hmax_m'])
        ax = axs[i]
        if i in first_in_row:
            for ii in range(len(basin)):
                b = basin[ii]
                ax.scatter(hwm[hwm['Name'] == b]['elev_m'], hwm[hwm['Name'] == b]['sfincs_m'],
                           color=colors[ii], s=30, edgecolors='black', alpha=1.0, marker=marker[ii], zorder=2)
                stats = stats_wl
        else:
            for ii in range(len(basin)):
                b = basin[ii]
                ax.scatter(subset[subset['Name'] == b]['height_above_gnd_m'],
                           subset[subset['Name'] == b]['sfincs_hmax_m'],
                           color=colors[ii], s=30, edgecolors='black', alpha=1.0, marker=marker[ii], zorder=2)
                stats = stats_depth

        # row_names = ['Elev.\n0-100m', 'Elev.\n0-20m', 'Elev.\n20-40m']
        # count = 0
        # for kk in first_in_row:
        #     axs[kk].set_title(row_names[count], loc='right')
        #     count += 1
            # axs[first_in_row[kk]].text(-0.5, 0.5, row_names[kk],
            #                            horizontalalignment='left',
            #                            verticalalignment='center',
            #                            rotation='horizontal',
            #                            transform=axs[first_in_row[kk]].transAxes)
            # pos0 = axs[last_in_row[kk]].get_position()  # get the original position

        if i == 0:
            ax.set_title('Peak Water Level (m)')
        if i == 1:
            ax.set_title('Peak Water Depth (m)')

        if i in first_in_row:
            ax.set_ylabel('Modeled')
        if i in last_row:
            ax.set_xlabel('Observed')

        # Fix the axis tick marks
        if i == first_in_row[0]:
            vv = axlimits[0]
        elif i == first_in_row[1]:
            vv = axlimits[1]
        elif i == first_in_row[2]:
            vv = axlimits[2]
        else:
            vv = [0, 6]
        ax.set_xlim(vv)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end + 1, stp[i]))
        ax.set_ylim(vv)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end + 1, stp[i]))

        ax.grid(axis='both', alpha=0.7, zorder=-1)

        ss1 = 'Bias: ' + str(stats[2])
        ss2 = 'RMSE: ' + str(stats[1])
        locater = [8, 0.6, 1.6, 0.6, 1.6, 0.6]
        if i in first_in_row:
            ax.text(x=end - 0.15, y=start + locater[i], s=ss1, ha='right', va='bottom')
            ax.text(x=end - 0.15, y=start + 0.01, s=ss2, ha='right', va='bottom')
        else:
            ax.text(x=end - 0.1, y=start + locater[i], s=ss1, ha='right', va='bottom')
            ax.text(x=end - 0.1, y=start + 0.05, s=ss2, ha='right', va='bottom')

        for ax in axs:
            line = mlines.Line2D([0, 1], [0, 1], color='black', alpha=0.8, linestyle='--', zorder=3)
            transform = ax.transAxes
            line.set_transform(transform)
            ax.add_line(line)
            ax.margins(x=0, y=0)

    axs[0].legend(legend_nick, loc='upper left', fontsize=9)
    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.savefig(os.path.join(out_dir, 'hwm_qq_plot.png'), bbox_inches='tight', dpi=225, tight_layout=True)
    plt.close()

# plot_qq_dtm = True
# if plot_qq_dtm is True:
#     # Setup labeling and plot markers/colors
#     colors = ["lightgray", "gray", "darkgray", "gray", 'black']
#     basin = ['Lower Pee Dee', 'Neuse', 'Cape Fear', 'Pamlico', 'Onslow Bay']
#     legend_nick = ['LPD', 'N', 'CF', 'P', 'OB']
#     marker = ["o", "^", "s", "d", "x"]
#
#     # Axis limits for the subplots
#     axlimits = [[0, 100], [0, 20], [20, 40]]
#
#     fig, axs = plt.subplots(nrows=1, ncols=len(axlimits), tight_layout=True, figsize=(6, 3))
#     for i in range(len(axlimits)):
#         axislim = axlimits[i]
#         subset = hwm[(hwm['elev_m'] > axlimits[i][0]) & (hwm['elev_m'] <= axlimits[i][1])]
#         subset = subset[~subset['height_above_gnd_m'].isna()]
#         stats = calc_stats(observed=subset['height_above_gnd_m'], modeled=subset['sfincs_depth_m'])
#         print(stats)
#         for ii in range(len(basin)):
#             b = basin[ii]
#             axs[i].scatter(subset[subset['Name'] == b]['height_above_gnd_m'],
#                            subset[subset['Name'] == b]['sfincs_depth_m'],
#                            color=colors[ii], s=30, edgecolors='black', alpha=1.0, marker=marker[ii], zorder=2)
#
#         line = mlines.Line2D([0, 1], [0, 1], color='black', alpha=0.8, linestyle='--', zorder=3)
#         transform = axs[i].transAxes
#         line.set_transform(transform)
#         axs[i].add_line(line)
#         if i == 0:
#             stp = 2
#             axs[i].set_ylabel('Modeled Depth (m)')
#             axs[i].set_xlabel('Observed Depth (m)')
#             axs[i].set_title('Elevations 0-100m', loc='center')
#         elif i == 1:
#             stp = 2
#             axs[i].set_xlabel('Observed Depth (m)')
#             axs[i].set_ylabel('')
#             axs[i].set_title('Elevations 0-20m', loc='center')
#         else:
#             stp = 2
#             axs[i].set_xlabel('Observed Depth (m)')
#             axs[i].set_ylabel('')
#             axs[i].set_title('Elevations 20-40m', loc='center')
#
#         axs[i].set_xlim([0, 6])
#         start, end = axs[i].get_xlim()
#         axs[i].xaxis.set_ticks(np.arange(start, end + 1, stp))
#         axs[i].set_ylim([0, 6])
#         start, end = axs[i].get_ylim()
#         axs[i].yaxis.set_ticks(np.arange(start, end + 1, stp))
#
#         axs[i].grid(axis='both', alpha=0.7, zorder=-1)
#
#         ss1 = 'Bias: ' + str(stats[2])
#         ss2 = 'RMSE: ' + str(stats[1])
#
#         locater = [0.05, 0.05, 0.05]
#
#         axs[i].text(x=6 - 0.1, y=0.4, s=ss1, ha='right', va='bottom')
#         axs[i].text(x=6 - 0.1, y=locater[i], s=ss2, ha='right', va='bottom')
#
#         # if i == 0:
#         #     axs[i].legend(legend_nick, loc='best')
#
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.tight_layout()
#     plt.margins(x=0, y=0)
#     plt.savefig('hwm_qq_plot_depth.png', bbox_inches='tight', dpi=225)
#     plt.close()

# load_lyrs = True
# if load_lyrs is True:
#     # l_gdf = cat.get_geodataframe('enc_domain_HUC6_clipped')
#     # l_gdf.to_crs(epsg=32617, inplace=True)
#     domain = mod.region
# 
#     l_gdf = cat.get_geodataframe('carolinas_coastal_wb')
#     l_gdf.to_crs(epsg=32617, inplace=True)
#     coastal_wb = l_gdf.clip(domain)
# 
#     l_gdf = cat.get_geodataframe('carolinas_major_rivers')
#     l_gdf.to_crs(epsg=32617, inplace=True)
#     major_rivers = l_gdf.clip(domain)
# 
#     l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\fris\bathy_v5\NHDArea_LowerPeeDee.shp')
#     l_gdf.to_crs(epsg=32617, inplace=True)
#     lpd_riv = l_gdf.clip(domain)
# 
#     l_gdf = cat.get_geodataframe(
#         r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
#     l_gdf = l_gdf[l_gdf['Name'].isin(['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee'])]
#     l_gdf.to_crs(epsg=32617, inplace=True)
#     basins = l_gdf
# 
#     # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\carolinas_15m_contour_poly.shp')
#     # l_gdf.to_crs(epsg=32617, inplace=True)
#     # contour_15m = l_gdf.clip(basins)
# 
#     # l_gdf = cat.get_geodataframe(
#     #     r'Z:\users\lelise\geospatial\infrastructure\tl_2019_us_primaryroads\tl_2019_us_primaryroads.shp')
#     # l_gdf = l_gdf[l_gdf['FULLNAME'].isin(['I- 95'])]
#     # l_gdf.to_crs(epsg=32617, inplace=True)
#     # roads = l_gdf.clip(basins.total_bounds)
#     #
#     # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
#     # l_gdf = l_gdf[l_gdf['Name'].isin(['Myrtle Beach', 'Wilmington', 'New Bern', 'Ocracoke', 'Raleigh'])]
#     # l_gdf.set_index('Name', inplace=True)
#     # l_gdf.to_crs(epsg=32617, inplace=True)
#     # cities = l_gdf.clip(basins)
#     #
#     # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_reservoirs.shp')
#     # l_gdf.set_index('Name', inplace=True)
#     # l_gdf.to_crs(epsg=32617, inplace=True)
#     # reservoirs = l_gdf.clip(basins)
# 
# plt_hwm_map = True
# if plt_hwm_map is True:
#     wkt = mod.grid['dep'].raster.crs.to_wkt()
#     utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
#     utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
#     extent = np.array(basins.buffer(10000).total_bounds)[[0, 2, 1, 3]]
# 
#     fig, ax = plt.subplots(
#         nrows=1, ncols=1,
#         figsize=(6.5, 4.5),
#         subplot_kw={'projection': utm},
#         tight_layout=True)
# 
#     cmap = mpl.cm.binary
#     bounds = [0, 15, 30, 45, 60, 120, 200]
#     norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
#     mod.grid['dep'].plot(ax=ax,
#                          cmap=cmap,
#                          norm=norm,
#                          add_colorbar=False,
#                          zorder=2,
#                          alpha=0.75
#                          )
#     ax.set_title('')
#     # Plot background/geography layers
#     # major_rivers.plot(ax=ax, color='steelblue', edgecolor='none', linewidth=0.5, linestyle='-', zorder=3, alpha=1)
#     # lpd_riv.plot(ax=ax, color='steelblue', edgecolor='steelblue', linewidth=0.5, linestyle='-', zorder=3, alpha=1)
#     basins.plot(ax=ax, color='none', edgecolor='black', linewidth=1, linestyle='-', zorder=3, alpha=1)
# 
#     n_bins_ranges = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
#     vmax = round(max(n_bins_ranges), 0)
#     vmin = round(min(n_bins_ranges), 0)
#     hwm.plot(column='error',
#              cmap='seismic',
#              legend=False,
#              vmin=vmin, vmax=vmax,
#              ax=ax,
#              markersize=20,
#              alpha=0.85,
#              edgecolor='black',
#              linewidth=0.5,
#              zorder=3,
#              marker='v'
#              )
# 
#     sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=vmin, vmax=vmax))
#     fig.colorbar(sm,
#                  ax=ax,
#                  shrink=0.7,
#                  extend='both',
#                  spacing='uniform',
#                  label='Water Level Diff (m)\n Modeled - Observed')
# 
#     # Add title and save figure
# 
#     ax.set_extent(extent, crs=utm)
#     # ax.set_title('HWM Modeled minus Observed Water Level', loc='left')
#     ax.set_ylabel(f"y coord UTM zone {utm_zone} [m]")
#     ax.yaxis.set_visible(True)
#     ax.set_xlabel(f"x coord UTM zone {utm_zone} [m]")
#     ax.xaxis.set_visible(True)
#     ax.ticklabel_format(style='sci', useOffset=False)
#     ax.set_aspect('equal')
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.margins(x=0, y=0)
#     plt.savefig('hwm_error_map.png', bbox_inches='tight', dpi=255)  # , pil_kwargs={'quality': 95})
#     plt.close()
