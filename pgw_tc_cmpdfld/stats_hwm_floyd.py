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


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    out = LinearSegmentedColormap.from_list(cmap_name, color_list, N)
    return out


def hwm_to_gdf(csv_file_path, agency, quality=None, dst_crs=None):
    df = pd.read_csv(csv_file_path)

    # If the HWM is downloaded from the USGS
    if agency == 'usgs':
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['longitude'], y=df['latitude'], crs=4326))
        gdf['elev_m'] = gdf['elev_ft'] * 0.3048
        if quality:
            gdf = gdf[gdf['hwm_quality_id'] <= quality]
        if dst_crs:
            gdf.to_crs(dst_crs, inplace=True)
        gdf = gdf[gdf['elev_m'].notna()]

    # If the HWM data is from the NCEM
    elif agency == 'ncem':
        df = pd.read_csv(csv_file_path)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['lon_dd'], y=df['lat_dd'], crs=4326))
        gdf['elev_m'] = gdf['elev_ft'] * 0.3048
        if dst_crs:
            gdf.to_crs(dst_crs, inplace=True)
        gdf = gdf[gdf['elev_m'].notna()]
    return gdf


def extract_water_level(gdf, raster):
    xcoords = gdf.geometry.x.to_xarray()
    ycoods = gdf.geometry.y.to_xarray()
    gdf['sfincs_m'] = raster.sel(x=xcoords, y=ycoods, method='nearest').values.transpose()
    gdf = gdf[gdf['sfincs_m'].notna()]
    gdf['error'] = gdf['sfincs_m'] - gdf['elev_m']
    return gdf


def calc_stats(observed, modeled):
    mae = abs(observed - modeled).values.mean()
    rmse = ((observed - modeled) ** 2).mean() ** 0.5
    bias = (modeled - observed).values.mean()
    return [round(mae, 2), round(rmse, 2), round(bias, 2)]


def plot_obs_vs_mod(gdf, fileout, figsize=(2, 2), fontsize=9, axislim=None, gdf2=None, legend=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=figsize)

    ax.scatter(gdf['elev_m'], gdf['sfincs_m'],
               color='lightgreen',
               s=50,
               edgecolors='black',
               alpha=0.7,
               marker="o")
    if gdf2 is not None:
        ax.scatter(gdf2['elev_m'], gdf2['sfincs_m'],
                   color='darkgrey',
                   s=70,
                   edgecolors='black',
                   alpha=0.7,
                   marker="^")

    line = mlines.Line2D([0, 1], [0, 1], color='black', alpha=0.8, linestyle='--')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    if axislim is not None:
        ax.set_xlim(axislim)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))

        ax.set_ylim(axislim)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 5))

    ax.set_ylabel('Modeled WL (m +NAVD88)', fontsize=fontsize)
    ax.set_xlabel('Observed WL (m +NAVD88)', fontsize=fontsize)
    if legend is not None:
        plt.legend(legend, loc='best')
    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.savefig(fileout, bbox_inches='tight')
    plt.close()
    return None


def plot_obs_vs_mod_huc6(hwm, fileout, figsize=(3, 3), axislim=None, plot_legend=None, stats=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=figsize)

    colors = ["lightgray", "gray", "darkgray", "gray", 'black']
    legend = ['Lower Pee Dee', 'Neuse', 'Cape Fear', 'Pamlico', 'Onslow Bay']
    ax.scatter(hwm[hwm['Name'] == legend[0]]['elev_m'], hwm[hwm['Name'] == legend[0]]['sfincs_m'],
               color=colors[0], s=60, edgecolors='black', alpha=1.0, marker="o")
    ax.scatter(hwm[hwm['Name'] == legend[1]]['elev_m'], hwm[hwm['Name'] == legend[1]]['sfincs_m'],
               color=colors[1], s=60, edgecolors='black', alpha=0.9, marker="^")
    ax.scatter(hwm[hwm['Name'] == legend[2]]['elev_m'], hwm[hwm['Name'] == legend[2]]['sfincs_m'],
               color=colors[2], s=60, edgecolors='black', alpha=0.8, marker="s")
    ax.scatter(hwm[hwm['Name'] == legend[3]]['elev_m'], hwm[hwm['Name'] == legend[3]]['sfincs_m'],
               color=colors[3], s=60, edgecolors='black', alpha=0.9, marker="d")
    ax.scatter(hwm[hwm['Name'] == legend[4]]['elev_m'], hwm[hwm['Name'] == legend[4]]['sfincs_m'],
               color=colors[4], s=70, edgecolors='black', alpha=0.7, marker="x")

    line = mlines.Line2D([0, 1], [0, 1], color='black', alpha=0.8, linestyle='--')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    if axislim is not None:
        ax.set_xlim(axislim)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 5))

        ax.set_ylim(axislim)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 5))

    ax.set_ylabel('Modeled Water Level\n(m +NAVD88)')
    ax.set_xlabel('Observed Water Level\n(m +NAVD88)')
    if plot_legend is not None:
        plt.legend(legend, loc='best')
    plt.tight_layout()
    ss = 'Bias: ' + str(stats[2]) + '; RMSE: ' + str(stats[1])
    plt.text(x=axislim[1] - 0.1, y=axislim[0] + 0.1, s=ss, ha='right', va='bottom')
    plt.margins(x=0, y=0)
    plt.savefig(fileout, bbox_inches='tight', dpi=225)
    plt.close()
    return None


yml = r'Z:\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
yml_base = r'Z:\users\lelise\data\data_catalog_BASE_Carolinas.yml'
os.chdir(
    r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\arch_hindcast')
model_root = 'floy_pres_compound'
storm = 'floyd'
mod = SfincsModel(root=model_root, mode='r', data_libs=[yml, yml_base])
cat = mod.data_catalog
mod.read_results()

out_dir = os.path.join(os.getcwd(), model_root, 'validation', 'hwm')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
os.chdir(out_dir)

# Read USGS HWM file, extract modeled water levels, calculate error
# hwm_usgs = hwm_to_gdf(csv_file_path=(r'Z:\users\lelise\geospatial\observations\usgs_' + storm + r'_FilteredHWMs.csv'),
#                       agency='usgs',
#                       quality=3,
#                       dst_crs=mod.crs.to_epsg())
# hwm_usgs = extract_water_level(gdf=hwm_usgs, raster=mod.results['zsmax'].max(dim='timemax'))

# Read NCEM HWM file and write to CSV
# hwm_ncem = gpd.read_file(r'Z:\users\lelise\projects\ENC_CompFld\HWM_master_share_Sept2023\HWM_master_share.gdb')
# hwm_ncem_df = pd.DataFrame(hwm_ncem)
# hwm_ncem_df.drop('geometry', inplace=True, axis=1)
# hwm_ncem_df[hwm_ncem_df.isnull()] = np.nan
# hwm_ncem_df[hwm_ncem_df.isna()] = np.nan
# hwm_ncem_df.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\HWM_master_share_Sept2023\NCEM_hwm_database_Sep2023.csv',
#                    index=False)
hwm_ncem = hwm_to_gdf(csv_file_path=r'Z:\users\lelise\data\NC_State_Agencies\NCEM_HWM\NCEM_hwm_database_Sep2023.csv',
                      agency='ncem',
                      quality=None,
                      dst_crs=mod.crs.to_epsg())
# hwm_ncem_storm = hwm_ncem[hwm_ncem['storm_name'] == 'Hurricane Florence']
#hwm_ncem_storm = hwm_ncem[hwm_ncem['storm_name'] == 'Hurricane Matthew']
hwm_ncem_storm = hwm_ncem[hwm_ncem['storm_name'] == 'Hurricane Floyd']
hwm_ncem_storm = extract_water_level(gdf=hwm_ncem_storm, raster=mod.results['zsmax'].max(dim='timemax'))

# Assign to state
# hwm_nc = hwm[hwm['stateName'] == 'NC']
# hwm_sc = hwm[hwm['stateName'] == 'SC']
# stats_sc = calc_stats(observed=hwm_sc['elev_m'], modeled=hwm_sc['sfincs_m'])
# stats_nc = calc_stats(observed=hwm_nc['elev_m'], modeled=hwm_nc['sfincs_m'])
stats = calc_stats(observed=hwm_ncem_storm['elev_m'], modeled=hwm_ncem_storm['sfincs_m'])

# Assign to HUC6
huc_boundary = gpd.read_file(r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape'
                             r'\WBDHU6.shp')
huc_boundary.to_crs(mod.crs, inplace=True)
huc_boundary = huc_boundary[["HUC6", "Name", "geometry"]]
hwm = gpd.tools.sjoin(left_df=hwm_ncem_storm,
                      right_df=huc_boundary,
                      how='left')
hwm.to_csv('hwm_error_all_points_ncem.csv', index=False)

# Calculate stats by HUC
stats_out = []
for h in hwm['Name'].unique():
    subset = hwm[hwm['Name'] == h]
    stats_subset = calc_stats(observed=subset['elev_m'], modeled=subset['sfincs_m'])
    stats_subset.append(h)
    stats_out.append(stats_subset)

t = pd.DataFrame(stats_out)
t.columns = ['mae', 'rmse', 'bias', 'name']
t.set_index('name', inplace=True, drop=True)
t.to_csv('hwm_stats_huc6.csv', index=True)

# Plotting histogram
hwm['elev_grp'] = 'xx'
hwm['elev_grp'][hwm['elev_m'] <= 15] = 'Coastal (Elevation <= 15m)'
hwm['elev_grp'][(hwm['elev_m'] > 15)] = 'Inland (Elevation > 15m)'

font = {'family': 'Arial',
        'size': 10
        }
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})

attr = 'elev_grp'
s = len(hwm[attr].unique())
fig, ax = plt.subplots(tight_layout=True, figsize=(3.25, 3.25))
hwm.hist(column='error',
         ax=ax,
         by=attr,
         sharex=True,
         sharey=True,
         bins=20,
         grid=True,
         layout=(s, 1),
         color='darkgrey',
         rwidth=0.9,
         )
plt.xlabel("HWM Modeled minus Observed Water Level (m)")
plt.ylabel("Frequency")
plt.margins(x=0, y=0)
plt.savefig(attr, bbox_inches='tight', dpi=225)
plt.close()

''' Q-Q PLOTS '''
colors = ["lightgray", "gray", "darkgray", "gray", 'black']
basin = ['Lower Pee Dee', 'Neuse', 'Cape Fear', 'Pamlico', 'Onslow Bay']
legend_nick = ['LPD', 'N', 'CF', 'P', 'OB']
marker = ["o", "^", "s", "d", "x"]
axlimits = [[0, 100], [0, 15], [15, 45]]

fig, axs = plt.subplots(nrows=1, ncols=len(axlimits),
                        tight_layout=True,
                        figsize=(6.5, 4))
for i in range(len(axlimits)):
    axislim = axlimits[i]
    subset = hwm[(hwm['elev_m'] > axlimits[i][0]) & (hwm['elev_m'] <= axlimits[i][1])]
    stats = calc_stats(observed=subset['elev_m'], modeled=subset['sfincs_m'])
    for ii in range(len(basin)):
        b = basin[ii]
        axs[i].scatter(hwm[hwm['Name'] == b]['elev_m'], hwm[hwm['Name'] == b]['sfincs_m'],
                       color=colors[ii], s=60, edgecolors='black', alpha=1.0, marker=marker[ii], zorder=2)

    line = mlines.Line2D([0, 1], [0, 1], color='black', alpha=0.8, linestyle='--', zorder=3)
    transform = axs[i].transAxes
    line.set_transform(transform)
    axs[i].add_line(line)
    if i == 0:
        stp = 20
        axs[i].set_ylabel('Modeled WL\n(m +NAVD88)')
        axs[i].set_xlabel('Observed WL\n(m +NAVD88)')
        axs[i].set_title('Entire Domain', loc='left')
    elif i == 1:
        stp = 5
        axs[i].set_xlabel('Observed WL\n(m +NAVD88)')
        axs[i].set_ylabel('')
        axs[i].set_title('Coastal', loc='left')
    else:
        stp = 10
        axs[i].set_xlabel('Observed WL\n(m +NAVD88)')
        axs[i].set_ylabel('')
        axs[i].set_title('Inland Subset', loc='left')

    axs[i].set_xlim(axislim)
    start, end = axs[i].get_xlim()
    axs[i].xaxis.set_ticks(np.arange(start, end + 1, stp))
    axs[i].set_ylim(axislim)
    start, end = axs[i].get_ylim()
    axs[i].yaxis.set_ticks(np.arange(start, end + 1, stp))

    axs[i].grid(axis='both', alpha=0.7, zorder=-1)

    ss1 = 'Bias: ' + str(stats[2])
    ss2 = 'RMSE: ' + str(stats[1])

    locater = [4.5, 1, 1.5]

    axs[i].text(x=axislim[1] - 0.15, y=axislim[0] + locater[i], s=ss1, ha='right', va='bottom')
    axs[i].text(x=axislim[1] - 0.15, y=axislim[0] + 0.1, s=ss2, ha='right', va='bottom')

    if i == 0:
        axs[i].legend(legend_nick, loc='best')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.margins(x=0, y=0)
plt.savefig('qq_plots_3panel2.png', bbox_inches='tight', dpi=225)
plt.close()

# ''' PLOT MAP '''
# font = {'family': 'Arial',
#         'size': 10}
# mpl.rc('font', **font)
# # Plotting the data on a map with contextual layers
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
#              alpha=0.9,
#              edgecolor='black',
#              linewidth=0.5,
#              zorder=3)
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
#     ax.set_ylabel(f"y coordinate UTM zone {utm_zone} [m]")
#     ax.yaxis.set_visible(True)
#     ax.set_xlabel(f"x coordinate UTM zone {utm_zone} [m]")
#     ax.xaxis.set_visible(True)
#     ax.ticklabel_format(style='plain', useOffset=False)
#     ax.set_aspect('equal')
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.margins(x=0, y=0)
#     plt.savefig('hwm_error_map4.png', bbox_inches='tight', dpi=255)  # , pil_kwargs={'quality': 95})
#     plt.close()
