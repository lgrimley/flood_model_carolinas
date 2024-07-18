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


# Load in model and read results
cat_dir = r'Z:\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')

os.chdir(r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\arch_hindcast')
model_root = 'flor_pres_compound'
mod = SfincsModel(root=model_root, mode='r', data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
cat = mod.data_catalog
mod.read_results()

# Create a directory to save data and figures to
out_dir = os.path.join(os.getcwd(), model_root, 'validation', 'hwm')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
os.chdir(out_dir)

'''' Read in HWM data '''
# Read USGS HWM data
storm = 'florence'
hwm_usgs = hwm_to_gdf(csv_file_path=rf'Z:\users\lelise\geospatial\observations\usgs_{storm}_FilteredHWMs.csv',
                      agency='usgs',
                      quality=3,
                      dst_crs=mod.crs.to_epsg())
hwm_usgs['data_source'] = 'USGS'
hwm_usgs = hwm_usgs[hwm_usgs['stateName'].isin(['NC', 'SC'])]

# Read NCEM HWM file and write to CSV
if os.path.exists(r'Z:\users\lelise\data\NC_State_Agencies\NCEM_HWM\NCEM_hwm_database_Sep2023.csv') is False:
    hwm_ncem = gpd.read_file(r'Z:\users\lelise\projects\ENC_CompFld\HWM_master_share_Sept2023\HWM_master_share.gdb')
    hwm_ncem_df = pd.DataFrame(hwm_ncem)
    hwm_ncem_df.drop('geometry', inplace=True, axis=1)
    hwm_ncem_df[hwm_ncem_df.isnull()] = np.nan
    hwm_ncem_df[hwm_ncem_df.isna()] = np.nan
    hwm_ncem_df.to_csv(r'Z:\users\lelise\projects\ENC_CompFld\HWM_master_share_Sept2023\NCEM_hwm_database_Sep2023.csv',
                       index=False)

# Read in NCEM HWM data
hwm_ncem = hwm_to_gdf(csv_file_path=r'Z:\users\lelise\data\NC_State_Agencies\NCEM_HWM'
                                    r'\NCEM_hwm_database_Sep2023.csv', agency='ncem',
                      quality=None, dst_crs=mod.crs.to_epsg())
print(hwm_ncem['storm_name'].unique())
print(hwm_ncem['data_source'].unique())

# Subset by storm of interest
hwm_ncem_storm = hwm_ncem[hwm_ncem['storm_name'] == 'Hurricane Florence']
print(hwm_ncem_storm['data_source'].unique())
print(hwm_ncem_storm['confidence'].unique())

# Remove data with Poor or lower quality
quality_category = ['Unknown/Historical', 'VP: > 0.40 ft', 'Poor: +/- 0.40 ft']
hwm_ncem_storm = hwm_ncem_storm.loc[~hwm_ncem_storm['confidence'].isin(quality_category)]
hwm_ncem_storm = hwm_ncem_storm.loc[hwm_ncem_storm['data_source'] == 'NCGS']
hwm_ncem_storm.columns = ['latitude_dd', 'longitude_dd', 'elev_ft', 'eventName',
                          'data_source', 'hwmQualityName', 'geometry', 'elev_m']

# Combine datasets
hwm = pd.concat([hwm_usgs, hwm_ncem_storm], axis=0, ignore_index=True)
hwm = hwm.drop_duplicates(subset='geometry', keep='first')

''' Extract modeled water levels at HWMs and Calc Stats '''
# Extract peak modeled water levels at the HWM points
hwm = extract_water_level(gdf=hwm, raster=mod.results['zsmax'].max(dim='timemax'))
_, rmse, bias = calc_stats(observed=hwm['elev_m'], modeled=hwm['sfincs_m'])
print(rmse, bias)

''' Calculate HWM stats by HUC6 Watershed '''
# Assign to HUC6
huc_boundary = gpd.read_file(r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape'
                             r'\WBDHU6.shp')
huc_boundary.to_crs(mod.crs, inplace=True)
huc_boundary = huc_boundary[["HUC6", "Name", "geometry"]]
hwm = gpd.tools.sjoin(left_df=hwm, right_df=huc_boundary, how='left')

# Extract Elevation at HWM locations from subgrid file
dep_file = r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_obs\arch_hindcast\sfincs_floodmaps_072024' \
           r'\downscale_5m\test.tif'
#r'Z:\users\lelise\projects\ENC_CompFld\Chapter2\sfincs_models_wrf\subgrid\dep.tif'
dep = mod.data_catalog.get_rasterdataset(dep_file)
xcoords = hwm.geometry.x.to_xarray()
ycoods = hwm.geometry.y.to_xarray()
hwm['sfincs_depth_m'] = dep.sel(x=xcoords, y=ycoods, method='nearest').values.transpose()
hwm['sfincs_depth_m'].fillna(0, inplace=True)
# hwm['sfincs_minus_dtm'] = hwm['sfincs_m'] - hwm['gnd_elev_m']
# hwm['hwm_minus_dtm'] = hwm['elev_m'] - hwm['gnd_elev_m']
hwm['xcoords'] = hwm.geometry.x.to_xarray()
hwm['ycoords'] = hwm.geometry.y.to_xarray()
hwm['height_above_gnd_m'] = hwm['height_above_gnd']*0.3048
hwm['depth_error'] = hwm['sfincs_depth_m'] - hwm['height_above_gnd_m']

# Write out all the HWM data
hwm.to_csv('hwm_error_all.csv', index=False)

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

# Calculate depth stats
print(len(hwm[~hwm['height_above_gnd_m'].isna()]))
stats_out = []
for h in hwm_dep['Name'].unique():
    subset = hwm[hwm['Name'] == h]
    subset = subset[~subset['height_above_gnd_m'].isna()]
    stats_subset = calc_stats(observed=subset['height_above_gnd_m'], modeled=subset['sfincs_depth_m'])
    stats_subset.append(h)
    stats_out.append(stats_subset)
t = pd.DataFrame(stats_out)
t.columns = ['mae', 'rmse', 'bias', 'name']
t.set_index('name', inplace=True, drop=True)
t.to_csv('hwm_stats_huc6_depth.csv', index=True)

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

    # Elevation errors
    fig, ax = plt.subplots(tight_layout=True, figsize=(3, 3))
    hwm.hist(column='error',
             ax=ax, by=attr,
             sharex=True, sharey=True,
             bins=40, grid=True,
             layout=(s, 1),
             color='darkgrey', rwidth=0.9,
             )
    plt.xlabel("HWM Modeled minus Observed\nMax Water Level (m)")
    plt.ylabel("Frequency")
    plt.margins(x=0, y=0)
    plt.savefig('hwm_errors_water_level.png', bbox_inches='tight', dpi=225)
    plt.close()

    # Depth errors
    fig, ax = plt.subplots(tight_layout=True, figsize=(3, 3))
    hwm.hist(column='depth_error',
             ax=ax, by=attr,
             sharex=True, sharey=True,
             bins=40, grid=True,
             layout=(s, 1),
             color='darkgrey', rwidth=0.9,
             )
    plt.xlabel("HWM Modeled minus Observed\nMax Depth (m)")
    plt.ylabel("Frequency")
    plt.margins(x=0, y=0)
    plt.savefig('hwm_errors_depth.png', bbox_inches='tight', dpi=225)
    plt.close()

# Plot a QQ of the HWM errors (Figure 4)
plot_qq = True
if plot_qq is True:
    # Setup labeling and plot markers/colors
    colors = ["lightgray", "gray", "darkgray", "gray", 'black']
    basin = ['Lower Pee Dee', 'Neuse', 'Cape Fear', 'Pamlico', 'Onslow Bay']
    legend_nick = ['LPD', 'N', 'CF', 'P', 'OB']
    marker = ["o", "^", "s", "d", "x"]

    # Axis limits for the subplots
    axlimits = [[0, 100], [0, 20], [20, 40]]

    fig, axs = plt.subplots(nrows=1, ncols=len(axlimits), tight_layout=True, figsize=(6, 3))
    for i in range(len(axlimits)):
        axislim = axlimits[i]
        subset = hwm[(hwm['elev_m'] > axlimits[i][0]) & (hwm['elev_m'] <= axlimits[i][1])]
        stats = calc_stats(observed=subset['elev_m'], modeled=subset['sfincs_m'])
        for ii in range(len(basin)):
            b = basin[ii]
            axs[i].scatter(hwm[hwm['Name'] == b]['elev_m'], hwm[hwm['Name'] == b]['sfincs_m'],
                           color=colors[ii], s=30, edgecolors='black', alpha=1.0, marker=marker[ii], zorder=2)

        line = mlines.Line2D([0, 1], [0, 1], color='black', alpha=0.8, linestyle='--', zorder=3)
        transform = axs[i].transAxes
        line.set_transform(transform)
        axs[i].add_line(line)
        if i == 0:
            stp = 20
            axs[i].set_ylabel('Modeled WL\n(m +NAVD88)')
            axs[i].set_xlabel('Observed WL\n(m +NAVD88)')
            axs[i].set_title('Elevations 0-100m', loc='left')
        elif i == 1:
            stp = 5
            axs[i].set_xlabel('Observed WL\n(m +NAVD88)')
            axs[i].set_ylabel('')
            axs[i].set_title('Elevations 0-20m', loc='left')
        else:
            stp = 5
            axs[i].set_xlabel('Observed WL\n(m +NAVD88)')
            axs[i].set_ylabel('')
            axs[i].set_title('Elevations 20-40m', loc='left')

        axs[i].set_xlim(axislim)
        start, end = axs[i].get_xlim()
        axs[i].xaxis.set_ticks(np.arange(start, end + 1, stp))
        axs[i].set_ylim(axislim)
        start, end = axs[i].get_ylim()
        axs[i].yaxis.set_ticks(np.arange(start, end + 1, stp))

        axs[i].grid(axis='both', alpha=0.7, zorder=-1)

        ss1 = 'Bias: ' + str(stats[2])
        ss2 = 'RMSE: ' + str(stats[1])

        locater = [8, 1.5, 1.5]

        axs[i].text(x=axislim[1] - 0.15, y=axislim[0] + locater[i], s=ss1, ha='right', va='bottom')
        axs[i].text(x=axislim[1] - 0.15, y=axislim[0] +0.01, s=ss2, ha='right', va='bottom')

        if i == 0:
            axs[i].legend(legend_nick, loc='best', fontsize=8)

    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.savefig('hwm_qq_plot.png', bbox_inches='tight', dpi=225, tight_layout=True)
    plt.close()

plot_qq_dtm = True
if plot_qq_dtm is True:
    # Setup labeling and plot markers/colors
    colors = ["lightgray", "gray", "darkgray", "gray", 'black']
    basin = ['Lower Pee Dee', 'Neuse', 'Cape Fear', 'Pamlico', 'Onslow Bay']
    legend_nick = ['LPD', 'N', 'CF', 'P', 'OB']
    marker = ["o", "^", "s", "d", "x"]

    # Axis limits for the subplots
    axlimits = [[0, 100], [0, 20], [20, 40]]

    fig, axs = plt.subplots(nrows=1, ncols=len(axlimits), tight_layout=True, figsize=(6, 3))
    for i in range(len(axlimits)):
        axislim = axlimits[i]
        subset = hwm[(hwm['elev_m'] > axlimits[i][0]) & (hwm['elev_m'] <= axlimits[i][1])]
        subset = subset[~subset['height_above_gnd_m'].isna()]
        stats = calc_stats(observed=subset['height_above_gnd_m'], modeled=subset['sfincs_depth_m'])
        print(stats)
        for ii in range(len(basin)):
            b = basin[ii]
            axs[i].scatter(subset[subset['Name'] == b]['height_above_gnd_m'],
                           subset[subset['Name'] == b]['sfincs_depth_m'],
                           color=colors[ii], s=30, edgecolors='black', alpha=1.0, marker=marker[ii], zorder=2)

        line = mlines.Line2D([0, 1], [0, 1], color='black', alpha=0.8, linestyle='--', zorder=3)
        transform = axs[i].transAxes
        line.set_transform(transform)
        axs[i].add_line(line)
        if i == 0:
            stp = 2
            axs[i].set_ylabel('Modeled Depth (m)')
            axs[i].set_xlabel('Observed Depth (m)')
            axs[i].set_title('Elevations 0-100m', loc='left')
        elif i == 1:
            stp = 2
            axs[i].set_xlabel('Observed Depth (m)')
            axs[i].set_ylabel('')
            axs[i].set_title('Elevations 0-20m', loc='left')
        else:
            stp = 2
            axs[i].set_xlabel('Observed Depth (m)')
            axs[i].set_ylabel('')
            axs[i].set_title('Elevations 20-40m', loc='left')

        axs[i].set_xlim([0, 6])
        start, end = axs[i].get_xlim()
        axs[i].xaxis.set_ticks(np.arange(start, end + 1, stp))
        axs[i].set_ylim([0, 6])
        start, end = axs[i].get_ylim()
        axs[i].yaxis.set_ticks(np.arange(start, end + 1, stp))

        axs[i].grid(axis='both', alpha=0.7, zorder=-1)

        ss1 = 'Bias: ' + str(stats[2])
        ss2 = 'RMSE: ' + str(stats[1])

        locater = [0.05, 0.05, 0.05]

        axs[i].text(x=6-0.1, y=0.4, s=ss1, ha='right', va='bottom')
        axs[i].text(x=6-0.1, y=locater[i], s=ss2, ha='right', va='bottom')

        # if i == 0:
        #     axs[i].legend(legend_nick, loc='best')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.savefig('hwm_qq_plot_depth.png', bbox_inches='tight', dpi=225)
    plt.close()

load_lyrs = True
if load_lyrs is True:
    # l_gdf = cat.get_geodataframe('enc_domain_HUC6_clipped')
    # l_gdf.to_crs(epsg=32617, inplace=True)
    domain = mod.region

    l_gdf = cat.get_geodataframe('carolinas_coastal_wb')
    l_gdf.to_crs(epsg=32617, inplace=True)
    coastal_wb = l_gdf.clip(domain)

    l_gdf = cat.get_geodataframe('carolinas_major_rivers')
    l_gdf.to_crs(epsg=32617, inplace=True)
    major_rivers = l_gdf.clip(domain)

    l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\fris\bathy_v5\NHDArea_LowerPeeDee.shp')
    l_gdf.to_crs(epsg=32617, inplace=True)
    lpd_riv = l_gdf.clip(domain)

    l_gdf = cat.get_geodataframe(
        r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
    l_gdf = l_gdf[l_gdf['Name'].isin(['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee'])]
    l_gdf.to_crs(epsg=32617, inplace=True)
    basins = l_gdf

    # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\carolinas_15m_contour_poly.shp')
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # contour_15m = l_gdf.clip(basins)

    # l_gdf = cat.get_geodataframe(
    #     r'Z:\users\lelise\geospatial\infrastructure\tl_2019_us_primaryroads\tl_2019_us_primaryroads.shp')
    # l_gdf = l_gdf[l_gdf['FULLNAME'].isin(['I- 95'])]
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # roads = l_gdf.clip(basins.total_bounds)
    #
    # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_cities.shp')
    # l_gdf = l_gdf[l_gdf['Name'].isin(['Myrtle Beach', 'Wilmington', 'New Bern', 'Ocracoke', 'Raleigh'])]
    # l_gdf.set_index('Name', inplace=True)
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # cities = l_gdf.clip(basins)
    #
    # l_gdf = cat.get_geodataframe(r'Z:\users\lelise\geospatial\infrastructure\enc_major_reservoirs.shp')
    # l_gdf.set_index('Name', inplace=True)
    # l_gdf.to_crs(epsg=32617, inplace=True)
    # reservoirs = l_gdf.clip(basins)

plt_hwm_map = True
if plt_hwm_map is True:
    wkt = mod.grid['dep'].raster.crs.to_wkt()
    utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
    utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
    extent = np.array(basins.buffer(10000).total_bounds)[[0, 2, 1, 3]]

    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(6.5, 4.5),
        subplot_kw={'projection': utm},
        tight_layout=True)

    cmap = mpl.cm.binary
    bounds = [0, 15, 30, 45, 60, 120, 200]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    mod.grid['dep'].plot(ax=ax,
                         cmap=cmap,
                         norm=norm,
                         add_colorbar=False,
                         zorder=2,
                         alpha=0.75
                         )
    ax.set_title('')
    # Plot background/geography layers
    # major_rivers.plot(ax=ax, color='steelblue', edgecolor='none', linewidth=0.5, linestyle='-', zorder=3, alpha=1)
    # lpd_riv.plot(ax=ax, color='steelblue', edgecolor='steelblue', linewidth=0.5, linestyle='-', zorder=3, alpha=1)
    basins.plot(ax=ax, color='none', edgecolor='black', linewidth=1, linestyle='-', zorder=3, alpha=1)

    n_bins_ranges = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    vmax = round(max(n_bins_ranges), 0)
    vmin = round(min(n_bins_ranges), 0)
    hwm.plot(column='error',
             cmap='seismic',
             legend=False,
             vmin=vmin, vmax=vmax,
             ax=ax,
             markersize=20,
             alpha=0.85,
             edgecolor='black',
             linewidth=0.5,
             zorder=3,
             marker='v'
             )

    sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm,
                 ax=ax,
                 shrink=0.7,
                 extend='both',
                 spacing='uniform',
                 label='Water Level Diff (m)\n Modeled - Observed')

    # Add title and save figure

    ax.set_extent(extent, crs=utm)
    # ax.set_title('HWM Modeled minus Observed Water Level', loc='left')
    ax.set_ylabel(f"y coord UTM zone {utm_zone} [m]")
    ax.yaxis.set_visible(True)
    ax.set_xlabel(f"x coord UTM zone {utm_zone} [m]")
    ax.xaxis.set_visible(True)
    ax.ticklabel_format(style='sci', useOffset=False)
    ax.set_aspect('equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig('hwm_error_map.png', bbox_inches='tight', dpi=255)  # , pil_kwargs={'quality': 95})
    plt.close()
