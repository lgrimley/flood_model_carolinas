import os
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel, utils
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import shapely


# This script calculates water level stats using SFINCS map (sfincs_map.nc) output extracted at gage locations.
# Use createPlots = True/False to get the script to create figures of the model vs. observed water levels


def calculate_stats(df, tstart=None, tend=None):
    if tstart:
        df = df[df.index > tstart]
    if tend:
        df = df[df.index < tend]

    n = len(df)
    # Mean Absolute Error
    mae = sum(abs(df.Modeled - df.Observed)) / n
    # Mean Error or Bias
    bias = sum(df.Modeled - df.Observed) / n
    # Root Mean Squared Error
    rmse = sum(((df.Observed - df.Modeled) ** 2) / n) ** 0.5
    # Peak Error
    pe = df.Modeled.max() - df.Observed.max()
    # Time to Peak - Error
    tpe = df.Modeled.idxmax() - df.Observed.idxmax()
    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (sum((df.Modeled - df.Observed) ** 2) / sum((df.Observed - df.Observed.mean()) ** 2))
    # Correlation Coefficient (Pearson)
    try:
        r = sum(((df.Modeled - df.Modeled.mean()) * (df.Observed - df.Observed.mean()))) / (
                sum((df.Modeled - df.Modeled.mean()) ** 2) * sum((df.Observed - df.Observed.mean()) ** 2)) ** 0.5
    except:
        print('Correlation Coefficient problem')
        r = 0

    # Coefficient of Determination
    r2 = r ** 2

    # Save stats in a dataframe for output
    stats = pd.DataFrame(data={'mae': round(mae, 2),
                               'rmse': round(rmse, 2),
                               'nse': round(nse, 2),
                               'bias': round(bias, 2),
                               'r': round(r, 2),
                               'r2': round(r2, 2),
                               'pe': round(pe, 2),
                               'tpe': round(tpe.seconds, 1),
                               },
                         index=[0]
                         )
    peak_dt = [df.Observed.idxmax(), df.Modeled.idxmax()]

    return stats, peak_dt


def clean_obs_coords(obs_df, source_crs, target_crs):
    # Clean up the observation data and the coordinates
    if 'geometry' in list(obs_df.coords):
        pts = gpd.GeoDataFrame(obs_df.index.values,
                               geometry=obs_df.geometry.values,
                               crs=source_crs)
        pts.to_crs(target_crs, inplace=True)
        pts.columns = ['site_no', 'geometry']
        obs_df.geometry.values = pts.geometry
    else:
        pts = gpd.GeoDataFrame(obs_df.index,
                               geometry=gpd.points_from_xy(x=obs_df.x.values,
                                                           y=obs_df.y.values,
                                                           crs=source_crs))
        pts.to_crs(target_crs, inplace=True)
        pts.columns = ['site_no', 'geometry']
        obs_df.x.values = pts.geometry.x
        obs_df.y.values = pts.geometry.y

    return pts, obs_df


def plot_hydrograph(df, title, figname):
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4))
    plt.rcParams.update({'font.size': 9})
    ax.plot(df['Observed'], linewidth=0.5, color='black', alpha=0.6)
    ax.plot(df['Modeled'], linewidth=1.5, color='blue', alpha=0.9)
    plt.legend(['Observed', 'Modeled'])
    d1 = df.drop(axis=1, columns=["Modeled"])
    d1['Datetime'] = d1.index
    d1.plot.scatter(x='Datetime',
                    y='Observed',
                    c='black',
                    style='.',
                    s=3,
                    ax=ax)
    ax.set_ylabel('Water Level (m+NAVD88)')
    ax.set_xlabel('')
    fig.autofmt_xdate()
    ax.grid(axis='y', color='gray',
            linewidth=0.5, linestyle='dashed', alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def stats_by_zone(gdf, target_crs, stats_df, output_dir, group_col):
    # project geodataframe
    gdf.to_crs(target_crs, inplace=True)

    # assign points to HUC6
    df_out = gpd.tools.sjoin(left_df=stats_df, right_df=gdf, how='left')
    df_out = df_out.drop(['index_right'], axis=1)
    df_out.reset_index(inplace=True, drop=True)
    df_out.dropna(axis=0, how='any', inplace=True)

    # Output points w/ stats to csv and shapefile
    df_out.to_csv(os.path.join(output_dir, 'obs_stats.csv'))
    df_out.to_file(os.path.join(output_dir, 'obs_stats.shp'))

    # Calculate avg stats for each HUC6
    df_out_stats = df_out[['mae', 'rmse', 'nse', 'bias', 'r2', 'pe', 'r', 'tpe', group_col]]
    grp_stats = df_out_stats.groupby([group_col]).mean().round(2)
    grp_stats.to_csv(os.path.join(output_dir, 'gage_stats_by_group.csv'), index_label=True, index=True)

    domain_stats = df_out[['mae', 'rmse', 'nse', 'bias', 'r2', 'pe', 'r', 'tpe']].mean().round(2)
    domain_stats.to_csv(os.path.join(output_dir, 'domain_stats.csv'), index_label=True, index=True)
    return df_out


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    out = LinearSegmentedColormap.from_list(cmap_name, color_list, N)
    return out


# Load data catalog and model
yml = r'Z:\users\lelise\data\data_catalog.yml'
cat = hydromt.DataCatalog(yml)
os.chdir(r'Z:\users\lelise\projects\NBLL\sfincs')
model_root = 'nbll_40m_sbg3m_v3_eff25'
mod = SfincsModel(root=model_root, mode='r')
mod.read_results()

createPlots = True
buffer_peak = None  # '20 days'
update_site = {'2088383': {'gage_height_ft': -2}}

# Loop through the agency
storm = 'florence'
count = 0
for agency in ['usgs', 'ncem']:
    dataset_name = agency + '_waterlevel_' + storm
    # Load the observation data from the data catalog for the model region and time
    obs1 = cat.get_geodataset(dataset_name,
                              geom=mod.region,
                              # buffer=None,
                              variables=["waterlevel"],
                              time_tuple=mod.get_model_time())

    pts, obs = clean_obs_coords(obs_df=obs1,
                                source_crs=4326,
                                target_crs=mod.crs.to_epsg())

    # Create empty lists/df to save information to when looping through the observation gages
    invalid_obs = []
    valid_obs = []
    station_stats = pd.DataFrame()

    # Create directories
    out_dir = os.path.join(os.getcwd(), model_root, 'validation', 'waterlevel', agency)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'figs')):
        os.makedirs(os.path.join(out_dir, 'figs'))

    # Loop through the observation locations and extract model data
    for gage in obs.index:
        print(gage.values.item())
        if str(gage.values.item()) in ['2130910', '2129000', '2102192', '2102000',
                                       '2098206', '208773375', '2087500', '2090380', '208250410']:
            print(gage.values.item(), " is boundary condition gage -- skipping.")

        else:
            obs_zs = obs.sel(index=gage.values.item())

            # Update observational data based on gage heights
            if str(gage.values.item()) in update_site.keys():
                datum_adj = update_site[str(gage.values.item())]['gage_height_ft'] * 0.3048  # convert to meters
                obs_zs[:] = obs_zs.values + datum_adj
                print('Adjusted data based on gage height for ', agency, str(gage.values.item()))

            # Extract water level from model output at gage location
            if 'geometry' in list(obs_zs.coords):
                mod_zs = mod.results['zs'].sel(x=obs_zs.geometry.values.item().x,
                                               y=obs_zs.geometry.values.item().y,
                                               method='nearest')
            else:
                mod_zs = mod.results['zs'].sel(x=obs_zs.x.values,
                                               y=obs_zs.y.values,
                                               method='nearest')

            # Add observed and modeled data into a single dataframe
            obs_df = pd.DataFrame(data=obs_zs.values, index=obs_zs.time.values, columns=['Observed'])
            mod_df = pd.DataFrame(data=mod_zs.values, index=mod_zs.time.values, columns=['Modeled'])
            merged_df = pd.concat([obs_df, mod_df], axis=1)
            merged_df.dropna(inplace=True)

            # If the dataframe is empty or there are fewer than 20 observation points,
            # append the gage ID to the list of "invalid_obs"
            if merged_df.empty or len(merged_df) < 20:
                print('No data for gage: ' + str(gage.values.item()))
                invalid_obs.append(gage.values.item())
            else:
                valid_obs.append(gage.values.item())

                # Calculate the hydrograph stats at the station and add to master dataframe
                if buffer_peak:
                    _, peak_dt = calculate_stats(df=merged_df, tstart=None, tend=None)
                    obs_pk, mod_pk = peak_dt
                    tstart0 = obs_pk - pd.to_timedelta(buffer_peak)
                    tend0 = obs_pk + pd.to_timedelta(buffer_peak)
                    ss, _ = calculate_stats(df=merged_df, tstart=tstart0, tend=tend0)
                    station_stats = pd.concat([station_stats, ss], ignore_index=True)
                else:
                    ss, _ = calculate_stats(df=merged_df, tstart=None, tend=None)
                    station_stats = pd.concat([station_stats, ss], ignore_index=True)

                # Plot the modeled and observed hydrographs
                if createPlots is True:
                    plot_hydrograph(df=merged_df,
                                    title=(agency + ' ' + str(gage.values.item())),
                                    figname=os.path.join(out_dir, 'figs', (agency + '_' + str(gage.values.item())))
                                    )

    # Output a shapefile of the locations where there was observational data to compare with modeled
    pts_out = pts[pts['site_no'].isin(valid_obs)]
    pts_out.reset_index(inplace=True, drop=True)
    pts_out.to_file(os.path.join(out_dir, (agency + '_obs.shp')))

    # Add the stat info and stats to master dataframe
    if count == 0:
        stats_out = pd.concat([pts_out, station_stats], axis=1)
        stats_out.columns = pts_out.columns.to_list() + station_stats.columns.to_list()
        count += 1
    else:
        stats_out2 = pd.concat([pts_out, station_stats], axis=1, ignore_index=True)
        stats_out2.columns = pts_out.columns.to_list() + station_stats.columns.to_list()
        stats_out = pd.concat([stats_out, stats_out2], axis=0)

# Calculate the hydrograph stats by zone and save output
grp_gdf = gpd.read_file(r'Z:\users\lelise\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
ss = stats_by_zone(gdf=grp_gdf[['Name', 'geometry']],
                   target_crs=mod.crs.to_epsg(),
                   stats_df=stats_out,
                   output_dir=os.path.join(out_dir, '../'),
                   group_col='Name')

domain_avg = stats_out.drop(['site_no', 'geometry'], axis=1)
domain_avg = round(domain_avg.mean(), 3)
domain_avg.to_csv(os.path.join(out_dir, '../', 'domain_avg_stats.csv'))

# Plotting the data on a map with contextual layers
coastal_wb = cat.get_geodataframe('carolinas_coastal_wb').to_crs(mod.crs).clip(mod.region)
major_rivers = cat.get_geodataframe('carolinas_major_rivers').to_crs(mod.crs).clip(mod.region)

da = mod.grid['dep']
wkt = da.raster.crs.to_wkt()
utm_zone = da.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
extent = np.array(mod.region.buffer(1000).total_bounds)[[0, 2, 1, 3]]

font = {'family': 'Arial',
        'size': 8}
mpl.rc('font', **font)

col = ['pe', 'bias', 'rmse', 'r2']
label = ['Peak Error', 'Bias', 'RMSE', 'R-Squared']
unit = ['(m)', '(m)', '(m)', '']
color_map = ['bwr', 'bwr', 'Reds', 'Reds_r']
n_bins_ranges = [[-0.5, 0.5], [-0.5, 0.5], [0, 1], [0, 1]]
ext = ['both', 'both', 'max', 'neither']

fig, axs = plt.subplots(
    nrows=2, ncols=2,
    figsize=(5, 5),
    subplot_kw={'projection': utm},
    tight_layout=True
)
axs = axs.flatten()

for i in range(len(col)):
    # Plot background/geography layers
    mod.region.plot(ax=axs[i], color='lightgray', edgecolor='none', linewidth=0, linestyle='-', zorder=2, alpha=0.75)
    coastal_wb.plot(ax=axs[i], color='steelblue', edgecolor='none', linewidth=0, linestyle='-', zorder=2, alpha=1)
    major_rivers.plot(ax=axs[i], color='steelblue', edgecolor='none', linewidth=0.5, linestyle='-', zorder=2, alpha=1)

    # Plot the stat at each gage
    vmax = round(max(n_bins_ranges[i]), 2)
    vmin = round(min(n_bins_ranges[i]), 2)
    ss.plot(column=col[i],
            cmap=color_map[i],
            legend=False,
            vmin=vmin, vmax=vmax,
            ax=axs[i],
            markersize=20,
            edgecolor='black',
            linewidth=0.5,
            zorder=3)

    #  Setup colorbar and add to plot
    sm = plt.cm.ScalarMappable(cmap=color_map[i], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm,
                 ax=axs[i],
                 shrink=0.6,
                 label=unit[i],
                 extend=ext[i],
                 spacing='uniform')

    # Add title and save figure
    axs[i].set_extent(extent, crs=utm)
    axs[i].set_title(label[i], loc='center')

    if i == 0:
        axs[i].set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
        axs[i].yaxis.set_visible(True)
        axs[i].xaxis.set_visible(False)
    elif i == 1:
        axs[i].yaxis.set_visible(False)
        axs[i].xaxis.set_visible(False)
    elif i == 2:
        axs[i].set_ylabel(f"Y Coord UTM zone {utm_zone} (m)")
        axs[i].yaxis.set_visible(True)
        axs[i].set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
        axs[i].xaxis.set_visible(True)
    elif i == 3:
        axs[i].yaxis.set_visible(False)
        axs[i].set_xlabel(f"X Coord UTM zone {utm_zone} (m)")
        axs[i].xaxis.set_visible(True)

    axs[i].ticklabel_format(style='sci', useOffset=False)
    axs[i].set_aspect('equal')
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(os.path.join(out_dir, '../water_level_gage_stats'), bbox_inches='tight', dpi=225)
plt.close()
