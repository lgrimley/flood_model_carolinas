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


def hwm_to_gdf(file, quality=None, dst_crs=None):
    df = pd.read_csv(file)
    gdf = gpd.GeoDataFrame(df,
                           geometry=gpd.points_from_xy(x=df['longitude'], y=df['latitude'], crs=4326))
    gdf['elev_m'] = gdf['elev_ft'] * 0.3048
    if quality:
        gdf = gdf[gdf['hwm_quality_id'] <= quality]
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


yml = r'Z:\users\lelise\data\data_catalog.yml'
cat = hydromt.DataCatalog(yml)
os.chdir(r'Z:\users\lelise\projects\NBLL\sfincs')
model_root = 'nbll_40m_sbg3m_v3_eff25'
storm = 'florence'
mod = SfincsModel(root=model_root, mode='r', data_libs=yml)
mod.read_results()

out_dir = os.path.join(os.getcwd(), model_root, 'validation', 'hwm')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
os.chdir(out_dir)

# Get gage peaks
pts_df = pd.DataFrame()
for agency in ['usgs']:
    dataset_name = agency + '_waterlevel_' + storm
    # Load the observation data from the data catalog for the model region and time
    obs1 = cat.get_geodataset(dataset_name,
                              geom=mod.region,
                              variables=["waterlevel"],
                              time_tuple=mod.get_model_time())

    pts, obs = clean_obs_coords(obs_df=obs1,
                                source_crs=4326,
                                target_crs=mod.crs.to_epsg())

    pts['elev_m'] = obs.max(dim='time').to_dataframe()['waterlevel'].to_list()
    pts_df = pd.concat([pts_df, pts], axis=0)

gage = extract_water_level(gdf=pts_df, raster=mod.results['zsmax'].max(dim='timemax'))

# Read HWM file, extract modeled water levels, calculate error
hwm_only = hwm_to_gdf(file=(r'Z:\users\lelise\geospatial\observations\usgs_' + storm + r'_FilteredHWMs.csv'),
                      quality=3, dst_crs=mod.crs.to_epsg())
hwm_only = extract_water_level(gdf=hwm_only, raster=mod.results['zsmax'].max(dim='timemax'))

# Merge gage and HWM
hwm = pd.concat([gage, hwm_only], axis=0)
hwm.to_csv('hwm_error_all_points.csv', index=False)

stats = pd.DataFrame(calc_stats(observed=hwm['elev_m'], modeled=hwm['sfincs_m'])).T
stats.columns = ['mae', 'rmse', 'bias']
stats.to_csv('hwm_stats.csv', index=True)

''' Q-Q PLOTS '''
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
axislim = [2, 12]
stp = 2
plt_qq = True
if plt_qq is True:
    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(3.5, 3.5))

    ax.scatter(hwm['elev_m'], hwm['sfincs_m'], color='grey', s=60, edgecolors='black', alpha=1.0, marker='o', zorder=2)

    line = mlines.Line2D([0, 1], [0, 1], color='black', alpha=0.8, linestyle='--', zorder=3)
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    ax.set_ylabel('Modeled WL\n(m +NAVD88)')
    ax.set_xlabel('Observed WL\n(m +NAVD88)')
    ax.set_title('')

    ax.set_xlim(axislim)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end + 1, stp))
    ax.set_ylim(axislim)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end + 1, stp))
    ax.grid(axis='both', alpha=0.7, zorder=-1)

    ss1 = 'Bias: ' + str(stats['bias'].item())
    ss2 = 'RMSE: ' + str(stats['rmse'].item())
    ax.text(x=axislim[1] - 0.15, y=axislim[0] + 1, s=ss1, ha='right', va='bottom')
    ax.text(x=axislim[1] - 0.15, y=axislim[0] + 0.1, s=ss2, ha='right', va='bottom')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.savefig('qq_plot.png', bbox_inches='tight', dpi=225)
    plt.close()

''' PLOT MAP '''
domain = mod.region
plt_hwm_map = True
if plt_hwm_map is True:
    wkt = mod.grid['dep'].raster.crs.to_wkt()
    utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
    utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
    extent = np.array(domain.buffer(1000).total_bounds)[[0, 2, 1, 3]]

    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(5, 4.5),
        subplot_kw={'projection': utm},
        tight_layout=True)

    cmap = mpl.cm.binary
    bounds = np.arange(-5, 15 + 1, 5)
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
    domain.plot(ax=ax, color='none', edgecolor='black', linewidth=1, linestyle='-', zorder=3, alpha=1)

    n_bins_ranges = [-1, -0.5, 0, 0.5, 1]
    vmax = round(max(n_bins_ranges), 0)
    vmin = round(min(n_bins_ranges), 0)
    hwm.plot(column='error',
             cmap='seismic',
             legend=False,
             vmin=vmin, vmax=vmax,
             ax=ax,
             markersize=20,
             alpha=0.9,
             edgecolor='black',
             linewidth=0.5,
             zorder=3)

    sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm,
                 ax=ax,
                 shrink=0.6,
                 extend='both',
                 spacing='uniform',
                 label='Water Level Diff (m)\n Modeled - Observed')

    ax.set_extent(extent, crs=utm)
    ax.set_ylabel(f"Y Coord UTM zone {utm_zone} [m]")
    ax.yaxis.set_visible(True)
    ax.set_xlabel(f"X Coord UTM zone {utm_zone} [m]")
    ax.xaxis.set_visible(True)
    ax.ticklabel_format(style='sci', useOffset=False)
    ax.set_aspect('equal')
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.savefig('hwm_error_map.png', bbox_inches='tight', dpi=255)  # , pil_kwargs={'quality': 95})
    plt.close()
