import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
import matplotlib.colors as mcolors
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True



cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
cat = mod.data_catalog
studyarea_gdf = mod.region.to_crs(epsg=32617)
da = mod.grid['dep']
wkt = da.raster.crs.to_wkt()
utm_zone = da.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)

load_geo_layers = True
if load_geo_layers is True:
    coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
    coastal_wb = coastal_wb.to_crs(mod.crs)
    coastal_wb_clip = coastal_wb.clip(mod.region)

    major_rivers = mod.data_catalog.get_geodataframe('carolinas_nhd_area_rivers')
    major_rivers = major_rivers.to_crs(mod.crs)
    major_rivers_clip = major_rivers.clip(mod.region)

    nc_major_rivers = mod.data_catalog.get_geodataframe('carolinas_major_rivers')
    nc_major_rivers = nc_major_rivers.to_crs(mod.crs)
    nc_major_rivers_clip = nc_major_rivers.clip(mod.region)

    urban_areas = mod.data_catalog.get_geodataframe(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\2010_Census_Urban_Areas\2010_Census_Urban_Areas.shp').to_crs(32617)
    urban_areas = urban_areas.clip(mod.region)

    tc_tracks = cat.get_geodataframe(r'Z:\Data-Expansion\users\lelise\data\geospatial\hurricane_tracks\IBTrACS.NA.list'
                                     r'.v04r00.lines\IBTrACS.NA.list.v04r00.lines.shp')
    tc_tracks.to_crs(epsg=32617, inplace=True)


os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure\historical_storms')
building_df = pd.read_csv('SFINCS_buildings_FlorFloyMatt.csv', index_col=0, low_memory=True)
depth_threshold=0.5

# 500m resolution, assuming CRS units are in meters
gridsize = 5000

# Get bounding box
minx, miny, maxx, maxy = mod.region.total_bounds

# Define bin edges at selected spacing
xedges = np.arange(minx, maxx + gridsize, gridsize)
yedges = np.arange(miny, maxy + gridsize, gridsize)

# Storm name or RP
for storm in ['flor','floy','matt']:
    if storm == 'flor':
        tc = tc_tracks[tc_tracks['NAME'] == 'FLORENCE']
    elif storm=='floy':
        tc = tc_tracks[tc_tracks['NAME'] == 'FLOYD']
    elif storm == 'matt':
        tc = tc_tracks[tc_tracks['NAME'] == 'MATTHEW']

    data_plot = []
    var_plot = []
    fld_build = building_df[building_df[[f'{storm}_compound_pres_zsmax', f'{storm}_compound_fut_zsmax']].notna().all(axis=1)]
    for clim in ['pres', 'fut']:
        # Reclass the compound
        fld_build_sub = fld_build[fld_build[f'{storm}_compound_{clim}_hmax'] > depth_threshold]
        colname = f'{storm}_{clim}_class'
        fld_build_sub.loc[fld_build_sub[colname] == 2.0, colname] = 5.0
        fld_build_sub.loc[fld_build_sub[colname] == 4.0, colname] = 5.0

        # Coordinates of buildings
        xcoords = fld_build_sub['xcoords']
        ycoords = fld_build_sub['ycoords']

        H, _, _ = np.histogram2d(xcoords, ycoords, bins=[xedges, yedges])
        data_plot.append(H)

    H_diff = data_plot[1] - data_plot[0]
    data_plot.append(H_diff)

    # Plot
    X, Y = np.meshgrid(xedges, yedges)
    titles = ['Present', 'Future', 'Future minus Present']
    nrow = 3
    ncol = 1
    n_subplots = nrow * ncol
    first_in_row = np.arange(0, n_subplots, ncol)
    last_in_row = np.arange(ncol - 1, n_subplots, ncol)
    first_row = np.arange(0, ncol)
    last_row = np.arange(first_in_row[-1], n_subplots, 1)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6, 8),
                             constrained_layout=True, subplot_kw={'projection': utm})
    axes= axes.flatten()
    for i in range(len(axes)):
        ax =axes[i]
        data = data_plot[i]
        data = np.where(np.abs(data) > 1, data, np.nan)

        if i <2:
            bounds = [1, 25, 50, 100, 200, 300, 400, 500]
            cmap = plt.get_cmap('Reds', len(bounds) +1)  # Discrete Reds with 8 colors
            norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='max')
            pcm = ax.pcolormesh(X, Y,
                                data.T,
                                cmap=cmap,
                                norm=norm,
                                rasterized=True,
                                zorder=3
                                )
        else:
            bounds2 = [-500, -100, -50, -25, 0, 25, 50, 100, 500]
            cmap = plt.get_cmap('bwr', len(bounds) + 2)  # Discrete Reds with 8 colors
            norm = mcolors.BoundaryNorm(boundaries=bounds2, ncolors=cmap.N, extend='neither')
            pcm2 = ax.pcolormesh(X, Y,
                                data.T,
                                cmap=cmap,
                                norm=norm,
                                rasterized=True,
                                zorder=3
                                )

        mod.region.plot(ax=ax, color='lightgrey', edgecolor='none', zorder=0, alpha=1)
        major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=0, alpha=1)
        nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=0, alpha=1)
        coastal_wb_clip.plot(ax=ax, color='darkgrey', edgecolor='black', linewidth=0.25, zorder=0, alpha=1)
        #urban_areas.plot(ax=ax, color='darkgrey', edgecolor='none', alpha=0, zorder=0)
        mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.35, zorder=2, alpha=1)
        tc.plot(ax=ax, color='black', zorder=2)
        ax.set_axis_off()
        minx, miny, maxx, maxy = mod.region.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(titles[i])
    cbar = fig.colorbar(pcm, ax=[axes[0], axes[1]], ticks=bounds, spacing='uniform',
                        extend='max',label='No. of Buildings',fraction=0.03, pad=0.02)
    cbar2 = fig.colorbar(pcm2, ax=axes[2], ticks=bounds2, spacing='uniform',
                         extend='both', label='Difference',fraction=0.03, pad=0.02)
    plt.subplots_adjust(right=0.85, wspace=0, hspace=0.0)
    plt.margins(x=0, y=0)
    plt.savefig(f'{storm}_building_exposure_grid_{gridsize}m.jpg', dpi=300, bbox_inches='tight')
    plt.close()

