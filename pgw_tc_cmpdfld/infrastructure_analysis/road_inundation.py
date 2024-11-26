import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
import numpy as np

cat_dir = r'Z:\Data-Expansion\users\lelise\data'
yml_base_CONUS = os.path.join(cat_dir, 'data_catalog_BASE_CONUS.yml')
yml_base_Carolinas = os.path.join(cat_dir, 'data_catalog_BASE_Carolinas.yml')
yml_sfincs_Carolinas = os.path.join(cat_dir, 'data_catalog_SFINCS_Carolinas.yml')
os.chdir(r'Z:\Data-Expansion\users\lelise\projects')
root = r'.\Carolinas_SFINCS\Chapter2_PGW\sfincs\01_AGU2023\future_florence\future_florence_ensmean'
mod = SfincsModel(root=root, mode='r', data_libs=[yml_base_CONUS, yml_base_Carolinas, yml_sfincs_Carolinas])
cat = mod.data_catalog
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

    urban_areas = gpd.read_file(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\2010_Census_Urban_Areas\2010_Census_Urban_Areas.shp').to_crs(32617)
    urban_areas = urban_areas.clip(mod.region)

    tc_tracks = cat.get_geodataframe(r'Z:\Data-Expansion\users\lelise\data\geospatial\hurricane_tracks\IBTrACS.NA.list'
                                     r'.v04r00.lines\IBTrACS.NA.list.v04r00.lines.shp')
    tc_tracks.to_crs(epsg=32617, inplace=True)

''' Load the road data '''
os.chdir(r'.\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_3\infrastructure_exposure')
# Present Florence
pres_road = pd.read_csv('floodedroadareas_presFlor.csv', index_col=0)
pres_road.set_index(keys='County', drop=True, inplace=True)
pres_road.dropna(axis=0, inplace=True)

# Future Florence
fut_road = pd.read_csv('floodedroadareas_futFlor.csv', index_col=0)
fut_road.set_index(keys='County', drop=True, inplace=True)
fut_road.columns = [f'{s}_fut' for s in fut_road.columns]
fut_road.dropna(axis=0, inplace=True)

# County shapefile
county_gdf = gpd.read_file(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\nc_county\North_Carolina_State_and_County_Boundary_Polygons\North_Carolina_State_and_County_Boundary_Polygons.shp')
county_gdf.set_index(keys='County', drop=True, inplace=True)

# Combine all the data together
joined = pd.concat(objs=[pres_road, fut_road, county_gdf], ignore_index=False, axis=1)
joined = gpd.GeoDataFrame(joined).to_crs(32617)
joined = joined.dropna(axis=0, subset=joined.columns[0])

subset = joined[(joined[pres_road.columns[0]] > 0.0) | (joined[fut_road.columns[0]] > 0.0)]
subset = np.round(subset, 3)

subset['area_diff'] = subset[fut_road.columns[0]] - subset[pres_road.columns[0]]
subset['area_diff_rel'] = (subset['area_diff'] / subset[pres_road.columns[0]])+1
subset['area_diff_rel'][subset['area_diff_rel'] == np.inf] = 1 + subset['area_diff'][subset['area_diff_rel'] == np.inf]
#subset['evac_area_diff'] = subset[fut_road.columns[1]] - subset[pres_road.columns[1]]

pres = subset[pres_road.columns[0]][subset[pres_road.columns[0]] > 0.0]
pres_evac = subset[pres_road.columns[1]][subset[pres_road.columns[1]] > 0.0]
print(pres.describe())
print(pres.sum())
print(pres_evac.describe())
print(pres_evac.sum())

fut = subset[fut_road.columns[0]][subset[fut_road.columns[0]] > 0.0]
fut_evac = subset[fut_road.columns[1]][subset[fut_road.columns[1]] > 0.0]
print(fut.describe())
print(fut.sum())
print(fut_evac.describe())
print(fut_evac.sum())

# PLOTTING
please_plot = True
figname = 'pres vs. future county total road flood area - evac routes'
if please_plot is True:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), subplot_kw={'projection': utm}, tight_layout=True)
    axs =axs.flatten()
    ax =axs[0]
    vmin = 0.1
    vmax = 1
    cs = subset.plot(ax=ax, column=pres_road.columns[1],
                     cmap='Reds',
                     legend=False,
                     zorder=0,
                     vmin=vmin, vmax=vmax)
    ax.set_title('Present')

    ax=axs[1]
    cs = subset.plot(ax=ax, column=fut_road.columns[1],
                     cmap='Reds',
                     legend=False,
                     zorder=0,
                     vmin=vmin, vmax=vmax)
    ax.set_title('Future')

    for i in range(len(axs)):
        ax =axs[i]
        major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=1, alpha=1)
        nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=1, alpha=1)
        coastal_wb_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=1, alpha=0.75)
        urban_areas.plot(ax=ax, color='grey', alpha=0.7)
        mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=1, alpha=1)
        tc_tracks[(tc_tracks['NAME'] == 'FLORENCE') & (tc_tracks['SEASON'] == 2018)].plot(ax=ax, color='red',
                                                                                          edgecolor='none',
                                                                                          label='Florence\nTrack',
                                                                                          linewidth=2, linestyle='-',
                                                                                          zorder=3, alpha=0.80)
        ax.set_aspect('equal')
        ax.set_axis_off()
        minx, miny, maxx, maxy = mod.region.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    pos0 = axs[1].get_position()
    cax = fig.add_axes([pos0.x1 + 0.12, pos0.y0, 0.03, pos0.height * 0.8])
    cbar = fig.colorbar(sm,
                        cax=cax,
                        orientation='vertical',
                        extend='max', label = 'Total Area (sq.km)'
                        )

    plt.savefig(f'{figname}.png', dpi=300, bbox_inches="tight")
    plt.close()

# PLOTTING
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), subplot_kw={'projection': utm}, tight_layout=True)
joined.plot(ax=ax, column='evac_area_diff', cmap='Reds', legend=True, zorder=0, vmin= 0.2, vmax=1,
            legend_kwds={'label':"Flooded Evacuation Road Area (sq.km.)",'orientation':"vertical"})
major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=1, alpha=1)
nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=1, alpha=1)
coastal_wb_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=1, alpha=0.75)
urban_areas.plot(ax=ax, color='grey',alpha=0.7)
mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=1, alpha=1)
ax.set_title('')
ax.set_aspect('equal')
ax.set_axis_off()
minx, miny, maxx, maxy = mod.region.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
plt.savefig('Increase in flooded evacuation road area (sq.km.).png', dpi=300, bbox_inches="tight")
plt.close()




