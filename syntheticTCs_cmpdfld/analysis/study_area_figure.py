import xarray as xr
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
#from matplotlib import patheffects

sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()


os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

# Plot
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL_RUNS\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
cat = mod.data_catalog
# Plotting details
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

##################################################################################################################
#Load in data catalog and all the masks
res = 200
proj_crs = 32617
elevation_file = rf'.\03_MODEL_RUNS/subgrid/dep_subgrid_{res}m.tif'
water_mask = rf'.\04_MODEL_OUTPUTS\masks/water_mask_sbgRes{res}m.tif'

# Load the basins geodataframe
basins = cat.get_geodataframe(data_like=rf'.\04_MODEL_OUTPUTS\masks\basins_shp\huc6_basins.shp')
clip_geom = basins.to_crs(epsg=proj_crs)

# Chunk size specified for dask
chunks_size = {'x': 5000, 'y': 5000}

# Load the elevation dataarray
elevation_da = mod.grid['dep']
elevation_da.name = 'gnd_elevation'

# Load layers - run once because it takes a while...
coastal_wb = mod.data_catalog.get_geodataframe('carolinas_coastal_wb')
coastal_wb = coastal_wb.to_crs(mod.crs)
coastal_wb_clip = coastal_wb.clip(mod.region)

major_rivers = mod.data_catalog.get_geodataframe('carolinas_nhd_area_rivers')
major_rivers = major_rivers.to_crs(mod.crs)
major_rivers_clip = major_rivers.clip(mod.region)

nc_major_rivers = mod.data_catalog.get_geodataframe('carolinas_major_rivers')
nc_major_rivers = nc_major_rivers.to_crs(mod.crs)
nc_major_rivers_clip = nc_major_rivers.clip(mod.region)

# Load the ADCIRC poinst
adcirc_locs = cat.get_geodataframe(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.shp',
                                   geom=clip_geom.buffer(distance=15*1000))
adcirc_locs = adcirc_locs.to_crs(32617)

l_gdf = cat.get_geodataframe(
    r'Z:\Data-Expansion\users\lelise\data\geospatial\hydrography\nhd\NHD_H_North_Carolina_State_Shape\Shape\WBDHU6.shp')
l_gdf = l_gdf[l_gdf['Name'].isin(['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee'])]
l_gdf.to_crs(epsg=32617, inplace=True)
l_gdf.set_index('Name', inplace=True)
basins = l_gdf
basins['nickname'] = ['Pamlico', 'Neuse', 'Onslow Bay', 'Cape Fear', 'Lower Pee Dee']

states = cat.get_geodataframe(
    r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\us_boundary\cb_2018_us_state_500k\cb_2018_us_state_500k.shp')
states_sub = states[states['STUSPS'].isin(['NC', 'SC'])]
states_sub.to_crs(epsg=32617, inplace=True)
states_sub.set_index('STUSPS', inplace=True)

census_areas = mod.data_catalog.get_geodataframe(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\2024_Census_Urban_Areas\tl_2024_us_uac20\tl_2024_us_uac20.shp')
census_areas = census_areas.to_crs(mod.crs)
census_areas = census_areas.clip(mod.region)

l_gdf = cat.get_geodataframe(r'Z:\Data-Expansion\users\lelise\data\geospatial\infrastructure\enc_major_cities.shp')
l_gdf = l_gdf[l_gdf['Name'].isin(
     ['New Bern', 'Wilmington', 'Jacksonville', 'Myrtle Beach', 'Morehead City'])]
l_gdf['label'] = ['A', 'B', 'C' ,'D', 'E']
l_gdf.set_index('label', inplace=True)
l_gdf.to_crs(epsg=32617, inplace=True)
cities = l_gdf.clip(mod.region)
##################################################################################################################
import matplotlib.patches as mpatches
from matplotlib import patheffects
from matplotlib.lines import Line2D

unique_values = basins['nickname'].unique()
num_values = len(unique_values)
colors = ['mediumpurple', 'lightblue','darkorange','darkseagreen','lightcoral']
color_map = dict(zip(unique_values, colors))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), subplot_kw={'projection': utm}, tight_layout=True)
# Plot each basin
legend_patches = []
for index, row in basins.iterrows():
    value = row['nickname']
    geo = row['geometry']

    # Create and add the colored polygon patch
    patch = mpatches.Polygon(
        list(geo.exterior.coords),
        closed=True,
        facecolor=color_map[value],
        edgecolor='black',
        linewidth=0.5,
        label=value,
        alpha=0.5
    )
    ax.add_patch(patch)

for value in unique_values:
    legend_patches.append(mpatches.Patch(color=color_map[value], label=value))

# Add other layers and features
census_areas.plot(ax=ax, color='grey', edgecolor='black', linewidth=0.08, zorder=1, alpha=0.7)
states_sub.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1)
major_rivers_clip.plot(ax=ax, color='none', edgecolor='darkblue', linewidth=0.25, zorder=2, alpha=0.7)
nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='darkblue', linewidth=0.25, zorder=2, alpha=0.7)
coastal_wb_clip.plot(ax=ax, color='steelblue', edgecolor='darkblue', linewidth=0.25, zorder=2, alpha=0.7)
basins.plot(ax=ax, color='none', edgecolor='black', linewidth=0.35, zorder=2)
mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=1, zorder=2)
legend_patches.append(mpatches.Patch(facecolor='none', edgecolor='black',linewidth=2, label='Model Domain'))
legend_patches.append(mpatches.Patch(color='grey', label='Urban Areas'))
cities.plot(ax=ax, marker='o', markersize=25, color="black",
            edgecolor='white', linewidth=0.75, label='Cities', zorder=4)
#adcirc_locs.plot(ax=ax, color='red', edgecolor='none', zorder=2, markersize=12, label='Storm tide Locs', alpha=1)
adcirc_patch = Line2D(
    [0], [0], marker='o', color='black',
    label='Cities',
    markerfacecolor='black', #markeredgecolor='white',
    markersize=6, linestyle='None'
)
legend_patches.append(adcirc_patch)


# Update extent of figure
minx, miny, maxx, maxy = [185000, 3540000, 1150000, 4070000] #states_sub.buffer(10**4.5).total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

for label, grow in states_sub.iterrows():
    if label in ['SC', 'NC',]:
        if label == 'NC':
            ll = -40
        elif label == 'SC':
            ll = -20
        ann_kwargs = dict(xytext=(ll, 0), textcoords="offset points", zorder=4,
                          path_effects=[
                              patheffects.Stroke(linewidth=2, foreground="white", alpha=1),
                              patheffects.Normal(), ], )
        x, y = grow.geometry.centroid.x, grow.geometry.centroid.y
        ax.annotate(f'{label}', xy=(x, y), weight='bold', **ann_kwargs)

# Label Cities
for label, grow in cities.iterrows():
    x, y = grow.geometry.x, grow.geometry.y
    ann_kwargs = dict(
        xytext=(-10, 2),
        #xytext=(-50, 5),
        textcoords="offset points",
        zorder=4,
        path_effects=[
            patheffects.Stroke(linewidth=2, foreground="white", alpha=1),
            patheffects.Normal(), ], )
    ax.annotate(f'{label}', xy=(x, y), **ann_kwargs)

# Add title and save figure
ax.set_title('')
ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (meters)")
ax.yaxis.set_visible(True)
ax.set_xlabel(f"X Coord UTM zone {utm_zone} (meters)")
ax.xaxis.set_visible(True)
ax.ticklabel_format(style='sci', useOffset=False)
ax.set_aspect('equal')

ax.legend(handles=legend_patches, loc='lower right', title=None, fontsize='medium')

plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\study_area_fig.png',
            bbox_inches='tight',
            dpi=300)
plt.close()



