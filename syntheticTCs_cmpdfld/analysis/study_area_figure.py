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
#states_sub = states[states['STUSPS'].isin(['VA', 'NC', 'TN', 'WV', 'GA', 'KY', 'DC', 'SC', 'AL','FL','MD',])]
states.to_crs(epsg=32617, inplace=True)
states.set_index('STUSPS', inplace=True)

##################################################################################################################
import matplotlib.patches as mpatches
from matplotlib import patheffects
from matplotlib.lines import Line2D

unique_values = basins['nickname'].unique()
num_values = len(unique_values)
colors = ['mediumpurple', 'lightblue','darkorange','darkseagreen','lightcoral']
color_map = dict(zip(unique_values, colors))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), subplot_kw={'projection': utm}, tight_layout=True)
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
        alpha=0.9
    )
    ax.add_patch(patch)

for value in unique_values:
    legend_patches.append(mpatches.Patch(color=color_map[value], label=value))

# Add other layers and features
states.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, zorder=1)
mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=2, zorder=1)
legend_patches.append(mpatches.Patch(facecolor='none', edgecolor='black',linewidth=2, label='Model Domain'))
adcirc_locs.plot(ax=ax, color='lightgrey', edgecolor='black', zorder=2, markersize=18, label='ADCIRC Gages')
adcirc_patch = Line2D(
    [0], [0], marker='o', color='black',
    label='ADCIRC Gages',
    markerfacecolor='lightgrey', markeredgecolor='black',
    markersize=8, linestyle='None'
)
legend_patches.append(adcirc_patch)

# Update extent of figure
minx, miny, maxx, maxy = [300944, 3550131, 1101544, 4100000]
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

for label, grow in states.iterrows():
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

# Add title and save figure
ax.set_title('')
ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (meters)")
ax.yaxis.set_visible(True)
ax.set_xlabel(f"X Coord UTM zone {utm_zone} (meters)")
ax.xaxis.set_visible(True)
ax.ticklabel_format(style='sci', useOffset=False)
ax.set_aspect('equal')

# Add layer legend
# legend_kwargs0 = dict(
#     bbox_to_anchor=(1, 1),
#     title=None,
#     loc="upper left",
#     frameon=True,
#     prop=dict(size=10),
#     fontsize='small'
# )
# ax.legend(handles=legend_patches,**legend_kwargs0)
ax.legend(handles=legend_patches, loc='lower right', title=None, fontsize='medium')


plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\study_area_fig.png', dpi=300)
plt.close()



