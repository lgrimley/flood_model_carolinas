import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel
import geopandas as gpd
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
water_mask_file = rf'.\04_MODEL_OUTPUTS\masks/water_mask_sbgRes{res}m.tif'
water_mask = cat.get_rasterdataset(data_like=water_mask_file)
water_mask = water_mask.where(water_mask==1)
# Load the basins geodataframe
basins = cat.get_geodataframe(data_like=rf'.\04_MODEL_OUTPUTS\masks\basins_shp\huc6_basins.shp')
basins = basins.to_crs(epsg=proj_crs)
basins = basins.clip(mod.region)

# Load gage points
dis_pts = mod.forcing['dis'].to_dataframe().reset_index()['geometry'].unique()
gdf = gpd.GeoDataFrame(geometry=dis_pts).set_crs(32617, inplace=True)
bzs_pts = cat.get_geodataframe(fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\stormTide\coastline_lores_NCSC.shp').to_crs(32617)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), subplot_kw={'projection': utm}, tight_layout=True)
water_mask.plot(ax=ax, zorder=0, add_colorbar=False)
basins.plot(ax=ax, color='none', edgecolor='black')
gdf.plot(ax=ax, color='red', edgecolor='black', label='USGS Locs')
bzs_pts.plot(ax=ax, color='orange', edgecolor='black', label='ADCIRC Locs')

# Add title and save figure
ax.set_title('')
ax.set_ylabel(f"Y Coord UTM zone {utm_zone} (meters)")
ax.yaxis.set_visible(True)
ax.set_xlabel(f"X Coord UTM zone {utm_zone} (meters)")
ax.xaxis.set_visible(True)
ax.ticklabel_format(style='sci', useOffset=False)
ax.set_aspect('equal')

plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\bc_waterMask_figure.png',
            bbox_inches='tight',
            dpi=300)
plt.close()



