
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
from hydromt_sfincs import SfincsModel

sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True



os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure')
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

    # urban_areas = gpd.read_file(r'Z:\Data-Expansion\users\lelise\data\geospatial\boundary\2010_Census_Urban_Areas\2010_Census_Urban_Areas.shp').to_crs(32617)
    # urban_areas = urban_areas.clip(mod.region)

    # tc_tracks = cat.get_geodataframe(r'Z:\Data-Expansion\users\lelise\data\geospatial\hurricane_tracks\IBTrACS.NA.list'
    #                                  r'.v04r00.lines\IBTrACS.NA.list.v04r00.lines.shp')
    # tc_tracks.to_crs(epsg=32617, inplace=True)

building_df = pd.read_csv('buildings_tc_exposure_rp_real.csv', index_col=0, low_memory=True)


depth_threshold=0.5
storm='flor'
clim='h'
fld_build = building_df[building_df[[f'{storm}_compound_hzsmax', f'{storm}_compound_pzsmax']].notna().all(axis=1)]
fld_build_sub = fld_build[fld_build[f'{storm}_compound_{clim}depth'] > depth_threshold]
colname = f'{storm}_{clim}class'
fld_build_sub.loc[fld_build_sub[colname] == 2.0, colname] = 5.0
fld_build_sub.loc[fld_build_sub[colname] == 4.0, colname] = 5.0

# Plotting map hexbin below
nrow = 1
ncol = 1
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)
mincnt = 3
gridsize = 700
minx, miny, maxx, maxy = mod.region.total_bounds
xax_hbsize = (maxx - minx) / gridsize
yax_hbsize = (maxy - miny) / gridsize

fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5, 4), subplot_kw={'projection': utm})
d = fld_build_sub
hb = ax.hexbin(d['xcoords'], d['ycoords'],
               mincnt = mincnt,
               gridsize=gridsize,
               cmap="viridis",
               bins='log',
               reduce_C_function = np.mean,
               extent = mod.region.total_bounds,
               alpha=1, edgecolors='black', linewidth=0.1, zorder=1)
mod.region.plot(ax=ax, color='white', edgecolor='none', zorder=0, alpha=1)
major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=1)
nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
coastal_wb_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.35, zorder=2, alpha=1)
ax.set_axis_off()

pos0 = ax.get_position()  # get the original position
cax1 = fig.add_axes([pos0.x1 +  pos0.x1*0.7 , pos0.y0 +0.05 , 0.03, pos0.height * 0.9])
cbar1 = fig.colorbar(hb, cax=cax1, orientation='vertical', label='No. of Buildings')

minx, miny, maxx, maxy = mod.region.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
plt.savefig(f'FLORENCE_HEXBIN_NOAA_PROPOSAL.jpg', dpi=300, bbox_inches='tight')
plt.close()

