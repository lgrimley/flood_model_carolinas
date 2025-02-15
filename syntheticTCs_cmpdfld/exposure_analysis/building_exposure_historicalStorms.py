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


''' 

Get total buildings flooded for historical storms 

'''
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter4_Exposure')
bld_tot_df_filpath = 'bld_fld_counts_FlorFloyMatt.csv'
if os.path.exists(bld_tot_df_filpath) is False:
    building_df = pd.read_csv('buildings_tc_exposure_rp_real.csv', index_col=0, low_memory=True)
    print(building_df.columns)

    depth_thresholds = [0.1, 0.25, 0.5, 0.64, 1 , 1.5, 2]
    storms = ['flor','floy','matt']
    building_totals = pd.DataFrame()
    for depth_threshold in depth_thresholds:
        for storm in storms:
            fld_build = building_df[building_df[[f'{storm}_compound_hzsmax',
                                                 f'{storm}_compound_pzsmax']].notna().all(axis=1)]
            for clim in ['h', 'p']:
                # Get the buildings that had a depth greater than the threshold for select climate
                flooded = fld_build[fld_build[f'{storm}_compound_{clim}depth'] > depth_threshold]

                # Count the classification of the flooding (runoff, coastal, compound)
                codes, counts = np.unique(flooded[f'{storm}_{clim}class'], return_counts=True)

                # Save these totals to a dataframe
                d = counts.tolist() + [clim, depth_threshold]
                counts_df = pd.DataFrame(data=d, index=codes.tolist() + ['period','hmin'])
                counts_df.columns = [f'{storm}_{clim}']

                # Append this dataframe to larger dataframe
                building_totals = pd.concat(objs=[building_totals, counts_df], axis=1, ignore_index=False)


    bld_tot_df = building_totals.T.drop(0.0, axis=1)
    bld_tot_df.columns = ['Coastal', 'Coastal-Comp', 'Runoff', 'Runoff-Comp', 'Period', 'hmin']
    bld_tot_df['Compound'] = bld_tot_df['Coastal-Comp'] + bld_tot_df['Runoff-Comp']
    bld_tot_df['Total'] = bld_tot_df['Compound']  + bld_tot_df['Coastal'] + bld_tot_df['Runoff']
    bld_tot_df.loc[bld_tot_df['Period'] == 'h','Period'] = 'Present'
    bld_tot_df.loc[bld_tot_df['Period'] == 'p','Period'] = 'Future'
    bld_tot_df.to_csv('bld_fld_counts_FlorFloyMatt.csv')
else:
    bld_tot_df = pd.read_csv(bld_tot_df_filpath, index_col=0)


bld_tot_df['Storm'] = [x.split('_')[0] for x in bld_tot_df.index]
storms = np.unique(bld_tot_df['Storm'])

# Mean Absolute Error depth threshold
subset = bld_tot_df[bld_tot_df['hmin'] == 0.64]
d1 = subset[subset['Period']=='Present'][['Coastal', 'Runoff', 'Compound']]
d2 = subset[subset['Period']=='Future'][['Coastal', 'Runoff', 'Compound']]
colors = ['#3F5565', '#879BEE', '#DD7596']#, '#8EB897']
bar_width = 0.3
index = np.arange(len(storms))
nrow, ncol = 1, 1
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5.5, 3), layout='constrained')
bars1 = d2.plot(kind='bar', stacked=True, ax=ax, position=-0.05, width=bar_width,
                legend=False, color=colors, align='center')
bars2 = d1.plot(kind='bar', stacked=True, ax=ax, position=1.05, width=bar_width,
                color=colors, align='center')
ax.set_xticks(index)
ax.set_xlim(-0.5,2.5)
ax.set_ylabel('No. of Buildings\nwhere Depth > 0.64m')
ax.set_xticklabels(['Florence', 'Floyd', 'Matthew'], rotation=0)
plt.tight_layout()
plt.savefig('flor_floy_matt_building_exposure_hmin0.64.png', dpi=300)
plt.close()

# Mean Absolute Error depth threshold
colors = ['#3F5565', '#879BEE', '#DD7596', '#8EB897']
bar_width = 0.3
index = np.arange(len(storms))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 4), sharex=True, sharey=True)
hmins = [0.1, 0.25, 0.5, 1]
axes=axes.flatten()
for i in range(len(hmins)):
    hmin = hmins[i]
    subset = bld_tot_df[bld_tot_df['hmin'] == hmin]

    d1 = subset[subset['Period']=='Present'][['Coastal', 'Runoff', 'Coastal-Comp', 'Runoff-Comp']]
    d2 = subset[subset['Period']=='Future'][['Coastal', 'Runoff', 'Coastal-Comp', 'Runoff-Comp']]
    ax = axes[i]
    bars1 = d2.plot(kind='bar', stacked=True, ax=ax, position=-0.05, width=bar_width,
                    legend=False, color=colors, align='center')
    if i == 3:
        ax.legend(bbox_to_anchor=(2.05, 1), loc='upper left')
    bars2 = d1.plot(kind='bar', stacked=True, ax=ax, position=1.05, width=bar_width,
                    legend = False, color=colors, align='center')
    ax.set_xticks(index)
    ax.set_title(f'Depth threshold: {hmin} m')
    ax.set_xlim(-0.5,2.5)
    ax.set_ylabel('No. of Buildings')
    ax.set_xticklabels(['Florence', 'Floyd', 'Matthew'], rotation=0)
plt.tight_layout()
plt.savefig('flor_floy_matt_building_exposure_allHmin_split.png', dpi=300)
plt.close()



''' 

Hexbin map of buildings exposure Historical Storms 

'''

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
print(building_df.columns)

depth_threshold=0.64
data_plot = []
var_plot = []
for storm in storms:
    fld_build = building_df[building_df[[f'{storm}_compound_hzsmax', f'{storm}_compound_pzsmax']].notna().all(axis=1)]
    for clim in ['h', 'p']:
        # Reclass the compound
        fld_build_sub = fld_build[fld_build[f'{storm}_compound_{clim}depth'] > depth_threshold]
        colname = f'{storm}_{clim}class'
        fld_build_sub.loc[fld_build_sub[colname] == 2.0, colname] = 5.0
        fld_build_sub.loc[fld_build_sub[colname] == 4.0, colname] = 5.0

        data_plot.append(fld_build_sub)
        var_plot.append(colname)

storms_label = ['Florence', 'Floyd', 'Matthew']
nrow = 3
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)


mincnt = 30
gridsize = 250
minx, miny, maxx, maxy = mod.region.total_bounds
xax_hbsize = (maxx - minx) / gridsize
yax_hbsize = (maxy - miny) / gridsize

fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5, 6), subplot_kw={'projection': utm})
axs = axs.flatten()
for i in range(len(axs)):
    ax = axs[i]
    d = data_plot[i]
    cmap = mpl.colors.ListedColormap(['#3F5565', '#879BEE', '#DD7596'])
    bounds = [0, 2, 4, 6]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    hb = ax.hexbin(d['xcoords'], d['ycoords'], C=d[var_plot[i]],
                   mincnt = mincnt,
                   #marginals=True,
                   gridsize=gridsize,
                   cmap=cmap, norm=norm,
                   reduce_C_function = np.mean,
                   extent = mod.region.total_bounds,
                   alpha=1, edgecolors='black', linewidth=0.1, zorder=1)
    mod.region.plot(ax=ax, color='white', edgecolor='none', zorder=0, alpha=1)
    major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=1)
    nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
    coastal_wb_clip.plot(ax=ax, color='white', edgecolor='slategrey', linewidth=0.25, zorder=0, alpha=0.75)
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.35, zorder=2, alpha=1)
    ax.set_axis_off()

    minx, miny, maxx, maxy = mod.region.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
pos0 = axs[3].get_position()  # get the original position
cax1 = fig.add_axes([pos0.x1 + 0.1 , pos0.y0 , 0.03, pos0.height * 1.1])
cbar1 = fig.colorbar(hb, cax=cax1, orientation='vertical', ticks=[1, 3, 5])
cbar1.ax.set_yticklabels(labels=['Coastal', 'Runoff', 'Compound'])
axs[0].set_title('Present')
axs[1].set_title('Future (4C)')
for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.05, 0.5, storms_label[i],
                               horizontalalignment='right',
                               verticalalignment='center',
                               rotation='vertical',
                               transform=axs[first_in_row[i]].transAxes)

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
plt.savefig(f'hexbin_mincnt{mincnt}_gridsize{gridsize}_map_florfloymatt_hmin{depth_threshold}_mean2.jpg', dpi=300, bbox_inches='tight')
plt.close()

