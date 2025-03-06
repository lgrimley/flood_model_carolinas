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
building_df = pd.read_csv('buildings_rps_exposure.csv', index_col=0, low_memory=True)
print(building_df.columns)
bld_tot_df_filpath = 'bld_fld_counts_rps.csv'
if os.path.exists(bld_tot_df_filpath) is False:
    building_totals = pd.DataFrame()
    for T in [10, 25,100, 250, 500]:
        fld_build = building_df[building_df[[f'hist_rp{T}_zsmax', f'fut_rp{T}_zsmax']].notna().all(axis=1)]
        depth_thresholds = [0.1, 0.25, 0.5, 0.64, 1 , 1.5, 2]
        for depth_threshold in depth_thresholds:
            for clim in ['fut', 'hist']:
                # Get the buildings that had a depth greater than the threshold for select climate
                flooded = fld_build[fld_build[f'{clim}_rp{T}_depth'] > depth_threshold]

                # Count the classification of the flooding (runoff, coastal, compound)
                codes, counts = np.unique(flooded[f'{clim}_rp{T}_class'], return_counts=True)

                # Save these totals to a dataframe
                d = counts.tolist() + [clim, depth_threshold]
                counts_df = pd.DataFrame(data=d, index=codes.tolist() + ['period','hmin'])
                counts_df.columns = [f'{clim}_rp{T}']

                # Append this dataframe to larger dataframe
                building_totals = pd.concat(objs=[building_totals, counts_df], axis=1, ignore_index=False)

    bld_tot_df = building_totals.T.drop(columns=0)
    bld_tot_df.columns = ['Coastal', 'Coastal-Comp', 'Runoff', 'Runoff-Comp', 'Period', 'hmin']
    bld_tot_df['Compound'] = bld_tot_df['Coastal-Comp'] + bld_tot_df['Runoff-Comp']
    bld_tot_df['Total'] = bld_tot_df['Compound']  + bld_tot_df['Coastal'] + bld_tot_df['Runoff']
    bld_tot_df.loc[bld_tot_df['Period'] == 'hist','Period'] = 'Present'
    bld_tot_df.loc[bld_tot_df['Period'] == 'fut','Period'] = 'Future'
    bld_tot_df['RP'] = [x.split('_rp')[-1] for x in bld_tot_df.index]
    bld_tot_df.to_csv('bld_fld_counts_rps.csv')
else:
    bld_tot_df = pd.read_csv(bld_tot_df_filpath, index_col=0)


# Flood extent
fld_extent = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_attribution\aep_extent_rel_contribution_Wbmasked.csv',
                                 index_col=0)
fld_extent = fld_extent.T
fld_extent = fld_extent[['Coastal','Runoff','Compound']]

fld_extent['rp'] = [int(x.split('_')[-1]) for x in fld_extent.index]
fld_extent['period'] = [x.split('_')[0] for x in fld_extent.index]
fld_extent.set_index('rp', drop=True,inplace=True)


storms = [100, 250, 500]
subset = bld_tot_df[bld_tot_df['hmin'] == 0.64]
d1 = subset[subset['Period']=='Present'][['Coastal', 'Runoff', 'Compound']]
d1 = d1[2:]
tot1 = d1.sum(axis=1)
rc1 = d1.div(tot1, axis=0)
d2 = subset[subset['Period']=='Future'][['Coastal', 'Runoff', 'Compound']]
d2 = d2[2:]
tot2 = d2.sum(axis=1)
rc2 = d2.div(tot2, axis=0)

fld_extent1 = fld_extent[fld_extent['period'] == 'hist']
fld_extent1 = fld_extent1[fld_extent1.index.isin(storms)][['Coastal', 'Runoff', 'Compound']]
tot1 = fld_extent1.sum(axis=1)
rc_fld1 = fld_extent1.div(tot1, axis=0)
fld_extent2 = fld_extent[fld_extent['period'] == 'proj']
fld_extent2 = fld_extent2[fld_extent2.index.isin(storms)][['Coastal', 'Runoff', 'Compound']]
tot2 = fld_extent2.sum(axis=1)
rc_fld2 = fld_extent2.div(tot2, axis=0)

colors = ['#3F5565', '#879BEE', '#DD7596']#, '#8EB897']
bar_width = 0.3
index = np.arange(len(storms))
nrow, ncol = 1, 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_row = np.arange(n_subplots - ncol, n_subplots, 1)
fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6, 3), layout='constrained',
                         sharey=False, sharex=True)

ax = axes[0]
bars1 = fld_extent2.plot(kind='bar', stacked=True, ax=ax, position=-0.05, width=bar_width, #edgecolor='red',
                legend=False, color=colors, align='center')
bars2 = fld_extent1.plot(kind='bar', stacked=True, ax=ax, position=1.05, width=bar_width, #edgecolor='black',
                color=colors, align='center', legend=False)
ax.set_xticks(index)
ax.set_xlabel('')
ax.set_xlim(-0.5,2.5)
ax.set_ylim(0,80000)
ax.set_ylabel('Flood Extent (sq.km.)')
ax.set_xticklabels(['1.0%', '0.4%', '0.2%'], rotation=0)

ax = axes[1]
bars1 = d2.plot(kind='bar', stacked=True, ax=ax, position=-0.05, width=bar_width,
                legend=False, color=colors, align='center')
bars2 = d1.plot(kind='bar', stacked=True, ax=ax, position=1.05, width=bar_width,
                color=colors, align='center')
ax.set_xticks(index)
ax.set_xlabel('')
ax.set_xlim(-0.5,2.5)
ax.set_ylim(0,105000)
ax.set_ylabel('No. of Buildings\nwhere Depth > 0.64m')
ax.set_xticklabels(['1.0%', '0.4%', '0.2%'], rotation=0)
plt.tight_layout()
#plt.savefig('rps_building_exposure_hmin0.64.png', dpi=300)
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


#building_df = pd.read_csv('buildings_rps_exposure.csv', index_col=0, low_memory=True)
print(building_df.columns)

depth_threshold=0.64
data_plot = []
var_plot = []
for T in [100, 250, 500]:
    fld_build = building_df[building_df[[f'hist_rp{T}_zsmax', f'fut_rp{T}_zsmax']].notna().all(axis=1)]
    for clim in ['hist', 'fut']:
        # Reclass the compound
        fld_build_sub = fld_build[fld_build[f'{clim}_rp{T}_depth'] > depth_threshold]
        colname = f'{clim}_rp{T}_class'
        fld_build_sub.loc[fld_build_sub[colname] == 2.0, colname] = 5.0
        fld_build_sub.loc[fld_build_sub[colname] == 4.0, colname] = 5.0

        data_plot.append(fld_build_sub)
        var_plot.append(colname)
    break

a0 = fld_build
a1 = data_plot[0]
a2 = data_plot[1]

b0 = a2[a2[f'hist_rp100_depth'] < depth_threshold]
v, counts = np.unique(b0[f'fut_rp100_class'], return_counts=True)

b1 = a2[a2[f'hist_rp100_depth'] > depth_threshold]
v, counts = np.unique(b1[f'fut_rp100_class'], return_counts=True)
v2, counts2 = np.unique(b1[f'hist_rp100_class'], return_counts=True)


# plotting
storms_label = ['1.0%', '0.4%', '0.2%']
nrow = 3
ncol = 2
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
axs[0].set_title('Historic (1980-2005)')
axs[1].set_title('Future (2070-2100)')
for i in range(len(first_in_row)):
    axs[first_in_row[i]].text(-0.05, 0.5, storms_label[i],
                               horizontalalignment='right',
                               verticalalignment='center',
                               rotation='vertical',
                               transform=axs[first_in_row[i]].transAxes)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.margins(x=0, y=0)
#plt.savefig(f'hexbin_mincnt{mincnt}_gridsize{gridsize}_map_100yr_hmin{depth_threshold}_rps.jpg', dpi=300, bbox_inches='tight')
plt.close()

