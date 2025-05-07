import xarray as xr
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
import seaborn as sns
import scipy.io as sio
from src.utils import track_points_to_linestring

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS')
period = 'future'
rp_threshold = 80
# Now just look at the storms we already have
rp_file = fr'.\results_jointprob\basin_data\Domain_data_rp_{period}.csv'
rp_data = pd.read_csv(rp_file, index_col=0)
fut_tcs = rp_data[rp_data[f'Compound_rp'] > rp_threshold]

# Load in the flood extent calculation for these events
fld_area_df = pd.read_csv(r'.\SLR_analysis\canesm_ssp585_SRL112cm\hmin0.05\overland_flooded_area_table.csv',
                          index_col=0)
fld_area_df.index = fld_area_df.index.astype(str)
fld_area_df['AOI'] = ["".join(x.split()) for x in fld_area_df['AOI']]
fld_area_df2 = fld_area_df[['Coastal', 'Runoff', 'Compound','Total_Flooded', 'AOI']] #[fld_area_df['AOI'] == 'Domain']
mask = fld_area_df2.index.str.contains('SLR')
noslr_fld_area_df2 = fld_area_df2[~mask]
noslr_fld_area_df2.index = noslr_fld_area_df2.index.astype(int)
noslr_fld_area_df2['Scenario'] = 'NoSLR'
noslr_fld_area_df2['tc_id'] = noslr_fld_area_df2.index

slr_fld_area_df2 = fld_area_df2[mask]
slr_fld_area_df2.index = [x.split('_')[0] for x in slr_fld_area_df2.index.values]
slr_fld_area_df2.index = slr_fld_area_df2.index.astype(int)
slr_fld_area_df2['Scenario'] = 'SLR'
slr_fld_area_df2['tc_id'] = slr_fld_area_df2.index

t = noslr_fld_area_df2.groupby('AOI').describe()
tt = t.T
tt.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\SLR_analysis\fld_extent_stats_noSLR.csv')
# # Calculate the relative contribution of each flood processes relative to the total
# total_area = fld_area_df2.sum(axis=1)
# fld_area_fraction = fld_area_df2.div(total_area, axis=0)
#
# # Subset the runs into the TCs w/out and w/ SLR for comparison
# mask = fld_area_fraction.index.str.contains('SLR')
# noslr_fld_area_fraction = fld_area_fraction[~mask]
# noslr_fld_area_fraction.index = noslr_fld_area_fraction.index.astype(int)
#
# slr_fld_area_fraction = fld_area_fraction[mask]
# slr_fld_area_fraction.index = noslr_fld_area_fraction.index.astype(int)
#
# # Add the RP info for the storm
# noslr_fld_area_fraction = pd.concat(objs=[noslr_fld_area_fraction, fut_tcs['Compound_rp']], axis=1, ignore_index=False)
# noslr_fld_area_fraction.set_index('Compound_rp', inplace=True, drop=True)
# noslr_fld_area_fraction = noslr_fld_area_fraction.sort_index(ascending=True)
#
# slr_fld_area_fraction = pd.concat(objs=[slr_fld_area_fraction, fut_tcs['Compound_rp']], axis=1, ignore_index=False)
# slr_fld_area_fraction.set_index('Compound_rp', inplace=True, drop=True)
# slr_fld_area_fraction = slr_fld_area_fraction.sort_index(ascending=True)

# Violin Plot of Areas

violin_plot = True
if violin_plot is True:
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True
    basin_order = ['LPD', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico', 'Domain']
    scenarios = ['Runoff','Coastal', 'Compound', 'Total_Flooded']
    fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(5.5, 7), sharey=False, sharex=True)
    axs = axs.flatten()
    for i in range(len(scenarios)):
        ax =axs[i]
        scenario = scenarios[i]
        df_combined = pd.concat([noslr_fld_area_df2[['tc_id', 'AOI', scenario, 'Scenario']],
                                 slr_fld_area_df2[['tc_id', 'AOI', scenario, 'Scenario']]])

        df_combined['AOI'] = df_combined['AOI'].replace('LowerPeeDee', 'LPD')
        df_long = pd.melt(df_combined, id_vars=['tc_id', 'AOI', 'Scenario'], value_vars=[scenario],
                          var_name='data_type', value_name='data_value')
        df_long.dropna(axis=0, inplace=True)

        # Violin plot
        sns.violinplot(x='AOI', y='data_value', hue='Scenario', data=df_long,
                       ax=ax,
                       order=basin_order,
                       split=True, fill=True, gap=0.05,
                       log_scale=True, density_norm='count', linecolor='none', zorder=1,
                       palette={'NoSLR': 'silver', 'SLR': 'grey'},
                       inner_kws=dict(box_width=5, whis_width=1.5, color="black"))

        pos = ax.get_position()  # Get the axis position

        # # Calculate the mean values for each group and plot them as diamonds
        # means_by_period = df_long.groupby(['AOI', 'Scenario'])['data_value'].mean().reset_index()
        # basin_positions = {'Domain': 5, 'Pamlico': 4, 'Neuse': 3, 'OnslowBay': 2, 'CapeFear': 1, 'LPD': 0}
        # # Plot the mean values as diamond markers for each period (source)
        # for _, row in means_by_period.iterrows():
        #     basin = row['AOI']
        #     mean_value = row['data_value']
        #     source = row['Scenario']
        #     if source == 'NoSLR':
        #         ax.scatter(basin_positions[basin]-0.06, mean_value + pos.height,
        #                     color='white', edgecolor='black', alpha=0.8, marker='D', s=30, zorder=3)
        #     else:
        #         ax.scatter(basin_positions[basin]+0.06, mean_value + pos.height,
        #                     color='white', edgecolor='black', alpha=0.8, marker='D', s=30, zorder=3)

        n_by_period = df_long.groupby(['AOI', 'Scenario']).size().reset_index(name='n')
        new_xtick_labels = []
        for basin in basin_order:
            d = n_by_period[n_by_period['AOI'] == basin]
            nh = d[d['Scenario'] == 'NoSLR']['n'].item()
            np = d[d['Scenario'] == 'SLR']['n'].item()
            new_string = f'{basin}'#\n(nH={nh},\nnP={np})'
            new_xtick_labels.append(new_string)

        ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5], labels=new_xtick_labels, rotation=0, fontsize=9, color='black')
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgray', zorder=0)

        if i == 1:
            ax.legend(frameon=False, title=None)
        else:
            ax.get_legend().set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('Area (sq.km)')
        ax.set_title(scenario, fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.savefig(r'.\SLR_analysis\flood_area_violin_plots_SLRcomparison_hmin0.05.png', dpi=300)
    plt.close()

noslr_domain  = noslr_fld_area_df2[noslr_fld_area_df2['AOI'] == 'Domain']
noslr_median = noslr_domain['Compound'].describe()['50%']
noslr_domain['mode'] = 0
noslr_domain.loc[(noslr_domain['Compound'] > noslr_median), 'mode'] = 1

slr_domain  = slr_fld_area_df2[slr_fld_area_df2['AOI'] == 'Domain']
slr_median = slr_domain['Compound'].describe()['50%']
slr_domain['mode'] = 0
slr_domain.loc[(slr_domain['Compound'] > slr_median), 'mode'] = 1

# Storms
rp_file = fr'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_jointprob\basin_data\Domain_data_rp_{period}.csv'
rp_data = pd.read_csv(rp_file, index_col=0)
rp_sel = rp_data[rp_data.index.isin(noslr_domain['tc_id'].values)]
rp_nosel = rp_data[~rp_data.index.isin(noslr_domain['tc_id'].values)]

bounds = [1, 10, 25, 50, 100, 200, 500]
cmap = plt.cm.Blues
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
ckwargs = dict(norm=norm, cmap=cmap)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), tight_layout=True, sharex=True, sharey=False)
sc = ax.scatter(rp_nosel['maxWS_rp'], rp_nosel['MaxTotPrecipMM_rp'], c=rp_nosel['Compound_rp'], **ckwargs, marker='o',
                s=40, alpha=0.9, edgecolor='grey', linewidths=0.5)
sc = ax.scatter(rp_sel['maxWS_rp'], rp_sel['MaxTotPrecipMM_rp'], c=rp_sel['Compound_rp'], **ckwargs, marker='D',
                s=40, alpha=0.9, edgecolor='black', linewidths=0.5)
ax.set_yscale('log')
ax.set_ylim(1, 2200)
ax.set_xscale('log')
ax.set_xlim(1, 2200)
ax.grid('both')
ax.set_aspect('equal', adjustable='box')
ax.set_axisbelow(True)
ax.set_ylabel('Total Rainfall RP')
ax.set_xlabel('Max Wind Speed RP')
# Define the desired tick positions and labels
ticks = [1, 10, 100, 500, 1000]
labels = ['1', '10', '100', '500', '1000']
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
pos0 = ax.get_position()
cax1 = fig.add_axes([pos0.x1 + 0.05, pos0.y0 + pos0.height*0.1, 0.03, pos0.height * 0.9])
cbar1 = fig.colorbar(sc, cax=cax1, orientation='vertical',
                     label='Compound Flood Extent RP',
                     extend='max')
plt.margins(x=0, y=0)
plt.savefig(fr'.\SLR_analysis\slr_storms_cmpd.png', dpi=300, bbox_inches="tight")
plt.close()




# Plot
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
base_root = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\03_MODEL\sfincs_initcond_mod'
mod = SfincsModel(root=base_root, mode='r', data_libs=yml_base)
dem = mod.grid['dep']
# Plotting details
wkt = mod.grid['dep'].raster.crs.to_wkt()
utm_zone = mod.grid['dep'].raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True

# Load the TC Track file
fname = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585\tracks\UScoast6_AL_canesm_ssp585cal_roEst1rmEst1_trk100'
tc_tracks = sio.loadmat(f'{fname}.mat')
tc_tracks_polyline = track_points_to_linestring(tc_tracks, output_shpfile=None)
wl_da = xr.open_dataset(r'.\SLR_analysis\canesm_ssp585_SRL112cm\zsmax_canesm_slr.nc')
da = xr.open_dataset(r'.\SLR_analysis\canesm_ssp585_SRL112cm\hmin0.05\attribution_canesm_slr_0.05m.nc')

x = rp_sel[(rp_sel['MaxTotPrecipMM_rp'] > 0) & (rp_sel['maxWS_rp'] > 0) & (rp_sel['Compound_rp'] > 200)]
# PLOTTING MAPS
tc_id = 3214
tc_track_gdf = tc_tracks_polyline[tc_tracks_polyline['tc_id'] == tc_id]
tc_track_gdf = tc_track_gdf.to_crs(32617)

wl_diff = wl_da.sel(tc_id = f'{str(tc_id)}_SLR112cm', scenario='compound') - wl_da.sel(tc_id = str(tc_id), scenario='compound')
zs = wl_diff['zsmax']
zs = zs.where(zs > 0.05)

# Diff in Water Level for 100-year compound
da_noslr =  da.sel(tc_id = str(tc_id))['zsmax_diff']
da_slr = da.sel(tc_id = f'{str(tc_id)}_SLR112cm')['zsmax_diff']
da_diff = da_slr - da_noslr
dp = [da_noslr, da_slr, da_diff]
title = [f'TC {tc_id} No SLR', 'SLR 112cm', 'SLR minus No SLR']

# Mask out the cells that are considered water bodies
water_mask = xr.open_dataarray(r'waterbody_mask.nc')
water_mask = (water_mask == 0.0)

# Map it!
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 7), subplot_kw={'projection': utm},tight_layout=True, layout='constrained')
ckwargs = dict(cmap='seismic', vmin=-0.5, vmax=0.5)
for i in range(len(dp)):
    ax = axes[i]
    d = dp[i].where(water_mask)
    cs = d.plot(ax=ax, add_colorbar=False, **ckwargs, zorder=1)
    mod.region.plot(ax=ax, color='grey', edgecolor='none', zorder=0, alpha=0.6)
    mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=2, alpha=1)
    tc_track_gdf.geometry.plot(ax=ax, color='blue', linewidth=2, zorder=2)
    ax.set_axis_off()
    ax.set_title(title[i])

    minx, miny, maxx, maxy = mod.region.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

ax = axes[1]
label = 'Compound minus\nmax. indiv (m)'
pos0 = ax.get_position()  # get the original position
cax = fig.add_axes([pos0.x1 + 0.1, pos0.y0, 0.03, pos0.height * 0.9])
cbar2 = fig.colorbar(cs, cax=cax, orientation='vertical', label=label, extend='both'
                     #ticks=[0.05, 0.5, 1, 1.5],
                     )
plt.subplots_adjust(wspace=0.0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(rf'.\SLR_analysis\canesme_TC_{tc_id}_CompoundMinusMaxIndiv_waterMasked.jpg', bbox_inches='tight', dpi=300)
plt.close()


# PLOTTING COMPOUND FREQUENCY
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\SLR_analysis')

attr_ds = xr.open_dataset(r'.\canesm_ssp585_SRL112cm\hmin0.05\attribution_canesm_slr_0.05m.nc',
                          chunks={'x': 300, 'y': 300, 'tc_id': 25})

# Mask out the cells that are considered water bodies
water_mask = xr.open_dataarray(r'..\waterbody_mask.nc')
water_mask = (water_mask == 0.0)

# Mask out water
attr_ds = attr_ds.where(water_mask)

# Calculate the compound flood extent
hmin = 0.05
compound_mask = attr_ds['zsmax_diff'] > hmin
cmpd_extent_ds = xr.where(compound_mask, x=1, y=0)

# Split the SLR and NO SLR IDs
tc_ids = cmpd_extent_ds.tc_id.values.tolist()
tc_ids_slr = [x for x in tc_ids if 'SLR' in x]
tc_ids_noslr = [x for x in tc_ids if 'SLR' not in x]

# Select the scenario runs and sum the frequency of being compound across all
cmpd_slr = cmpd_extent_ds.sel(tc_id=tc_ids_slr).sum(dim='tc_id')
cmpd_noslr = cmpd_extent_ds.sel(tc_id=tc_ids_noslr).sum(dim='tc_id')

max_num = len(tc_ids_slr)

# setup a mask for when both SLR and no SLR do not have compound flooding
data_mask = (cmpd_slr != 0) & (cmpd_noslr != 0)

# calculate the difference in the frequency of compound flooding across the events
diff = cmpd_slr - cmpd_noslr
diff2 = diff.where(data_mask)
#diff = xr.where(mask, np.nan, diff)

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

# PLOTTTT
perc = diff2/189

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), subplot_kw={'projection': utm}, tight_layout=True)
cmap = plt.cm.get_cmap('coolwarm', 5)
c = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.ListedColormap(c)
bounds = [-0.5, -0.25, -0.01, 0.01, 0.25, 0.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

cs = perc.plot(ax=ax,
               cmap=cmap,
               norm=norm,
               extend='neither',
               shading='auto',
               add_colorbar=False,
               zorder=2, alpha=1)
ax.set_title('')
ax.set_aspect('equal')
ax.set_axis_off()
major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=0, alpha=1)
nc_major_rivers_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=0, alpha=1)
coastal_wb_clip.plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, zorder=0, alpha=0.75)
mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=3, alpha=1)

#ax.set_title(f'Frequency of Compound Flooding for Storms RP > 80-yr (hmin:{hmin}m)')
pos0 = ax.get_position()  # get the original position
cax = fig.add_axes([pos0.x1 + 0.075, pos0.y0 + pos0.height * 0.1, 0.035, pos0.y0 + pos0.height * 0.8])
cbar = fig.colorbar(cs,
                     cax=cax,
                     orientation='vertical',
                     ticks=[-0.5, -0.25, 0, 0.25, 0.5],
                     #label='% Compound Frequency'
                     )
cbar.ax.set_yticklabels(labels=['50% More Likely\n w/out SLR', '25%', 'Equal Likelihood', '25%', '50% More Likely\nw/ SLR'])

plt.subplots_adjust(wspace=0.0, hspace=0)
plt.margins(x=0, y=0)
plt.savefig(rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\SLR_analysis\final_{hmin}.jpg',
            bbox_inches='tight', dpi=300)
plt.close()

#
# dp = [cmpd_slr.where(cmpd_slr > 0), cmpd_noslr.where(cmpd_noslr > 0), diff2]
# cc = ['Reds', 'Reds', 'coolwarm']
# title = ['SLR', 'No SLR', 'SLR minus No SLR']
#
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5,6.5),
#                          subplot_kw={'projection': utm}, tight_layout=True)
# axes = axes.flatten()
# for i in range(len(axes)):
#     d = dp[i]
#     cmap=cc[i]
#     ax =axes[i]
#     if i < 2:
#         norm = mpl.colors.Normalize(vmin=0, vmax=max_num)
#     else:
#         norm = mpl.colors.Normalize(vmin=-max_num, vmax=max_num)
#     cs = d.plot(ax=ax,
#                    cmap=cmap,
#                    norm=norm,
#                    extend='neither',
#                    shading='auto',
#                    add_colorbar=False, zorder=2, alpha=1)
#
#     ax.set_title(title[i])
#     ax.set_aspect('equal')
#     ax.set_axis_off()
#     major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='black', linewidth=0.25, zorder=0, alpha=1)
#     nc_major_rivers_clip.plot(ax=ax, color='slategrey', edgecolor='black', linewidth=0.25, zorder=0, alpha=1)
#     coastal_wb_clip.plot(ax=ax, color='slategrey', edgecolor='black', linewidth=0.25, zorder=0, alpha=0.75)
#     mod.region.plot(ax=ax, color='none', edgecolor='black', linewidth=0.75, zorder=3, alpha=1)
#
# #ax.set_title(f'Frequency of Compound Flooding for Storms RP > 80-yr (hmin:{hmin}m)')
# ax = axes[1]
# pos0 = ax.get_position()  # get the original position
# cax = fig.add_axes([pos0.x1 + 0.075, pos0.y0 + pos0.height * 0.1, 0.05, pos0.y0])
# cbar = fig.colorbar(cs,
#                      cax=cax,
#                      orientation='vertical',
#                      ticks=[-max_num, 0, max_num],
#                      label='Compound Frequency'
#                      )
#
# plt.subplots_adjust(wspace=0.0, hspace=0)
# plt.margins(x=0, y=0)
# plt.savefig(rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\SLR_analysis\ncep_cmpd_frequency_slr_hmin{hmin}.jpg',
#             bbox_inches='tight', dpi=300)
# plt.close()
#
#
#
