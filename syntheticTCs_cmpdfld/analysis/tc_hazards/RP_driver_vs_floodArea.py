import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

''' HISTORIC PERIOD DATA '''
# Load meteo information
ws_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\rain\basin_tc_precipitation_stats.csv', index_col=0)
meteo_df = pd.concat(objs=[ws_df, rain_df], ignore_index=False, axis=1)
df_list = [meteo_df[meteo_df.index == i].T  for i in meteo_df.index]
meteo_df = pd.concat(objs=df_list,axis=1)
basins = [x.split('_')[0] for x in meteo_df.index]
vars = [''.join(x.split('_')[1:]) for x in meteo_df.index]
tuples = [(basins[i], vars[i]) for i in range(len(vars))]
meteo_df.index = pd.MultiIndex.from_tuples(tuples, names=['Basin', 'Variable'])

# Correlate storm tide location with basin
stormtide_pt = ['188','194','198','204', '206']
stormtide_basin = ['LowerPeeDee', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico']

# Load in stormtide peaks
stormtide_data = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\stormTide\gage_peaks_ZerosRemoved_ncep.csv', index_col=0)
# Load in flood extent for each storm by process
dd = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep'
fld_extent_df = pd.read_csv(os.path.join(dd, f'overland_flooded_area_table.csv'), index_col=0)
fld_extent_df['AOI'] = [s.replace(" ", "") for s in fld_extent_df['AOI']]
fld_extent_df = fld_extent_df[['Coastal', 'Compound', 'Runoff', 'Total_Flooded', 'AOI']]

lambda_param = 3.38 * (1309 / 5018)
data_dict = {}
for i in range(len(stormtide_basin)):
    st_loc = stormtide_pt[i]
    basin = stormtide_basin[i]

    # Query stormtide data for location
    basin_st = stormtide_data[st_loc].dropna()
    basin_st.index = basin_st.index.astype(int)
    basin_data = meteo_df.xs(basin, level='Basin').T
    basin_data.index = basin_data.index.astype(int)

    # Query flood extent data for basin
    fld_extent_data = fld_extent_df[fld_extent_df['AOI'].str.contains(basin)]['Compound']

    # Combine the meteo, stormtide, and flood area data into a single dataframe
    driver = pd.concat(objs=[basin_data, basin_st, fld_extent_data], axis=1, ignore_index=False).fillna(0)

    # Sort the dataframe by flood area and calculate return period
    sorted = driver.sort_values(by='Compound', axis=0, ascending=True)
    sorted['ecdf_area'] = np.arange(1, len(sorted)+1) / (len(sorted))
    sorted['excedprob_area'] = 1 - sorted['ecdf_area']
    sorted['rp_area'] = (1 / (1 - np.exp(-lambda_param*sorted['excedprob_area'])))

    sorted = sorted.sort_values(by=st_loc, axis=0, ascending=True)
    sorted['ecdf_st'] = np.arange(1, len(sorted)+1) / (len(sorted))
    sorted['excedprob_st'] = 1 - sorted['ecdf_st']
    sorted['rp_st'] = (1 / (1 - np.exp(-lambda_param*sorted['excedprob_st'])))

    sorted = sorted.sort_values(by='meanWSthresh', axis=0, ascending=True)
    sorted['ecdf_ws'] = np.arange(1, len(sorted)+1) / (len(sorted))
    sorted['excedprob_ws'] = 1 - sorted['ecdf_ws']
    sorted['rp_ws'] = (1 / (1 - np.exp(-lambda_param*sorted['excedprob_ws'])))

    sorted = sorted.sort_values(by='MeanTotPrecipMM', axis=0, ascending=True)
    sorted['ecdf_mtp'] = np.arange(1, len(sorted)+1) / (len(sorted))
    sorted['excedprob_mtp'] = 1 - sorted['ecdf_mtp']
    sorted['rp_mtp'] = (1 / (1 - np.exp(-lambda_param*sorted['excedprob_mtp'])))

    # save dataframe to list for plotting below
    data_dict[basin] = sorted

''' PROJECTED PERIOD DATA '''
# Load meteo information
ws_df2 = pd.read_csv(r'.\02_DATA\CMIP6_585\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df2 = pd.read_csv(r'.\02_DATA\CMIP6_585\rain\basin_tc_precipitation_stats.csv', index_col=0)
meteo_df2 = pd.concat(objs=[ws_df2, rain_df2], ignore_index=False, axis=1)
df_list2 = [meteo_df2[meteo_df2.index == i].T  for i in meteo_df2.index]
meteo_df2 = pd.concat(objs=df_list2,axis=1)

vars = [''.join(x.split('_')[1:]) for x in meteo_df2.index]
tuples = [(basins[i], vars[i]) for i in range(len(vars))]
meteo_df2.index = pd.MultiIndex.from_tuples(tuples, names=['Basin', 'Variable'])

# Load in stormtide peaks
stormtide_data2 = pd.read_csv(r'.\02_DATA\CMIP6_585\stormTide\gage_peaks_ZerosRemoved_canesm_ssp585.csv', index_col=0)

# Load in flood extent for each storm by process
dd = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\canesm_ssp585'
fld_extent_df2 = pd.read_csv(os.path.join(dd, f'overland_flooded_area_table.csv'), index_col=0)
fld_extent_df2['AOI'] = [s.replace(" ", "") for s in fld_extent_df2['AOI']]
fld_extent_df2 = fld_extent_df2[['Coastal', 'Compound', 'Runoff', 'Total_Flooded', 'AOI']]

dd = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\BiasCorrection'
storm_weights = pd.read_csv(os.path.join(dd, r'canesm_ssp585_weighted.csv'), index_col=0, header=None)
storm_weights.columns = ['vmax','weight']
lambda_param = 3.38 * (1543 / 6200)
data_dict_fut = {}
for i in range(len(stormtide_basin)):
    st_loc = stormtide_pt[i]
    basin = stormtide_basin[i]

    # Query stormtide data for location
    basin_st = stormtide_data2[st_loc].dropna()
    basin_st.index = basin_st.index.astype(int)
    basin_data = meteo_df2.xs(basin, level='Basin').T
    basin_data.index = basin_data.index.astype(int)

    # Query flood extent data for basin
    fld_extent_data = fld_extent_df2[fld_extent_df2['AOI'].str.contains(basin)]['Compound']

    # Combine the meteo, stormtide, and flood area data into a single dataframe
    driver = pd.concat(objs=[basin_data, basin_st, fld_extent_data, storm_weights], axis=1, ignore_index=False).fillna(0)

    # Sort the dataframe by flood area and calculate return period
    sorted = driver.sort_values(by='Compound', axis=0, ascending=True)
    sorted['ecdf_area'] = sorted['weight'].cumsum()
    sorted['excedprob_area'] = 1 - sorted['ecdf_area']
    sorted['rp_area'] = (1 / (1 - np.exp(-lambda_param*sorted['excedprob_area'])))

    sorted = sorted.sort_values(by=st_loc, axis=0, ascending=True)
    sorted['ecdf_st'] = sorted['weight'].cumsum()
    sorted['excedprob_st'] = 1 - sorted['ecdf_st']
    sorted['rp_st'] = (1 / (1 - np.exp(-lambda_param*sorted['excedprob_st'])))

    sorted = sorted.sort_values(by='meanWSthresh', axis=0, ascending=True)
    sorted['ecdf_ws'] = sorted['weight'].cumsum()
    sorted['excedprob_ws'] = 1 - sorted['ecdf_ws']
    sorted['rp_ws'] = (1 / (1 - np.exp(-lambda_param*sorted['excedprob_ws'])))
    sorted.loc[(sorted['rp_ws'] > 10000), 'rp_ws'] = 1000

    sorted = sorted.sort_values(by='MeanTotPrecipMM', axis=0, ascending=True)
    sorted['ecdf_mtp'] = sorted['weight'].cumsum()
    sorted['excedprob_mtp'] = 1 - sorted['ecdf_mtp']
    sorted['rp_mtp'] = (1 / (1 - np.exp(-lambda_param*sorted['excedprob_mtp'])))
    sorted.loc[(sorted['rp_mtp'] > 10000), 'rp_mtp'] = 1000

    # save dataframe to list for plotting below
    data_dict_fut[basin] = sorted


''' PLOTTING '''
# Plotting prep
bounds = [1, 10, 50, 100, 250, 500]
cmap = plt.cm.Blues
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
ckwargs = dict(norm=norm, cmap=cmap)
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
plt.rcParams['figure.constrained_layout.use'] = True

outdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\figures'
variable_x = 'rp_mtp'
variable_x_label = 'RP Avg Total Precip (mm)'
variable_y = 'rp_ws'
variable_y_label = 'RP Mean Wind\nSpeed >5 m/s'
nrow = 5
ncol = 2
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)

fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5, 6.5), sharex=False, sharey=False)
for i in range(len(stormtide_basin)):
    st_loc = stormtide_pt[i]
    basin = stormtide_basin[i]
    data = data_dict[basin]
    data_fut = data_dict_fut[basin]

    if variable_y == 'surge':
        ax = axs[i][0]
        sc = ax.scatter(data[variable_x], data[st_loc], c=data['rp_area'], **ckwargs,
                        s=40, alpha=0.9, edgecolor='grey', linewidths=0.25)
        ax = axs[i][1]
        sc = ax.scatter(data_fut[variable_x], data_fut[st_loc], c=data_fut['rp_area'], **ckwargs,
                        s=40, alpha=0.9, edgecolor='grey', linewidths=0.25)
    else:
        ax = axs[i][0]
        sc = ax.scatter(data[variable_x], data[variable_y], c=data['rp_area'], **ckwargs,
                        s=40, alpha=0.9, edgecolor='grey',linewidths=0.25)
        ax = axs[i][1]
        sc = ax.scatter(data_fut[variable_x], data_fut[variable_y], c=data_fut['rp_area'], **ckwargs,
                        s=40, alpha=0.9, edgecolor='grey',linewidths=0.25)

    axs[i][0].text(-0.4, 0.5, basin, horizontalalignment='right', verticalalignment='center',
                   rotation='vertical', transform=axs[i][0].transAxes)
axs = axs.flatten()
for i in range(len(axs)):
    ax = axs[i]
    ax.set_yscale('log')
    ax.set_xscale('log')
    if i in last_row:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel(variable_x_label)
    if i in first_in_row:
        ax.set_ylabel(variable_y_label)
axs[0].set_title('Historic (1980-2005)')
axs[1].set_title('Projected (2070-2100)')
pos0 = axs[5].get_position()
cax1 = fig.add_axes([pos0.x1 + 0.1, pos0.y0, 0.03, pos0.height * 1.5])
cbar1 = fig.colorbar(sc, cax=cax1, orientation='vertical',
                     label='Compound Flood Extent\nReturn Period',
                     extend='max')
plt.margins(x=0, y=0)
plt.savefig(os.path.join(outdir, fr'{variable_y}_vs_{variable_x}_RP_compoundExtent.png'),
            dpi=300,
            bbox_inches="tight"
            )
plt.close()


# data = driver.values
# kde = KernelDensity(kernel='gaussian', bandwidth=1)  # Adjust bandwidth for smoother/rougher density
# kde.fit(data)
# # Create a grid of points to evaluate the density over
# x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
# y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
#
# # Evaluate the KDE over the grid
# grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
# log_density = kde.score_samples(grid_points)
# density = np.exp(log_density).reshape(xx.shape)
#
# # Normalize the density to get the cumulative distribution
# cumulative_density = np.cumsum(density.ravel()) / np.sum(density)
#
# # Reshape the cumulative density back to the grid
# cumulative_density_grid = cumulative_density.reshape(xx.shape)
#
# # Calculate exceedance probability (EP)
# exceedance_probability = 1 - cumulative_density_grid
#
# # Calculate the return period (1 / EP)
# lambda_param = 3.38*(len(data)/5018)
# return_period = (1 / (1 - np.exp(-lambda_param*exceedance_probability)))
#
# contour_levels = [100, 200, 500]  # Example return periods to plot
# # Plot the result
# plt.figure(figsize=(6, 5))
# contours = plt.contourf(xx, yy, return_period, levels=contour_levels, cmap='viridis')
# plt.scatter(data[:, 0], data[:, 1], c='black', s=5, alpha=0.1)  # Plot data points
# plt.xlabel('Mean Rain Rate (> 5 m/hr)')
# plt.ylabel('Peak Surge (m)')
# plt.title('Bivariate Kernel Density Estimate (KDE)')
# plt.show()
