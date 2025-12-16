import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import numpy as np
import matplotlib.patches as patches
from matplotlib.lines import Line2D

mpl.use('TkAgg')
plt.ion()
font = {'family': 'Arial', 'size': 10}
mpl.rc('font', **font)
mpl.rcParams.update({'axes.titlesize': 10})
mpl.rcParams["figure.autolayout"] = True
###################################################################################################################
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\01_return_period_tables')
outdir = r'..\07_correlation_matrix'


###################################################################################################################
ncep_csvfiles = [f for f in os.listdir() if f.endswith('ncep.csv')]
histdf = pd.concat((pd.read_csv(file, index_col=0) for file in ncep_csvfiles), ignore_index=False)

# Track info at landfall
track_table_filepath = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\NCEP_Reanalysis\tracks\ncep_landfall_track_info.csv'
track_df = pd.read_csv(track_table_filepath, index_col=0)
track_df.set_index('tc_id', inplace=True, drop=True)
track_df_hist = track_df[['rmw100','pstore100','speed100','vstore100']]

# Load the projected/future TC data
canesm_csvfiles = [f for f in os.listdir() if f.endswith('canesm.csv')]
futdf = pd.concat((pd.read_csv(file, index_col=0) for file in canesm_csvfiles), ignore_index=False)

# Track info at landfall
track_table_filepath = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\CMIP6_585\tracks\canesm_landfall_track_info.csv'
track_df = pd.read_csv(track_table_filepath, index_col=0)
track_df.set_index('tc_id', inplace=True, drop=True)
track_df_fut = track_df[['rmw100','pstore100','speed100','vstore100']]

###################################################################################################################
os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\07_correlation_matrix')
basins = histdf['basin'].unique()
var_cols = ['speed100','rmw100', 'vstore100', 'pstore100']
xlabels = ['Translation\nSpeed (m/s)', 'Radius of Max\nWind Speeds (km)',
          'Max Sustained\nWind Speeds (knots)', 'Min Central\nPressure (hPa)']
xlims = [(0,30), (0,300), (5,150), (900, 1010)]
rp_threshold = 80
climates = ['Historic', 'Future']
scenario = ['Total', 'Coastal', 'Compound', 'Runoff']
colors = ['black', 'mediumblue', '#DD7596', 'mediumseagreen']

############## Plotting shifts in TC climatology distribution by watershed ########################################
nrow = len(var_cols)
ncol = len(climates)
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)

for basin_name in basins:
    hist_data = histdf[histdf['basin'] == basin_name].drop(columns='basin', errors='ignore')
    hist_data = pd.concat([hist_data, track_df_hist], axis=1)
    fut_data = futdf[futdf['basin'] == basin_name].drop(columns='basin', errors='ignore')
    fut_data = pd.concat([fut_data, track_df_fut], axis=1)

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5.5, 6), sharey="row", sharex=False)
    legend_handles = []
    legend_labels = []

    for k in range(len(var_cols)):
        var_col = var_cols[k]
        x_vals = np.linspace(fut_data[var_col].min(), fut_data[var_col].max(), 1000)
        clim_data = [hist_data, fut_data]

        for i in range(len(clim_data)):
            data = clim_data[i]
            ax = axes[k, i]

            # Combine full hist/future data across all basins for this scenario
            if climates[i] == 'Historic':
                all_data = hist_data
            else:
                all_data = fut_data
                n_storms = len(all_data)
                num_repeats = (all_data['weight'] * n_storms).round().astype(int)
                expanded_data = np.repeat(all_data[var_col], num_repeats)
                expanded_data = pd.DataFrame(expanded_data)
                if len(expanded_data) > n_storms:
                    expanded_df = expanded_data.iloc[:n_storms]
                elif len(expanded_data) < n_storms:
                    # If it's smaller, repeat some rows to match the total number of points
                    expanded_data = expanded_data.sample(n=n_storms, replace=True, random_state=42)
                all_data=expanded_data

            x_vals_all = np.linspace(all_data[var_col].min(), all_data[var_col].max(), 1000)
            density_all = scipy.stats.gaussian_kde(all_data[var_col])
            full_set_handle = ax.fill_between(x_vals_all, density_all(x_vals_all),
                    color='gray',
                    alpha=0.3,
                    label='Full Set' if (k == 0 and i == 0) else None)
            # Add to legend handles/labels once
            if (k == 0 and i == 0):
                legend_handles.append(full_set_handle)
                legend_labels.append('Full Set')

            for s in range(len(scenario)):
                subset = data[data[f'{scenario[s]}_Area_sqkm_RP'] > rp_threshold]
                if subset.empty:
                    continue  # Skip if no data

                density = scipy.stats.gaussian_kde(subset[var_col])
                line, = ax.plot(x_vals, density(x_vals), colors[s],
                                linewidth=2.5, alpha=0.9,
                                label=f'{scenario[s]} (n={len(subset)})')

                # Add legend items only once
                if (k == 0 and i == 0):
                    legend_handles.append(line)
                    legend_labels.append(f'{scenario[s]}')

            ax.set_xlabel(xlabels[k])
            ax.set_xlim(xlims[k])
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)

    # Format titles and labels
    axes = axes.flatten()
    for ii in range(len(axes)):
        if ii in first_row:
            axes[ii].set_title(climates[ii], fontsize=10, fontweight='bold')
        if ii in first_in_row:
            axes[ii].set_ylabel('Density')

    # Adjust layout to leave space on the right for the legend
    plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.80)

    # Place the legend outside the plot area
    fig.legend(legend_handles, legend_labels,
               loc='center left', bbox_to_anchor=(0.99, 0.5),
               fontsize=10, frameon=False)
    plt.margins(x=0, y=0)
    plt.suptitle(basin_name)
    # Save the figure with 'tight' layout AFTER adjusting spacing
    plt.savefig(fr'density_hist_{basin_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


############## Plotting shifts in TC climatology distribution manuscript Fig ########################################

nrow = len(var_cols)
ncol = 4
n_subplots = nrow * ncol
first_in_row = np.arange(0, n_subplots, ncol)
last_in_row = np.arange(ncol - 1, n_subplots, ncol)
first_row = np.arange(0, ncol)
last_row = np.arange(first_in_row[-1], n_subplots, 1)

fig, axes = plt.subplots(nrows=nrow, ncols=4, figsize=(7, 6), sharey="row", sharex=False)
basins = ['CapeFear', 'Domain']
basin_cols = {'CapeFear': [0, 1], 'Domain': [2, 3]}
legend_handles = []
legend_labels = []
for k, var_col in enumerate(var_cols):
    for basin in basins:
        # Filter the data for this basin only
        hist_data = histdf[histdf['basin'] == basin].drop(columns='basin', errors='ignore')
        hist_data_basin = pd.concat([hist_data, track_df_hist], axis=1)
        fut_data = futdf[futdf['basin'] == basin].drop(columns='basin', errors='ignore')
        fut_data_basin = pd.concat([fut_data, track_df_fut], axis=1)

        clim_data = [hist_data_basin, fut_data_basin]
        cols = basin_cols[basin]
        x_vals = np.linspace(fut_data_basin[var_col].min(), fut_data_basin[var_col].max(), 1000)
        for i, col in enumerate(cols):
            data = clim_data[i]
            ax = axes[k, col]
            # Combine full hist/future data across all basins for this scenario
            if climates[i] == 'Historic':
                all_data = hist_data_basin
            else:
                all_data = fut_data_basin
                n_storms = len(all_data)
                num_repeats = (all_data['weight'] * n_storms).round().astype(int)
                expanded_data = np.repeat(all_data[var_col], num_repeats)
                expanded_data = pd.DataFrame(expanded_data)
                if len(expanded_data) > n_storms:
                    expanded_df = expanded_data.iloc[:n_storms]
                elif len(expanded_data) < n_storms:
                    # If it's smaller, repeat some rows to match the total number of points
                    expanded_data = expanded_data.sample(n=n_storms, replace=True, random_state=42)
                all_data=expanded_data

            x_vals_all = np.linspace(all_data[var_col].min(), all_data[var_col].max(), 1000)
            density_all = scipy.stats.gaussian_kde(all_data[var_col])
            full_set_handle = ax.fill_between(x_vals_all, density_all(x_vals_all),
                    color='gray',
                    alpha=0.3,
                    label='Full Set' if (k == 0 and col == 0) else None)
            # Add to legend handles/labels once
            if (k == 0 and col == 0):
                legend_handles.append(full_set_handle)
                legend_labels.append('Full Set')

            # Now plot the distributions by flood type
            for s in range(len(scenario)):
                subset = data[data[f'{scenario[s]}_Area_sqkm_RP'] > rp_threshold]
                if subset.empty:
                    continue

                density = scipy.stats.gaussian_kde(subset[var_col])
                line, = ax.plot(x_vals, density(x_vals), colors[s],
                                linewidth=2, alpha=0.8,
                                label=f'{scenario[s]}')

                if (k == 0 and col == 0):
                    legend_handles.append(line)
                    legend_labels.append(f'{scenario[s]}')

            ax.set_xlabel(xlabels[k], fontsize=10)
            ax.set_xlim(xlims[k])
            ax.set_axisbelow(True)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
# Titles for the columns to indicate basins
for basin, cols in basin_cols.items():
    for col in cols:
        axes[0, col].set_title(f'{basin} - {"Historic" if col % 2 == 0 else "Future"}', fontsize=10, fontweight='bold')
axes=axes.flatten()
for i in range(len(axes)):
    ax=axes[i]
    if i in first_in_row:
        ax.set_ylabel("Density", fontsize=10)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.tight_layout()
fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
    fontsize=10, frameon=True, ncol=len(legend_labels), borderaxespad=0, handletextpad=0.5, columnspacing=1.0)
plt.savefig(fr'density_hist_combined_CapeFear.png', dpi=300, bbox_inches='tight')
plt.close()

#########################################################
############## Calculating KS statistic to compare distributions ########################################
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu

stats_df = pd.DataFrame()
for basin in basins:
    # Filter the data for this basin only
    hist_data = histdf[histdf['basin'] == basin].drop(columns='basin', errors='ignore')
    hist_data_basin = pd.concat([hist_data, track_df_hist], axis=1)

    fut_data = futdf[futdf['basin'] == basin].drop(columns='basin', errors='ignore')
    fut_data_basin = pd.concat([fut_data, track_df_fut], axis=1)

    for s in scenario:
        hist_subset = hist_data_basin[hist_data_basin[f'{s}_Area_sqkm_RP'] > rp_threshold]
        fut_subset = fut_data_basin[fut_data_basin[f'{s}_Area_sqkm_RP'] > rp_threshold]

        for var in var_cols:
            hist_vals = hist_subset[var].dropna()
            fut_vals = fut_subset[var].dropna()

            ks_stat, ks_p = ks_2samp(hist_vals, fut_vals, mode='exact')
            u_stat, u_p = mannwhitneyu(hist_vals, fut_vals, alternative='two-sided')
            hist_median = np.median(hist_vals)
            fut_median = np.median(fut_vals)
            hist_n = len(hist_vals)
            fut_n = len(fut_vals)

            dfrow = pd.DataFrame([basin, s, var, ks_stat, ks_p, u_stat, u_p, hist_median, fut_median, hist_n, fut_n]).T

            stats_df = pd.concat(objs=[stats_df, dfrow], ignore_index=True, axis=0)

            print(f"{basin}, {s}, {var}: KS stat = {ks_stat:.3f}, p = {ks_p:.3f}")
            print(f"Mann-Whitney U stat = {u_stat}, p = {u_p:.3f}")
            print(f"Median (Hist) = {hist_median}, Median (Fut) = {fut_median}")

stats_df.columns = ['Basin','Flood_Type','Variable', 'ks_stat', 'ks_pvalue', 'mwu_stat', 'mwu_pvalue', 'hist_median',
                    'fut_median', 'hist_n', 'fut_n']

stats_df.round(4).to_csv('track_dist_test.csv')



