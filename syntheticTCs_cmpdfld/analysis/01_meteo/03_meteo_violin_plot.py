import os
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import numpy as np
import hydromt
from scipy.stats import gaussian_kde
mpl.use('TkAgg')  # Use the TkAgg backend (commonly works well)
plt.ion()

# Read in the data catalog to get the model and basin geom
data_catalog_yml = r'Z:\Data-Expansion\users\lelise\data\data_catalog_SFINCS_Carolinas.yml'
yml_base = r'Z:\Data-Expansion\users\lelise\data\data_catalog_BASE_Carolinas.yml'
cat = hydromt.DataCatalog(data_libs=[data_catalog_yml, yml_base])
# Read in the data catalog to get the model and basin geom
basins = cat.get_geodataframe(data_like=r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter2_PGW\sfincs\03_OBS\analysis_final\downscale_test\masks\huc6_basins.shp')
basins = basins.to_crs(epsg=32617)


os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\05_ANALYSIS\return_period_tables')
outdir = r'..\storm_set_stats'


def expand_df_by_weights(df, scenarios, groupby_col=None):
    for group, group_data in df.groupby(groupby_col):
        n_storms = 1543
        num_repeats = (group_data['weight'] * n_storms).round().astype(int)
        for scenario in scenarios:
            expanded_data = np.repeat(group_data[scenario], num_repeats)
            expanded_data = pd.DataFrame(expanded_data)

            if len(expanded_data) > n_storms:
                expanded_df = expanded_data.iloc[:n_storms]
            elif len(expanded_data) < n_storms:
                # If it's smaller, repeat some rows to match the total number of points
                expanded_data = expanded_data.sample(n=n_storms, replace=True, random_state=42)

            df.loc[(df[groupby_col] == group), scenario] = expanded_data.values

    return df


# Load the historical TC data
all_vars = ['maxWS', 'meanMaxWS', 'meanWS', 'meanWSthresh', 'meanDirection',
       'CumPrecipKM3', 'MeanTotPrecipMM', 'MaxTotPrecipMM', 'maxRR', 'meanRR',
       'meanRRthresh', 'stormtide']


ncep_csvfiles = [f for f in os.listdir() if f.endswith('ncep.csv')]
histdf = pd.concat((pd.read_csv(file, index_col=0) for file in ncep_csvfiles), ignore_index=False)
histdf['tc_id'] = histdf.index
histdf['Period'] = 'Historic'
select_columns = ['basin','tc_id','Period'] + all_vars
histdf = histdf[select_columns]

# Load the projected/future TC data
canesm_csvfiles = [f for f in os.listdir() if f.endswith('canesm.csv')]
futdf = pd.concat((pd.read_csv(file, index_col=0) for file in canesm_csvfiles), ignore_index=False)
futdf['tc_id'] = futdf.index
futdf['Period'] = 'Projected'
select_columns = ['basin','tc_id','Period', 'weight'] + all_vars
futdf = futdf[select_columns]
futdf = expand_df_by_weights(df=futdf, scenarios=all_vars, groupby_col='basin')


# Create a table of stats
remove = ['tc_id','Period', 'weight']
hist_group = histdf.groupby('basin')[histdf.columns[~histdf.columns.isin(remove)]].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
hist_group =hist_group.T

proj_group = futdf.groupby('basin')[futdf.columns[~futdf.columns.isin(remove)]].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
proj_group = proj_group.T
proj_group.index.name = 'basin'
proj_group.columns = [f'{s}_Fut' for s in proj_group.columns]

# Combine the hist and proj stats
combined_stats = pd.concat(objs=[hist_group,proj_group], axis=1, ignore_index=False)
combined_stats.index.names = ['variable', 'stat']

matching_cols = combined_stats.columns[combined_stats.columns.str.contains('Domain', case=False, regex=False)]
subset = combined_stats[matching_cols]
subset[f'Diff'] = subset[f'Domain_Fut'] - subset['Domain']
subset[f'FractionalChange'] = ((subset[f'Diff']) / subset['Domain'])
#subset.round(3).to_csv(rf'{outdir}\meteo_stormtide_histogramStats_Domain.csv')

watersheds = ['CapeFear', 'LowerPeeDee', 'Neuse', 'OnslowBay', 'Pamlico']
matching_cols = combined_stats.columns[~combined_stats.columns.str.contains('Domain', case=False, regex=False)]
subset = combined_stats[matching_cols]
for w in watersheds:
    subset[f'{w}_Diff'] = subset[f'{w}_Fut'] - subset[w]
    subset[f'{w}_FractionalChange'] = ((subset[f'{w}_Diff']) / subset[w])

#subset.round(3).to_csv(rf'{outdir}\meteo_stormtide_histogramStats_watershed.csv')



titles = ['Maximum Rain Rate', 'Cumulative Precipitation', 'Maximum Wind Speed', 'Peak Storm Tide']
scenarios = ['maxRR','CumPrecipKM3','maxWS','stormtide']

plot_biascorrected = True
if plot_biascorrected is True:
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True
    ylabels = ['mm/hr', 'cubic km','m/s', 'm']

    #fig, axs = plt.subplots(ncols=2, nrows=len(scenarios), figsize=(5, 6), sharey=False, sharex=False)
    #axs = axs.flatten()

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(8, 6))  # Set figure size
    gs = fig.add_gridspec(len(scenarios), 3, width_ratios=[3, 1, 0], height_ratios=[1, 1, 1, 1])  # First column takes 3/4 of the space
    # Iterate through the scenarios and plot
    for i in range(len(scenarios)):
        scenario = scenarios[i]
        df_combined = pd.concat([histdf[['tc_id', 'basin', scenario, 'Period']],
                                 futdf[['tc_id', 'basin', scenario, 'Period']]])
        df_combined['basin'] = df_combined['basin'].replace('LowerPeeDee', 'LPD')
        df_long = pd.melt(df_combined, id_vars=['tc_id', 'basin', 'Period'], value_vars=[scenario],
                          var_name='data_type', value_name='data_value')
        df_long.dropna(axis=0, inplace=True)
        df_long = df_long.replace([np.inf, -np.inf], np.nan).dropna(subset=['data_value'])
        df_long_domain = df_long[df_long['basin'] == 'Domain']
        df_long_basin = df_long[df_long['basin'] != 'Domain']

        # Create axes using GridSpec
        yaxis_lim = [(-10, 150), (-100, 5000), (0, 75), (0, 3)]
        basin_order = ['LPD', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico']
        ax1 = fig.add_subplot(gs[i, 0])  # First column (2/3 of the space)
        sns.violinplot(x='basin', y='data_value', hue='Period', data=df_long_basin,
                       ax=ax1,
                       order=basin_order,
                       split=True,
                       fill=True,
                       gap=0.05,
                       log_scale=False,
                       width=1,
                       density_norm='width',
                       dodge='auto',
                       bw_method=0.2,
                       common_norm=True,
                       linecolor='none',
                       zorder=1,
                       palette={'Historic': 'silver', 'Projected': 'grey'},
                       inner=None,
                       cut=0
                       )

        # Loop to calculate and plot 90th percentiles
        for ii, basin in enumerate(basin_order):
            for jj, period in enumerate(['Historic', 'Projected']):
                subset = df_long_basin[(df_long_basin['basin'] == basin) & (df_long_basin['Period'] == period)]
                if not subset.empty:
                    p5 = np.percentile(subset['data_value'], 5)
                    p95 = np.percentile(subset['data_value'], 95)
                    median = np.percentile(subset['data_value'], 50)

                    # X-position adjustment for split violin
                    offset = -0.2 if period == 'Historic' else 0.2
                    x = ii + offset

                    # Draw whisker line from 5th to 95th percentile
                    ax1.plot([x, x], [p5, p95], color='black', lw=2, zorder=2)

                    # Draw a horizontal line for the median
                    ax1.plot([x - 0.05, x + 0.05], [median, median], color='black', lw=2, zorder=2)

        ax1.set_ylim(yaxis_lim[i])
        pos = ax1.get_position()  # Get the axis position
        n_by_period = df_long_basin.groupby(['basin', 'Period']).size().reset_index(name='n')
        new_xtick_labels = []
        for basin in basin_order:
            d = n_by_period[n_by_period['basin'] == basin]
            nh = d[d['Period'] == 'Historic']['n'].item()
            npx = d[d['Period'] == 'Projected']['n'].item()
            new_string = f'{basin}'#\n(nH={nh},\nnP={np})'
            new_xtick_labels.append(new_string)

        ax1.set_xticks(ticks=[0, 1, 2, 3, 4], labels=new_xtick_labels, rotation=0, fontsize=10, color='black')
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.75, color='lightgray', zorder=0)
        ax1.get_legend().set_visible(False)
        ax1.set_xlabel('')
        ax1.set_ylabel(ylabels[i])
        ax1.set_title(titles[i])

        # Plot for the second subplot in the second column (1/3 space)
        ax2 = fig.add_subplot(gs[i, 1])  # Second column (1/3 of the space)
        basin_order = ['Domain']
        yaxis_lim = [(-10, 200), (-100, 80000), (0, 100), (0, 4)]
        sns.violinplot(x='basin', y='data_value', hue='Period', data=df_long_domain,
                       ax=ax2,
                       split=True,
                       fill=True,
                       gap=0.05,
                       log_scale=False,
                       width=1,
                       density_norm='width',
                       dodge='auto',
                       bw_method=0.2,
                       common_norm=True,
                       linecolor='none',
                       zorder=1,
                       palette={'Historic': 'silver', 'Projected': 'grey'},
                       inner=None,
                       cut=0
                       )

        # Loop to calculate and plot 90th percentiles
        for ii, basin in enumerate(basin_order):
            for jj, period in enumerate(['Historic', 'Projected']):
                subset = df_long_domain[(df_long_domain['basin'] == basin) & (df_long_domain['Period'] == period)]
                if not subset.empty:
                    p5 = np.percentile(subset['data_value'], 5)
                    p95 = np.percentile(subset['data_value'], 95)
                    median = np.percentile(subset['data_value'], 50)

                    # X-position adjustment for split violin
                    offset = -0.2 if period == 'Historic' else 0.2
                    x = ii + offset

                    # Draw whisker line from 5th to 95th percentile
                    ax2.plot([x, x], [p5, p95], color='black', lw=2, zorder=2)

                    # Draw a horizontal line for the median
                    ax2.plot([x - 0.05, x + 0.05], [median, median], color='black', lw=2, zorder=2)

        ax2.set_ylim(yaxis_lim[i])
        ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.75, color='lightgray', zorder=0)
        ax2.get_legend().set_visible(False)
        ax2.set_xlabel('')
        ax2.set_ylabel(ylabels[i])
        ax2.set_title('')

        # Set the same y-ticks and y-grid for ax2 to match ax1
        # ax2.set_yticks(ax1.get_yticks())
        # ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.75, color='lightgray', zorder=0)

        # Set the shared title for the row
        #fig.text(0.5, 1 - (i * 0.265), titles[i], ha='center', va='bottom', fontsize=12)  # Adjust vertical position based on row index
    #plt.tight_layout(h_pad=0.5)  # Increase the gap between rows (use `h_pad`)
    plt.tight_layout()
    plt.show()
    plt.savefig(rf'{outdir}\flood_drivers_violin_biasCorr.png',dpi=300)
