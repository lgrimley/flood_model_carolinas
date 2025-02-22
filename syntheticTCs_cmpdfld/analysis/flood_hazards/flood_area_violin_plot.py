import os
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
mpl.use('TkAgg')  # Use the TkAgg backend (commonly works well)
plt.ion()


# Load the historical TC data
histdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\ncep'
histdf = pd.read_csv(os.path.join(histdir, 'overland_flooded_area_table.csv'), index_col=0)
histdf['AOI'] = [s.replace(" ","") for s in histdf['AOI']]
histdf['tc_id'] = histdf.index
histdf['Period'] = 'Historic'

# Load the projected/future TC data
futdir = r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\canesm_ssp585'
futdf = pd.read_csv(os.path.join(futdir, 'overland_flooded_area_table.csv'), index_col=0)
futdf['AOI'] = [s.replace(" ","") for s in futdf['AOI']]
futdf['tc_id'] = futdf.index
futdf['Period'] = 'Projected'

storm_weights = pd.read_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\02_DATA\BiasCorrection\canesm_ssp585_weighted.csv', header=None)
storm_weights.columns = ['tc_id','vmax','weight']
futdf = pd.merge(futdf, storm_weights, on='tc_id', how='left')

def expand_df_by_weights(df, scenarios):
    for group, group_data in df.groupby('AOI'):
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

            df.loc[(df['AOI'] == group), scenario] = expanded_data.values

    return df

scenarios = ['Runoff', 'Coastal', 'Compound', 'Total_Flooded']
df1 = expand_df_by_weights(df=futdf,scenarios=scenarios)


violin_plot_biascorrected = False
if violin_plot_biascorrected is True:
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True
    basin_order = ['LPD', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico', 'Domain']
    scenarios = ['Runoff','Coastal', 'Compound', 'Total_Flooded']
    fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(5.5, 7), sharey=False, sharex=False)
    axs = axs.flatten()
    for i in range(len(scenarios)):
        ax =axs[i]
        scenario = scenarios[i]

        df_combined = pd.concat([histdf[['tc_id', 'AOI', scenario, 'Period']],
                                 df1[['tc_id', 'AOI', scenario, 'Period']]])

        df_combined['AOI'] = df_combined['AOI'].replace('LowerPeeDee', 'LPD')
        df_long = pd.melt(df_combined, id_vars=['tc_id', 'AOI', 'Period'], value_vars=[scenario],
                          var_name='data_type', value_name='data_value')
        df_long.dropna(axis=0, inplace=True)

        # Violin plot
        sns.violinplot(x='AOI', y='data_value', hue='Period', data=df_long,
                       ax=ax,
                       order=basin_order,
                       split=True, fill=True, gap=0.1, width=1,
                       log_scale=True, density_norm='count', linecolor='none', zorder=1,
                       palette={'Historic': 'silver', 'Projected': 'grey'},
                       inner_kws=dict(box_width=5, whis_width=1.5, color="black"))

        pos = ax.get_position()  # Get the axis position
        n_by_period = df_long.groupby(['AOI', 'Period']).size().reset_index(name='n')
        new_xtick_labels = []
        for basin in basin_order:
            d = n_by_period[n_by_period['AOI'] == basin]
            nh = d[d['Period'] == 'Historic']['n'].item()
            np = d[d['Period'] == 'Projected']['n'].item()
            new_string = f'{basin}\n(nH={nh},\nnP={np})'
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
    plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_attribution\flood_area_violin_plots_biasCorr.png',
                dpi=300)


# Create a table of stats
hist_group = histdf.groupby('AOI')[['Compound','Coastal', 'Runoff','Total_Flooded']].describe(percentiles=[0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95])
hist_group = hist_group.T

proj_group = df1.groupby('AOI')[['Compound','Coastal', 'Runoff','Total_Flooded']].describe(percentiles=[0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95])
proj_group = proj_group.T
proj_group.columns = [f'{s}_Fut' for s in proj_group.columns]

# Create a table of stats
a = histdf.groupby('AOI')[['Compound','Coastal', 'Runoff','Total_Flooded']].mean()
a.columns = [f'{s}_mean' for s in a.columns]
b = histdf.groupby('AOI')[['Compound','Coastal', 'Runoff','Total_Flooded']].std()
b.columns = [f'{s}_std' for s in b.columns]
c = histdf.groupby('AOI')[['Compound','Coastal', 'Runoff','Total_Flooded']].mean()
c.columns = [f'{s}_mean_Fut' for s in c.columns]
d = histdf.groupby('AOI')[['Compound','Coastal', 'Runoff','Total_Flooded']].std()
d.columns = [f'{s}_std_Fut' for s in d.columns]
abcd = pd.concat(objs = [a,b,c,d], axis=1, ignore_index=False).round(2)
abcd.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\flood_area_standardDev2.csv')


combined_stats = pd.concat(objs=[hist_group,proj_group], axis=1, ignore_index=False)
for c in hist_group.columns:
     combined_stats[f'{c}_Diff'] = combined_stats[f'{c}_Fut'] - combined_stats[c]
combined_stats.to_csv(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\flood_extent_stats2.csv')


# violin_plot = True
# if violin_plot is True:
#     font = {'family': 'Arial', 'size': 10}
#     mpl.rc('font', **font)
#     mpl.rcParams.update({'axes.titlesize': 10})
#     mpl.rcParams["figure.autolayout"] = True
#     basin_order = ['LPD', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico', 'Domain']
#     scenarios = ['Runoff','Coastal', 'Compound', 'Total_Flooded']
#     fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(5.5, 7), sharey=False, sharex=False)
#     axs = axs.flatten()
#     for i in range(len(scenarios)):
#         ax =axs[i]
#         scenario = scenarios[i]
#         df_combined = pd.concat([histdf[['tc_id', 'AOI', scenario, 'Period']],
#                                  futdf[['tc_id', 'AOI', scenario, 'Period']]])
#
#         df_combined['AOI'] = df_combined['AOI'].replace('LowerPeeDee', 'LPD')
#         df_long = pd.melt(df_combined, id_vars=['tc_id', 'AOI', 'Period'], value_vars=[scenario],
#                           var_name='data_type', value_name='data_value')
#         df_long.dropna(axis=0, inplace=True)
#
#         # Violin plot
#         sns.violinplot(x='AOI', y='data_value', hue='Period', data=df_long,
#                        ax=ax,
#                        order=basin_order,
#                        split=True, fill=True, gap=0.1,
#                        log_scale=True, density_norm='count', linecolor='none', zorder=1,
#                        palette={'Historic': 'silver', 'Projected': 'grey'},
#                        inner_kws=dict(box_width=5, whis_width=1.5, color="black"))
#
#         pos = ax.get_position()  # Get the axis position
#
#         # Calculate the mean values for each group and plot them as diamonds
#         means_by_period = df_long.groupby(['AOI', 'Period'])['data_value'].mean().reset_index()
#         basin_positions = {'Domain': 5, 'Pamlico': 4, 'Neuse': 3, 'OnslowBay': 2, 'CapeFear': 1, 'LPD': 0}
#         # Plot the mean values as diamond markers for each period (source)
#         for _, row in means_by_period.iterrows():
#             basin = row['AOI']
#             mean_value = row['data_value']
#             source = row['Period']
#             if source == 'Historic':
#                 ax.scatter(basin_positions[basin]-0.06, mean_value + pos.height,
#                             color='white', edgecolor='black', alpha=0.8, marker='D', s=30, zorder=3)
#             else:
#                 ax.scatter(basin_positions[basin]+0.06, mean_value + pos.height,
#                             color='white', edgecolor='black', alpha=0.8, marker='D', s=30, zorder=3)
#
#         n_by_period = df_long.groupby(['AOI', 'Period']).size().reset_index(name='n')
#         new_xtick_labels = []
#         for basin in basin_order:
#             d = n_by_period[n_by_period['AOI'] == basin]
#             nh = d[d['Period'] == 'Historic']['n'].item()
#             np = d[d['Period'] == 'Projected']['n'].item()
#             new_string = f'{basin}\n(nH={nh},\nnP={np})'
#             new_xtick_labels.append(new_string)
#
#         ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5], labels=new_xtick_labels, rotation=0, fontsize=9, color='black')
#         ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgray', zorder=0)
#
#         if i == 1:
#             ax.legend(frameon=False, title=None)
#         else:
#             ax.get_legend().set_visible(False)
#         ax.set_xlabel('')
#         ax.set_ylabel('Area (sq.km)')
#         ax.set_title(scenario, fontsize=9)
#
#     plt.tight_layout()
#     plt.show()
    # plt.savefig(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\figures\flood_area_violin_plots.png',
    #             dpi=300)



