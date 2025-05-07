import os
import xarray as xr
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydromt
import os
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns
sys.path.append(r'/')
mpl.use('TkAgg')
plt.ion()

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

# Present period
ws_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df = pd.read_csv(r'.\02_DATA\NCEP_Reanalysis\rain\basin_tc_precipitation_stats.csv', index_col=0)
meteo_df = pd.concat(objs=[ws_df, rain_df], ignore_index=False, axis=1)
df_list = [meteo_df[meteo_df.index == i].T  for i in meteo_df.index]
meteo_df_hist = pd.concat(objs=df_list,axis=1)
basins = [x.split('_')[0] for x in meteo_df_hist.index]
vars = [''.join(x.split('_')[1:]) for x in meteo_df_hist.index]
tuples = [(basins[i], vars[i]) for i in range(len(vars))]
meteo_df_hist.index = pd.MultiIndex.from_tuples(tuples, names=['Basin', 'Variable'])

master_df_ls = []
for basin_name in meteo_df_hist.index.levels[0]:
    basin_data = meteo_df_hist.xs(basin_name, level='Basin')
    basin_data = basin_data.T
    #basin_data = pd.concat(objs=[basin_data], axis=1, ignore_index=False)
    basin_data['Basin'] = basin_name
    #df1 = expand_df_by_weights(df=basin_data,variables=np.unique(vars))
    master_df_ls.append(basin_data)
meteo_df_hist_long = pd.concat(objs=master_df_ls, axis=0, ignore_index=True)
meteo_stats_hist = meteo_df_hist_long.groupby('Basin').describe(percentiles=[0.99, 0.95, 0.90, 0.75, 0.25, 0.1, 0.05]).T
meteo_stats_hist.to_csv(r'.\04_RESULTS\results_meteo\meteo_hist_stats.csv')


# Future period
ws_df2 = pd.read_csv(r'.\02_DATA\CMIP6_585\wind\basin_tc_windspd_stats.csv',index_col=0)
rain_df2 = pd.read_csv(r'.\02_DATA\CMIP6_585\rain\basin_tc_precipitation_stats.csv', index_col=0)
meteo_df2 = pd.concat(objs=[ws_df2, rain_df2], ignore_index=False, axis=1)
df_list2 = [meteo_df2[meteo_df2.index == i].T  for i in meteo_df2.index]
meteo_df_fut = pd.concat(objs=df_list2,axis=1)
vars = [''.join(x.split('_')[1:]) for x in meteo_df_fut.index]
tuples = [(basins[i], vars[i]) for i in range(len(vars))]
meteo_df_fut.index = pd.MultiIndex.from_tuples(tuples, names=['Basin', 'Variable'])
storm_weights = pd.read_csv(r'.\02_DATA\BiasCorrection\canesm_ssp585_weighted.csv', index_col=0, header=None)
storm_weights.columns = ['vmax','weight']

def expand_df_by_weights(df, variables):
    n_storms = 1543
    num_repeats = (df['weight'] * n_storms).round().astype(int)
    for v in variables:
        expanded_data = np.repeat(df[v], num_repeats)
        expanded_data = pd.DataFrame(expanded_data)

        if len(expanded_data) > n_storms:
            expanded_df = expanded_data.iloc[:n_storms]
        elif len(expanded_data) < n_storms:
            # If it's smaller, repeat some rows to match the total number of points
            expanded_data = expanded_data.sample(n=n_storms, replace=True, random_state=42)
        df[v] = expanded_data.values
    return df

master_df_ls = []
for basin_name in meteo_df_fut.index.levels[0]:
    basin_data = meteo_df_fut.xs(basin_name, level='Basin')
    basin_data = basin_data.T
    basin_data = pd.concat(objs=[basin_data, storm_weights], axis=1, ignore_index=False)
    basin_data['Basin'] = basin_name
    df1 = expand_df_by_weights(df=basin_data,variables=np.unique(vars))
    master_df_ls.append(df1)

meteo_df_fut_long = pd.concat(objs=master_df_ls, axis=0, ignore_index=True)
meteo_stats_fut = meteo_df_fut_long.groupby('Basin').describe(percentiles=[0.99, 0.95, 0.90, 0.75, 0.25, 0.1, 0.05]).T
meteo_stats_fut.to_csv(r'.\04_RESULTS\results_meteo\meteo_fut_stats_biasCorrected.csv')


''' 

PLOTTING METEO BELOW 

'''

vars = meteo_df_hist.xs('Domain', level='Basin').index.get_level_values('Variable').tolist()
variables = ['meanRRthresh', 'MeanTotPrecipMM', 'meanWSthresh', 'maxWS']
ylabels = ['Mean Rain Rate\n>5 mm/hr', 'Avg Total\nRainfall (m)',
           'Mean Wind Speed\n>5 m/s','Max Wind Speed\n(m/s)']
basin_order = ['LPD', 'CapeFear', 'OnslowBay', 'Neuse', 'Pamlico', 'Domain']

plot_meteo_boxplot = True
if plot_meteo_boxplot is True:
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5.5, 6), sharex=True, sharey=False)
    axes = axes.flatten()
    for i in range(len(variables)):
        v = variables[i]

        data_hist = meteo_df_hist.xs(v, level='Variable').T
        data_hist_melted = data_hist.melt(var_name='Basin', value_name='Variable')
        data_hist_melted['Period'] = 'Historic'

        # data_fut = meteo_df_fut.xs(v, level='Variable').T
        # data_fut_melted = data_fut.melt(var_name='Basin', value_name='Variable')
        data_fut = meteo_df_fut_long[[v, 'Basin']]
        data_fut.columns = ['Variable', 'Basin']
        data_fut['Period'] = 'Projected'

        df_combined = pd.concat([data_hist_melted, data_fut])
        df_combined['Basin'] = df_combined['Basin'].replace('LowerPeeDee', 'LPD')

        df_combined.reset_index(inplace=True)

        v = variables[i]
        ax = axes[i]
        if i < 2:
            logplease = True
        else:
            logplease=False
        # sns.boxplot(data=df_combined, x='Basin', y='Variable', hue='Period',ax=ax,
        #             log_scale=logplease, palette={'Historic': 'silver', 'Projected': 'grey'},
        #             gap=0.1, order=basin_order, flierprops={'marker': '.'})
        sns.violinplot(x='Basin', y='Variable', hue='Period', data=df_combined,
                       ax=ax,
                       order=basin_order,
                       split=True, fill=True, gap=0.1, width=1,
                       log_scale=False, density_norm='count', linecolor='none', zorder=1,
                       palette={'Historic': 'silver', 'Projected': 'grey'},
                       inner_kws=dict(box_width=5, whis_width=1.5, color="black"))
        ax.set_ylabel(ylabels[i])
        if i==0:
            ax.legend(frameon=False, title=None)
        else:
            ax.get_legend().set_visible(False)

    plt.tight_layout()
    plt.savefig(rf'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs\04_RESULTS\results_meteo\tc_meteo_biasCorrected.png',
                dpi=300)
