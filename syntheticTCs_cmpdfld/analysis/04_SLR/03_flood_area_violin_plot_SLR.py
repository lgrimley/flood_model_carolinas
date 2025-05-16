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

os.chdir(r'Z:\Data-Expansion\users\lelise\projects\Carolinas_SFINCS\Chapter3_SyntheticTCs')

# Read in the data catalog to get the model and basin geom
basins = cat.get_geodataframe(data_like=r'.\04_MODEL_OUTPUTS\masks\basins_shp\huc6_basins.shp')
basins = basins.to_crs(epsg=32617)
##################################################################################################################

def clean_up_slr_tables(filepath):
    df = pd.read_csv(filepath, index_col=0)
    df.index = df.index.str.replace(' ', '', regex=False)
    df['tc_id'] = [int(x.split('_')[0]) for x in df.index]
    mask = df.index.str.contains("SLR112cm")
    df['Period'] = 'NoSLR'
    df['Period'][mask] = 'SLR'
    df.index = df.index.str.replace("SLR112cm_", "", regex=False)

    df['basin'] = [str(x.split('_')[1]) for x in df.index]
    df['attr'] = [str(x.split('_')[-1]) for x in df.index]
    mapping = {'attr1': 'Coastal', 'attr2': 'Compound', 'attr3': 'Runoff'}
    df['attr'] = df['attr'].map(lambda x: mapping.get(x, 'Total'))

    # Reset index
    df.reset_index(inplace=True, drop=True)

    select_columns = ['basin', 'tc_id', 'Period', 'attr', 'Area_sqkm']
    df = df[select_columns]
    df_wide = df.pivot_table(
        index=["basin", "tc_id", "Period"],
        columns="attr",
        values="Area_sqkm"
    ).reset_index()

    return df_wide

filepath = r'.\04_MODEL_OUTPUTS\slr_runs\canesm_ssp585_SRL112cm\hmin0.05\canesm_slr_sbg20m_hmin0.05.csv'
df = clean_up_slr_tables(filepath=filepath)
outdir = r'.\05_ANALYSIS\05_SLR'

##################################################################################################################
scenarios = ['Runoff', 'Coastal', 'Compound', 'Total']
plot_SLR_violin = True
if plot_SLR_violin is True:
    font = {'family': 'Arial', 'size': 10}
    mpl.rc('font', **font)
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams["figure.autolayout"] = True

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(8, 6))  # Set figure size
    gs = fig.add_gridspec(len(scenarios), 3, width_ratios=[3, 1, 0], height_ratios=[1, 1, 1, 1])
    # Iterate through the scenarios and plot
    for i in range(len(scenarios)):
        scenario = scenarios[i]
        df_combined = df[['tc_id', 'basin', scenario, 'Period']]
        df_combined['basin'] = df_combined['basin'].replace('LowerPeeDee', 'LPD')
        df_long = pd.melt(df_combined, id_vars=['tc_id', 'basin', 'Period'], value_vars=[scenario],
                          var_name='data_type', value_name='data_value')
        df_long.dropna(axis=0, inplace=True)
        df_long = df_long.replace([np.inf, -np.inf], np.nan).dropna(subset=['data_value'])
        df_long_domain = df_long[df_long['basin'] == 'Domain']
        df_long_basin = df_long[df_long['basin'] != 'Domain']

        # Create axes using GridSpec
        yaxis_lim = [(-100, 6000), (-100, 3000), (-100, 2600), (-100, 8000)]
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
                       #w_method=0.2,
                       common_norm=True,
                       linecolor='none',
                       zorder=1,
                       palette={'NoSLR': 'grey', 'SLR': 'seagreen'},
                       inner=None,
                       cut=0
                       )

        # Loop to calculate and plot 90th percentiles
        for ii, basin in enumerate(basin_order):
            for jj, period in enumerate(['NoSLR', 'SLR']):
                subset = df_long_basin[(df_long_basin['basin'] == basin) & (df_long_basin['Period'] == period)]
                if not subset.empty:
                    p5 = np.percentile(subset['data_value'], 5)
                    p95 = np.percentile(subset['data_value'], 95)
                    median = np.percentile(subset['data_value'], 50)

                    # X-position adjustment for split violin
                    offset = -0.2 if period == 'NoSLR' else 0.2
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
            new_string = f'{basin}'
            new_xtick_labels.append(new_string)

        ax1.set_xticks(ticks=[0, 1, 2, 3, 4], labels=new_xtick_labels, rotation=0, fontsize=10, color='black')
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.75, color='lightgray', zorder=0)
        ax1.get_legend().set_visible(False)
        ax1.set_xlabel('')
        ax1.set_ylabel('Area (sq. km.)')
        ax1.set_title(scenarios[i])

        # Plot for the second subplot in the second column (1/3 space)
        ax2 = fig.add_subplot(gs[i, 1])  # Second column (1/3 of the space)
        basin_order = ['Domain']
        yaxis_lim = [(-100, 10000), (-100, 4500), (-100, 4500), (-100, 17000)]
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
                       palette={'NoSLR': 'grey', 'SLR': 'seagreen'},
                       inner=None,
                       cut=0
                       )

        # Loop to calculate and plot 90th percentiles
        for ii, basin in enumerate(basin_order):
            for jj, period in enumerate(['NoSLR', 'SLR']):
                subset = df_long_domain[(df_long_domain['basin'] == basin) & (df_long_domain['Period'] == period)]
                if not subset.empty:
                    p5 = np.percentile(subset['data_value'], 5)
                    p95 = np.percentile(subset['data_value'], 95)
                    median = np.percentile(subset['data_value'], 50)

                    # X-position adjustment for split violin
                    offset = -0.2 if period == 'NoSLR' else 0.2
                    x = ii + offset

                    # Draw whisker line from 5th to 95th percentile
                    ax2.plot([x, x], [p5, p95], color='black', lw=2, zorder=2)

                    # Draw a horizontal line for the median
                    ax2.plot([x - 0.05, x + 0.05], [median, median], color='black', lw=2, zorder=2)

        ax2.set_ylim(yaxis_lim[i])
        ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.75, color='lightgray', zorder=0)
        ax2.get_legend().set_visible(False)
        ax2.set_xlabel('')
        ax2.set_ylabel('Area (sq. km.)')
        ax2.set_title('')


    plt.tight_layout()
    plt.show()
    plt.savefig(rf'{outdir}\flood_hazard_violin_SLR.png',dpi=300)

##################################################################################################################

# Create a table of stats
df1 = df[df['Period'] == 'NoSLR']
df1 = df1.groupby('basin')[['Compound','Coastal', 'Runoff','Total']].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
df1 = df1.T

df2 = df[df['Period'] == 'SLR']
df2 = df2.groupby('basin')[['Compound','Coastal', 'Runoff','Total']].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
df2 = df2.T
df2.columns = [f'{s}_SLR' for s in df2.columns]

combined_stats = pd.concat(objs=[df1,df2], axis=1, ignore_index=False)
for c in df1.columns:
     combined_stats[f'{c}_Diff'] = combined_stats[f'{c}_SLR'] - combined_stats[c]

combined_stats.to_csv(rf'{outdir}\flood_extent_stats_sbg20m.csv')

